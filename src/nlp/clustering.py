import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime
from transformers import pipeline
from sklearn.metrics import silhouette_score
import time # Import time for sleep
from src.core.config import CONFIG # Import CONFIG

class CSAIHistory:
    def __init__(self):
        self.history = []

    def append(self, data):
        self.history.append(data)

class StreamClusterer:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2', n_clusters=5):
        self.embedder = SentenceTransformer(embedding_model)
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=256, n_init='auto') # Increased n_init for more stable results
        self.csai_history = CSAIHistory()
        self.cluster_history = [] # Store cluster summaries and metadata
        self.geolocator = Nominatim(user_agent="news15", timeout=10) # Increased timeout
        self.summarization_pipeline = pipeline("summarization")
        self.topic_extraction_pipeline = pipeline("text-generation", model="facebook/bart-large-cnn")

    def add_batch(self, headlines):
        if not headlines:
            return {} # Return empty if no headlines

        embeddings = self.embedder.encode(headlines, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()

        # Handle cases where n_clusters might be greater than n_samples
        if len(headlines) < self.kmeans.n_clusters:
            # Assign all to a single cluster if not enough samples for multiple clusters
            cluster_labels = np.zeros(len(headlines), dtype=int)
        else:
            # Fit incrementally and then predict
            self.kmeans.partial_fit(embeddings_np)
            cluster_labels = self.kmeans.predict(embeddings_np)

        clustered_headlines = {}
        for i, label in enumerate(cluster_labels):
            if label not in clustered_headlines:
                clustered_headlines[label] = []
            clustered_headlines[label].append(headlines[i])
        return clustered_headlines

    def postprocess_cluster(self, cluster_headlines, timestamp_source_info=None):
        filtered_headlines = self.temporal_spatial_filter(cluster_headlines, timestamp_source_info)
        cluster_summary = self.summarize_cluster(filtered_headlines)
        cluster_topic = self.label_cluster_topic(cluster_summary)

        return {
            'summary': cluster_summary,
            'topic': cluster_topic,
            'headlines': filtered_headlines
        }

    def temporal_spatial_filter(self, headlines, timestamp_source_info):
        filtered_headlines = []
        if not CONFIG["processing"]["enable_spatial_filter"]:
            # If spatial filter is disabled, only apply temporal filter
            if timestamp_source_info:
                for i, headline in enumerate(headlines):
                    timestamp, _ = timestamp_source_info[i]
                    if (datetime.now() - timestamp).total_seconds() <= 24 * 3600:
                        filtered_headlines.append(headline)
            else:
                filtered_headlines = headlines
            return filtered_headlines

        # Proceed with spatial filtering if enabled
        if timestamp_source_info:
            austin_coords = None
            try:
                austin_location = self.geolocator.geocode("Austin, Texas")
                if austin_location:
                    austin_coords = (austin_location.latitude, austin_location.longitude)
                else:
                    print("Warning: Could not geocode 'Austin, Texas'. Skipping spatial filter.")
            except Exception as e:
                print(f"Error geocoding 'Austin, Texas': {e}. Skipping spatial filter.")

            if austin_coords:
                for i, headline in enumerate(headlines):
                    timestamp, source = timestamp_source_info[i]
                    try:
                        source_location = self.geolocator.geocode(source)
                        if source_location:
                            source_coords = (source_location.latitude, source_location.longitude)
                            distance = geodesic(austin_coords, source_coords).miles
                            # Basic temporal filter: keep headlines from the last 24 hours
                            if (datetime.now() - timestamp).total_seconds() <= 24 * 3600 and distance <= 100:
                                filtered_headlines.append(headline)
                        else:
                            print(f"Warning: Could not geocode source '{source}'. Skipping headline: {headline}")
                    except Exception as e:
                        print(f"Error geocoding source '{source}': {e}. Skipping headline: {headline}")
                    time.sleep(1) # Add a delay to respect API rate limits
            else:
                # If Austin couldn't be geocoded, skip spatial filtering and just apply temporal
                for i, headline in enumerate(headlines):
                    timestamp, _ = timestamp_source_info[i]
                    if (datetime.now() - timestamp).total_seconds() <= 24 * 3600:
                        filtered_headlines.append(headline)
        else:
            filtered_headlines = headlines
        return filtered_headlines

    def summarize_cluster(self, headlines):
        if not headlines:
            return "No headlines to summarize"
        text = " ".join(headlines)
        summary = self.summarization_pipeline(text, max_length=130, min_length=30, do_sample=False, truncation=True)[0]['summary_text']
        return summary

    def label_cluster_topic(self, cluster_summary):
        prompt = f"What is the main topic of the following text? {cluster_summary}\n\nTopic:"
        topic = self.topic_extraction_pipeline(prompt, max_length=50, min_length=5, do_sample=True, truncation=True)[0]['generated_text']
        return topic

    def validate_clustering(self, embeddings, labels):
        silhouette = silhouette_score(embeddings, labels)
        self.csai_history.append({'silhouette': silhouette})
        return silhouette

    def process_batch(self, headlines, timestamp_source_info=None):
        clustered_headlines = self.add_batch(headlines)
        cluster_results = {}
        embeddings = self.embedder.encode(headlines, convert_to_tensor=True).cpu().numpy()
        # Ensure labels are available, especially if clustering was skipped
        if len(headlines) < self.kmeans.n_clusters:
            labels = np.zeros(len(headlines), dtype=int)
        else:
            labels = self.kmeans.labels_

        for cluster_id, cluster_headlines_list in clustered_headlines.items():
            # Filter timestamp_source_info for the current cluster's headlines
            current_cluster_ts_info = []
            for hl in cluster_headlines_list:
                try:
                    idx = headlines.index(hl)
                    current_cluster_ts_info.append(timestamp_source_info[idx])
                except ValueError:
                    # This should not happen if headlines are correctly matched
                    pass
            cluster_results[cluster_id] = self.postprocess_cluster(cluster_headlines_list, current_cluster_ts_info)

        self.validate_clustering(embeddings, labels)
        self.cluster_history.append(cluster_results)
        return cluster_results

    def get_cluster_history(self):
        return self.cluster_history
