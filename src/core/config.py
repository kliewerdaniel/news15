CONFIG = {
    "ollama_api": {"base_url": "http://localhost:11434"},
    "models": {
        "summary_model": "mistral-small:24b-instruct-2501-q8_0",
        "broadcast_model": "mistral-small:24b-instruct-2501-q8_0",
        "embedding_model": "nomic-embed-text"
    },
    "processing": {
        "max_articles_per_feed": 80,
        "min_article_length": 100,
        "max_clusters": 50,
        "target_segments": 2500,
        "enable_spatial_filter": False # New configuration option
    },
    "output": {"max_broadcast_length": 900000000}
}
