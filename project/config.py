# config.py
import os

# Required API keys (get free tiers from mentioned services)
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", " ")
YOUTUBE_API_KEY     = os.getenv("YOUTUBE_API_KEY", "")
NEWS_API_KEY        = os.getenv(" ")          # optional, if missing uses mock
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")   # optional, if missing uses placeholder

# Model for Groq
GROQ_MODEL = "llama-3.3-70b-versatile"