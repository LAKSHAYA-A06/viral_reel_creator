import os

# Required API keys (get free tiers from mentioned services)
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
YOUTUBE_API_KEY     = os.getenv("YOUTUBE_API_KEY", "")
NEWS_API_KEY        = os.getenv("NEWS_API_KEY", "")          # optional, if missing uses mock
  # optional, if missing uses placeholder

# Model for Groq
