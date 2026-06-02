# trend_fetcher_enhanced.py
import requests
from datetime import datetime, timezone
from config import YOUTUBE_API_KEY, NEWS_API_KEY
from youtube_trend import get_trends_by_category

def fetch_youtube_trends_enhanced(category_ids=["28","22","26"], max_per_category=3):
    """Fetch YouTube trends from given categories, return limited per category."""
    all_trends = []
    for cat_id in category_ids:
        trends = get_trends_by_category(cat_id, max_results=max_per_category)
        all_trends.extend(trends)
    return all_trends

def fetch_news_trends():
    """Fetch top headlines from NewsAPI (Tech, Business, Science)."""
    if not NEWS_API_KEY:
        # Mock data if no API key (so demo still works)
        return [
            {
                "title": "AI Breakthrough: New Model Beats Human Experts",
                "source": "mock_news",
                "category": "Technology",
                "views": 125000,
                "trend_score": 89000,
                "channel": "TechCrunch (mock)",
                "video_url": "https://example.com/article1",
                "source_type": "news"
            },
            {
                "title": "Federal Reserve Signals Rate Cut",
                "source": "mock_news",
                "category": "Finance",
                "views": 98000,
                "trend_score": 76000,
                "channel": "Bloomberg (mock)",
                "video_url": "https://example.com/article2",
                "source_type": "news"
            }
        ]
    
    url = "https://newsapi.org/v2/top-headlines"
    categories = {"technology": "Technology", "business": "Business", "science": "Science"}
    articles = []
    
    for cat_key, cat_name in categories.items():
        resp = requests.get(url, params={
            "apiKey": NEWS_API_KEY,
            "category": cat_key,
            "country": "us",
            "pageSize": 3
        })
        if resp.status_code == 200:
            data = resp.json()
            for art in data.get("articles", []):
                # Compute a simple trend_score based on recency (if publishedAt exists)
                pub_str = art.get("publishedAt")
                if pub_str:
                    pub_time = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    hours_ago = (now - pub_time).total_seconds() / 3600
                    recency_score = max(0, 10000 / (hours_ago + 1))
                else:
                    recency_score = 5000
                
                articles.append({
                    "title": art["title"],
                    "source": "newsapi",
                    "category": cat_name,
                    "views": 0,  # NewsAPI doesn't give views
                    "trend_score": recency_score,
                    "channel": art["source"]["name"],
                    "video_url": art["url"],
                    "source_type": "news",
                    "published_at": art.get("publishedAt", "")
                })
    return articles

def get_all_trends_enhanced(limit=8):
    """Combine YouTube and News trends, sort by trend_score, return top N."""
    youtube_trends = fetch_youtube_trends_enhanced()
    news_trends = fetch_news_trends()
    all_trends = youtube_trends + news_trends
    # Sort descending by trend_score
    all_trends.sort(key=lambda x: x.get("trend_score", 0), reverse=True)
    return all_trends[:limit]