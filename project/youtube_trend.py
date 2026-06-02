# youtube_trend.py

from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY

# ─────────────────────────────────────────
# STEP 1: Categories
# ─────────────────────────────────────────
CATEGORIES = {
    "28": "Science & Technology",
    "24": "Entertainment",
    "22": "People & Blogs",
    "23": "Comedy",
    "25": "News & Politics",
    "26": "How-to & Style",
    "27": "Education",
    "10": "Music"
}

REEL_CATEGORIES = ["28", "22", "26"]


# ─────────────────────────────────────────
# STEP 2: Connect to YouTube
# ─────────────────────────────────────────
def connect_youtube():
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    return youtube


# ─────────────────────────────────────────
# STEP 3: Fetch Trends Per Category
# ─────────────────────────────────────────
def get_trends_by_category(category_id, max_results=5):
    youtube = connect_youtube()

    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode="US",
        videoCategoryId=category_id,
        maxResults=max_results
    )
    response = request.execute()

    trends = []
    for video in response["items"]:
        snippet = video["snippet"]
        stats = video["statistics"]

        views    = int(stats.get("viewCount", 0))
        likes    = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))

        trend_score = round(
            (views * 0.5) + (likes * 0.3) + (comments * 0.2), 2
        )

        trends.append({
            "title"       : snippet["title"],
            "channel"     : snippet["channelTitle"],
            "tags"        : snippet.get("tags", [])[:5],
            "thumbnail"   : snippet["thumbnails"]["high"]["url"],
            "video_url"   : f"https://youtube.com/watch?v={video['id']}",
            "views"       : views,
            "likes"       : likes,
            "comments"    : comments,
            "trend_score" : trend_score,
            "category"    : CATEGORIES.get(category_id, "Unknown")
        })

    return trends


# ─────────────────────────────────────────
# STEP 4: Get ALL Trends (3 categories)
# ─────────────────────────────────────────
def get_all_trends():
    all_trends = []

    for cat_id in REEL_CATEGORIES:
        print(f"Fetching: {CATEGORIES[cat_id]}...")
        trends = get_trends_by_category(cat_id)
        all_trends.extend(trends)

    top_trends = sorted(
        all_trends,
        key=lambda x: x["trend_score"],
        reverse=True
    )[:5]

    return top_trends


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching YouTube Trends...\n")
    trends = get_all_trends()

    for i, t in enumerate(trends, 1):
        print(f"\n#{i} [{t['category']}]")
        print(f"  Title    : {t['title']}")
        print(f"  Channel  : {t['channel']}")
        print(f"  Views    : {t['views']:,}")
        print(f"  Likes    : {t['likes']:,}")
        print(f"  Comments : {t['comments']:,}")
        print(f"  Score    : {t['trend_score']:,}")
        print(f"  Tags     : {', '.join(t['tags'])}")
        print(f"  URL      : {t['video_url']}")