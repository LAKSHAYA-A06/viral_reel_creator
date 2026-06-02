# script_genarator.py

from groq import Groq
from config import GROQ_API_KEY
from textblob import TextBlob

# ─────────────────────────────────────────
# Configure Groq Client
# ─────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)
# ✅ NEW - working 2026
MODEL = "llama-3.3-70b-versatile"  # free + fast


# ─────────────────────────────────────────
# Helper: call Groq
# ─────────────────────────────────────────
def ask_groq(prompt):
    response = client.chat.completions.create(
        model      = MODEL,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 500
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# STEP 1: Generate Reel Script
# ─────────────────────────────────────────
def generate_script(trend):
    prompt = f"""
You are a viral content creator.
Write a 30-second reel script about this trending topic:

Topic   : {trend['title']}
Channel : {trend['channel']}
Views   : {trend['views']:,}
Category: {trend['category']}

Format EXACTLY like this:
HOOK: (attention grabbing first 3 seconds)
STORY: (3 key points, 20 seconds)
CTA: (call to action, last 5 seconds)

Keep it punchy, short and viral.
"""
    return ask_groq(prompt)


# ─────────────────────────────────────────
# STEP 2: Generate Captions & Hashtags
# ─────────────────────────────────────────
def generate_captions(trend):
    prompt = f"""
Generate social media content for a reel about:
Topic: {trend['title']}
Views: {trend['views']:,}

Format EXACTLY like this:
INSTAGRAM: (max 150 chars, punchy caption)
HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5
LINKEDIN: (professional post, 100 words)
"""
    return ask_groq(prompt)


# ─────────────────────────────────────────
# STEP 3: Predict Virality
# ─────────────────────────────────────────
def predict_virality(script, trend_score):
    sentiment = TextBlob(script).sentiment.polarity

    power_words = ["how", "why", "secret", "truth",
                   "shocking", "viral", "amazing", "insane",
                   "you", "watch", "wait", "never"]
    hook_text  = script[:150].lower()
    hook_score = 1.0 if any(w in hook_text for w in power_words) else 0.5

    normalized = min(trend_score / 10000000, 1.0)

    virality = (
        (normalized * 0.4) +
        (sentiment  * 0.3) +
        (hook_score * 0.3)
    ) * 100

    virality        = max(0, min(100, round(virality, 1)))
    predicted_views = int((virality / 100) * 50000)

    return {
        "virality_score"  : virality,
        "predicted_views" : predicted_views,
        "sentiment"       : round(sentiment, 2),
        "hook_score"      : hook_score
    }


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    from youtube_trend import get_all_trends

    print("Fetching trends...\n")
    trends = get_all_trends()
    top    = trends[0]

    print(f"Topic: {top['title']}")
    print(f"Views: {top['views']:,}\n")
    print("=" * 50)

    print("\nGENERATING SCRIPT...")
    print("=" * 50)
    script = generate_script(top)
    print(script)

    print("\nGENERATING CAPTIONS...")
    print("=" * 50)
    captions = generate_captions(top)
    print(captions)

    print("\nPREDICTING VIRALITY...")
    print("=" * 50)
    virality = predict_virality(script, top["trend_score"])
    print(f"  Virality Score  : {virality['virality_score']}/100")
    print(f"  Predicted Views : {virality['predicted_views']:,}")
    print(f"  Sentiment       : {virality['sentiment']}")
    print(f"  Hook Score      : {virality['hook_score']}")