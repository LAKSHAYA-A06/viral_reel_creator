# virality_enhanced.py
from textblob import TextBlob

def predict_virality_enhanced(script, trend_score):
    """
    Returns separate predictions for views, likes, shares, saves.
    """
    sentiment = TextBlob(script).sentiment.polarity
    power_words = ["how", "why", "secret", "truth", "shocking", "viral", "amazing", "insane", "you", "watch", "wait"]
    hook_score = 1.0 if any(w in script[:150].lower() for w in power_words) else 0.5
    
    # Normalize trend_score (assume max around 10 million)
    normalized = min(trend_score / 10_000_000, 1.0)
    # Base virality percentage (0-100)
    virality_pct = (normalized * 0.4 + sentiment * 0.3 + hook_score * 0.3) * 100
    virality_pct = max(0, min(100, round(virality_pct, 1)))
    
    # Base views (max 50k for a reel)
    predicted_views = int((virality_pct / 100) * 50000)
    
    # Derived metrics with realistic ratios
    predicted_likes  = int(predicted_views * 0.05)   # 5% like rate
    predicted_shares = int(predicted_views * 0.03)   # 3% share rate
    predicted_saves  = int(predicted_views * 0.02)   # 2% save rate
    
    return {
        "virality_score": virality_pct,
        "predicted_views": predicted_views,
        "predicted_likes": predicted_likes,
        "predicted_shares": predicted_shares,
        "predicted_saves": predicted_saves,
        "sentiment": round(sentiment, 2),
        "hook_score": hook_score
    }