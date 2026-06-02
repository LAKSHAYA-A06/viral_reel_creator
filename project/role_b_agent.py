# role_b_agent.py — Enhanced Viral Reel Agent (with multi-source trends)
from trend_fetcher_enhanced import get_all_trends_enhanced
from script_genarator import generate_script, generate_captions
from virality_enhanced import predict_virality_enhanced
import json
import os

def run_viral_reel_agent():
    print("\n" + "="*60)
    print("       🎬 VIRAL REEL AGENT (ENHANCED) — STARTING")
    print("="*60)

    # STEP 1: Fetch Trends (YouTube + NewsAPI)
    print("\n🔍 STEP 1: Fetching Multi-Source Trends...")
    trends = get_all_trends_enhanced(limit=5)

    print(f"✅ Found {len(trends)} trending topics!\n")
    for i, t in enumerate(trends, 1):
        print(f"  #{i} [{t.get('category', 'Unknown')}] [{t.get('source', 'unknown')}]")
        print(f"      Title  : {t['title'][:60]}")
        print(f"      Score  : {t.get('trend_score', 0):,.0f}")
        print(f"      Source : {t.get('source_type', 'youtube')}\n")

    # STEP 2: Process Top 3 Trends
    results = []
    for i, trend in enumerate(trends[:3], 1):
        print("\n" + "="*60)
        print(f"  🎯 PROCESSING TREND #{i}: {trend['title'][:50]}")
        print("="*60)

        script = generate_script(trend)
        print("\n✍️ Script:\n", script)

        captions = generate_captions(trend)
        print("\n📸 Captions:\n", captions)

        virality = predict_virality_enhanced(script, trend.get("trend_score", 0))
        print(f"\n📊 Virality:")
        print(f"  🔥 Score : {virality['virality_score']}/100")
        print(f"  👀 Views : {virality['predicted_views']:,}")
        print(f"  ❤️ Likes : {virality['predicted_likes']:,}")
        print(f"  🔁 Shares: {virality['predicted_shares']:,}")
        print(f"  💾 Saves : {virality['predicted_saves']:,}")

        results.append({
            "rank": i,
            "topic": trend["title"],
            "category": trend.get("category"),
            "trend_score": trend.get("trend_score"),
            "video_url": trend.get("video_url", ""),
            "script": script,
            "captions": captions,
            **virality
        })

    # STEP 3: Summary
    print("\n" + "="*60)
    print("        🏆 FINAL SUMMARY — TOP REELS")
    print("="*60)
    best = max(results, key=lambda x: x["virality_score"])
    for r in results:
        star = "🥇" if r["rank"] == best["rank"] else f"#{r['rank']}"
        print(f"\n  {star} {r['topic'][:50]}")
        print(f"     Virality: {r['virality_score']}/100 | Views: {r['predicted_views']:,}")

    # STEP 4: Save JSON
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n💾 Results saved to output/results.json")
    print("\n✅ ENHANCED VIRAL REEL AGENT COMPLETE!\n")
    return results

if __name__ == "__main__":
    run_viral_reel_agent()