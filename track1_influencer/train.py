# =============================================================
# FILE: train.py
# RUN THIS ONCE TO TRAIN ALL MODELS
# COMMAND: python track1_influencer/train.py
# =============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from track1_influencer.functions import (
    load_and_prepare,
    compute_authenticity_scores,
    compute_growth_scores,
    train_ratefluencer_model,
    brand_match,
    guess_niche
)

print("=" * 55)
print("   RATEFLUENCER AI — TRACK 1 TRAINING PIPELINE")
print("=" * 55 + "\n")

# STEP 1: Load
df = load_and_prepare("track1_influencer/data/influencers.csv")

# STEP 2: Authenticity
df = compute_authenticity_scores(df)

# STEP 3: Growth
df, growth_model = compute_growth_scores(df)

# STEP 4: Ratefluencer Score
df, final_model = train_ratefluencer_model(df)

# STEP 5: Brand matches
print("🔗 Computing Brand Matches for each influencer...")
df['niche'] = df['channel_info'].apply(guess_niche)
df['top_brands'] = df['niche'].apply(lambda x: str(brand_match(x)))
print("✅ Brand matches done!\n")

# STEP 6: Save final CSV
out = "track1_influencer/data/scored_influencers.csv"
df.to_csv(out, index=False)
print(f"💾 Saved → {out}")

print("\n" + "=" * 55)
print("  ✅ ALL DONE! FILES CREATED:")
print("  📄 track1_influencer/data/scored_influencers.csv")
print("  🤖 track1_influencer/models/ratefluencer_model.pkl")
print("=" * 55)
print(f"\n  Influencers scored : {len(df)}")
print(f"  Avg Ratefluencer   : {df['ratefluencer_score'].mean():.1f}/100")
print(f"  Avg Authenticity   : {df['authenticity_score'].mean():.1f}/100")
print(f"  Avg Growth         : {df['growth_potential_score'].mean():.1f}/100")
print("\n  NEXT: streamlit run app/app.py")
print("=" * 55)
