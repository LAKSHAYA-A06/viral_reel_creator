# =============================================================
# TRACK 1 - INFLUENCER INTELLIGENCE ENGINE
# FILE: functions.py
# DATASET COLUMNS USED:
#   rank, channel_info, influence_score, posts, followers,
#   avg_likes, 60_day_eng_rate, new_post_avg_like, total_likes, country
# =============================================================

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# HELPER: Convert "1.2m" → 1200000, "3.3k" → 3300, "1.39%" → 1.39
# ─────────────────────────────────────────────────────────────
def parse_number(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(',', '').replace('%', '')
    try:
        if s.lower().endswith('b'):
            return float(s[:-1]) * 1_000_000_000
        elif s.lower().endswith('m'):
            return float(s[:-1]) * 1_000_000
        elif s.lower().endswith('k'):
            return float(s[:-1]) * 1_000
        else:
            return float(s)
    except:
        return np.nan

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD & PREPARE DATASET
# ─────────────────────────────────────────────────────────────
def load_and_prepare(path="track1_influencer/data/influencers.csv"):
    print("📂 Loading dataset...")
    df = pd.read_csv(path, encoding='utf-8-sig')  # utf-8-sig handles the BOM character
    print(f"✅ Loaded {len(df)} influencers")
    print(f"📋 Columns: {list(df.columns)}\n")

    # Clean all number columns (they come as "1.2m", "3.3k", "1.39%")
    df['followers']         = df['followers'].apply(parse_number)
    df['avg_likes']         = df['avg_likes'].apply(parse_number)
    df['new_post_avg_like'] = df['new_post_avg_like'].apply(parse_number)
    df['total_likes']       = df['total_likes'].apply(parse_number)
    df['posts']             = df['posts'].apply(parse_number)
    df['influence_score']   = pd.to_numeric(df['influence_score'], errors='coerce')

    # engagement rate: remove "%" and convert
    df['60_day_eng_rate'] = df['60_day_eng_rate'].astype(str)\
                                .str.replace('%','').str.replace('NaN','').apply(parse_number)

    # Simulate missing columns we need for ML
    # posting_frequency = posts / account age (we estimate account age)
    df['posting_frequency'] = (df['posts'] / 365).clip(0.1, 20)  # posts per day approx

    # following count (not in dataset — simulate realistically)
    df['following'] = np.random.randint(200, 3000, len(df))

    # engagement rate as decimal
    df['engagement_rate'] = df['60_day_eng_rate'].fillna(
        (df['avg_likes'] / (df['followers'] + 1)) * 100
    )

    # Fill NaN
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['country'] = df['country'].fillna('Unknown')
    df['channel_info'] = df['channel_info'].fillna('unknown')

    print("✅ Dataset cleaned & ready!\n")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: AUTHENTICITY SCORE
# MODEL: Isolation Forest (detects fake/bot accounts)
# WHY: Finds accounts that behave unusually without needing labels
# ─────────────────────────────────────────────────────────────
def compute_authenticity_scores(df):
    print("🔍 Computing Authenticity Scores (Isolation Forest)...")

    feat = ['engagement_rate', 'avg_likes', 'followers', 'posting_frequency', 'influence_score']
    X = df[feat].fillna(0)

    # contamination=0.15 means we assume 15% accounts are suspicious
    iso = IsolationForest(contamination=0.15, random_state=42)
    df['anomaly'] = iso.fit_predict(X)   # -1 = suspicious, 1 = genuine

    def score_row(row):
        score = 100

        # Isolation Forest flagged it as bot/fake
        if row['anomaly'] == -1:
            score -= 35

        # Very low engagement = fake followers
        eng = row['engagement_rate']
        if eng < 0.3:
            score -= 30
        elif eng < 1.0:
            score -= 15
        elif eng > 20:
            # Unusually HIGH engagement can also be suspicious (pods)
            score -= 10

        # Very few new post likes vs avg likes = bought old likes
        like_ratio = row['new_post_avg_like'] / (row['avg_likes'] + 1)
        if like_ratio < 0.3:
            score -= 15

        return max(0, min(100, round(score, 1)))

    df['authenticity_score'] = df.apply(score_row, axis=1)
    print(f"✅ Done! Avg authenticity: {df['authenticity_score'].mean():.1f}\n")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: GROWTH PREDICTION SCORE
# MODEL: XGBoost Regressor
# WHY: Best ML model for tabular data, fast & accurate
# ─────────────────────────────────────────────────────────────
def compute_growth_scores(df):
    print("📈 Computing Growth Prediction Scores (XGBoost)...")

    feat = ['engagement_rate', 'posting_frequency', 'avg_likes',
            'new_post_avg_like', 'followers', 'influence_score']

    # Build realistic growth target:
    # High engagement + frequent posting + growing new_post_likes = high growth
    df['growth_target'] = (
        df['engagement_rate'] * 0.35 +
        df['posting_frequency'] * 0.25 +
        (df['new_post_avg_like'] / (df['avg_likes'] + 1)) * 30 * 0.25 +
        np.log1p(df['influence_score']) * 0.15
    ) + np.random.normal(0, 0.3, len(df))

    X = df[feat].fillna(0)
    y = df['growth_target']

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)

    raw = model.predict(X)
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['growth_potential_score'] = scaler.fit_transform(raw.reshape(-1, 1)).round(1).flatten()

    print(f"✅ Done! Avg growth score: {df['growth_potential_score'].mean():.1f}\n")
    return df, model


# ─────────────────────────────────────────────────────────────
# STEP 4: BRAND MATCHING
# MODEL: Sentence Transformers (NLP)
# WHY: Understands meaning of text — matches influencer niche to brand
# ─────────────────────────────────────────────────────────────
BRANDS = [
    {"name": "Nike",        "desc": "sports fitness gym running shoes athletic performance workout"},
    {"name": "Sephora",     "desc": "beauty makeup skincare cosmetics fashion lifestyle glamour"},
    {"name": "Apple",       "desc": "technology gadgets innovation premium lifestyle iPhone"},
    {"name": "Zomato",      "desc": "food delivery restaurant dining culinary cooking recipes"},
    {"name": "Nykaa",       "desc": "beauty skincare wellness fashion Indian cosmetics"},
    {"name": "boAt",        "desc": "audio technology earphones music youth gaming lifestyle"},
    {"name": "Myntra",      "desc": "fashion clothing style trends apparel streetwear"},
    {"name": "Puma",        "desc": "sports athleisure streetwear shoes fashion lifestyle"},
    {"name": "Louis Vuitton","desc": "luxury fashion premium bags high-end lifestyle brand"},
    {"name": "RedBull",     "desc": "sports energy extreme stunts music entertainment youth"},
    {"name": "Adidas",      "desc": "sports football soccer fitness shoes lifestyle brand"},
    {"name": "Netflix",     "desc": "entertainment movies series celebrity pop culture media"},
]

print("🤖 Loading NLP model for brand matching (~80MB, downloads once)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
brand_embeddings = embed_model.encode([b['desc'] for b in BRANDS])
print("✅ NLP model ready!\n")

# Map influencer names to niches (since our dataset has no niche column)
def guess_niche(name):
    name = str(name).lower()
    sports = ['cristiano','messi','neymar','kohli','lebron','mbappe','ronaldinho',
              'benzema','ramos','pogba','suarez','ibrahimovic','beckham','bale',
              'curry','ronaldo','salah','mahi','sachin','mbappe','nba','realmadrid',
              'fcbarcelona','manchesterunited','juventus','psg','liverpoolfc','433']
    fashion = ['kyliejenner','kimkardashian','khloekardashian','kendalljenner',
               'gigihadid','bellahadid','caradelevingne','haileybieber','zendaya',
               'victoriassecret','gucci','louisvuitton','dior','hm','adidasoriginals',
               'myntra','sonamkapoor']
    music   = ['arianagrande','beyonce','justinbieber','taylorswift','nickiminaj',
               'katyperry','dualipa','billieeilish','ladygaga','eminem','wizkhalifa',
               'snoopdogg','theweeknd','shawnmendes','camila_cabello','zayn',
               'travisscott','badbunnypr','jbalvin','daddyyankee','maluma']
    tech    = ['apple','nasa','marvel','marvelstudios','disney','natgeo']
    food    = ['zomato','nusr_et','buzzfeedtasty','cznburak']
    fitness = ['therock','kevinhart4real','chrishemsworth','danbilzerian']

    for n in sports:
        if n in name: return "sports fitness athlete performance"
    for n in fashion:
        if n in name: return "fashion beauty lifestyle luxury style"
    for n in music:
        if n in name: return "music entertainment pop culture celebrity"
    for n in tech:
        if n in name: return "technology innovation media entertainment"
    for n in food:
        if n in name: return "food cooking culinary restaurant lifestyle"
    for n in fitness:
        if n in name: return "fitness gym workout health lifestyle"
    return "lifestyle entertainment social media celebrity"

def brand_match(niche_text: str):
    vec = embed_model.encode([str(niche_text)])
    sims = cosine_similarity(vec, brand_embeddings)[0]
    top3 = sims.argsort()[-3:][::-1]
    return [(BRANDS[i]['name'], round(float(sims[i]) * 100, 1)) for i in top3]


# ─────────────────────────────────────────────────────────────
# STEP 5: RATEFLUENCER SCORE (Main ML Model)
# MODEL: XGBoost Regressor
# WHY WE CHOSE XGBOOST OVER OTHERS:
#   ✅ Best accuracy on structured/tabular data
#   ✅ Handles missing values automatically
#   ✅ Fast to train on small datasets (200 rows)
#   ✅ Industry standard for ranking/scoring tasks
#   ✅ Easy to explain in presentations
#   ❌ Random Forest — slower, less accurate
#   ❌ Logistic Regression — only for classification, not scores
#   ❌ Neural Networks — needs huge data (we only have 200 rows)
# ─────────────────────────────────────────────────────────────
SCORE_FEATURES = [
    'followers', 'engagement_rate', 'posting_frequency',
    'avg_likes', 'new_post_avg_like', 'influence_score',
    'authenticity_score', 'growth_potential_score'
]

def train_ratefluencer_model(df):
    print("🏆 Training Ratefluencer Score Model (XGBoost)...")

    df['target'] = (
        df['authenticity_score'] * 0.28 +
        df['growth_potential_score'] * 0.28 +
        df['engagement_rate'].clip(0, 15) * 2.5 +
        df['posting_frequency'] * 1.0 +
        np.log1p(df['influence_score']) * 1.5
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['target'] = scaler.fit_transform(df[['target']]).flatten()

    X = df[SCORE_FEATURES].fillna(0)
    y = df['target']

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    df['ratefluencer_score'] = model.predict(X).clip(0, 100).round(1)

    os.makedirs('track1_influencer/models', exist_ok=True)
    pickle.dump(model, open('track1_influencer/models/ratefluencer_model.pkl', 'wb'))
    print("💾 Model saved → track1_influencer/models/ratefluencer_model.pkl")

    top10 = df[['channel_info','ratefluencer_score','authenticity_score',
                'growth_potential_score','engagement_rate']]\
              .sort_values('ratefluencer_score', ascending=False).head(10)
    print("\n🏅 TOP 10 INFLUENCERS:")
    print(top10.to_string(index=False))
    print(f"\n✅ Ratefluencer Score done! Avg: {df['ratefluencer_score'].mean():.1f}\n")
    return df, model


# ─────────────────────────────────────────────────────────────
# STEP 6: REPORT FOR ONE INFLUENCER (called by Streamlit app)
# ─────────────────────────────────────────────────────────────
def get_influencer_report(influencer_data: dict, model) -> dict:
    row = pd.DataFrame([influencer_data])
    score = float(model.predict(row[SCORE_FEATURES].fillna(0))[0])
    score = round(max(0, min(100, score)), 1)

    auth   = round(float(influencer_data.get('authenticity_score', 70)), 1)
    growth = round(float(influencer_data.get('growth_potential_score', 50)), 1)
    niche  = influencer_data.get('niche', 'lifestyle')
    brands = brand_match(str(niche))

    if score >= 80:
        verdict = "🟢 High Value Creator"
        rec = "Excellent for brand partnerships. High ROI expected."
    elif score >= 60:
        verdict = "🟡 Mid Tier Creator"
        rec = "Good potential. Monitor growth before investing heavily."
    elif score >= 40:
        verdict = "🟠 Emerging Creator"
        rec = "Growing account. Good for low-budget campaigns."
    else:
        verdict = "🔴 Low Value / Risky"
        rec = "High fake follower risk or low engagement. Avoid."

    return {
        "ratefluencer_score": score,
        "authenticity_score": auth,
        "growth_potential_score": growth,
        "top_brand_matches": brands,
        "verdict": verdict,
        "recommendation": rec
    }
