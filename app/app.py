# =============================================================
# FILE: app/app.py
# RUN: streamlit run app/app.py
# =============================================================

import streamlit as st
import pandas as pd
import pickle, sys, os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from track1_influencer.functions import (
    brand_match, get_influencer_report, SCORE_FEATURES, guess_niche, parse_number
)

st.set_page_config(page_title="Ratefluencer AI", page_icon="⚡", layout="wide")

@st.cache_resource
def load_model():
    p = "track1_influencer/models/ratefluencer_model.pkl"
    return pickle.load(open(p,'rb')) if os.path.exists(p) else None

@st.cache_data
def load_data():
    p = "track1_influencer/data/scored_influencers.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

model = load_model()
df    = load_data()

# ── HEADER ───────────────────────────────────────────────
st.markdown("# ⚡ Ratefluencer AI")
st.markdown("#### AI-Powered Influencer Intelligence Engine")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏅 Leaderboard", "🔍 Score an Influencer", "📊 Analytics"])

# ══════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🏅 Top Influencer Rankings")

    if df is not None:
        show_cols = [c for c in ['channel_info','ratefluencer_score','authenticity_score',
                                  'growth_potential_score','engagement_rate',
                                  'followers','country'] if c in df.columns]
        top = df[show_cols].sort_values('ratefluencer_score', ascending=False)\
                           .head(50).reset_index(drop=True)
        top.index += 1

        def color_score(val):
            if isinstance(val,(int,float)):
                if val >= 75: return 'background-color:#d4edda'
                elif val >= 50: return 'background-color:#fff3cd'
                else: return 'background-color:#f8d7da'
            return ''

        st.dataframe(
            top.style.applymap(color_score, subset=['ratefluencer_score']),
            use_container_width=True, height=500
        )

        # score distribution
        fig, ax = plt.subplots(figsize=(9,3))
        ax.hist(df['ratefluencer_score'], bins=20, color='#4A90D9', edgecolor='white')
        ax.set_xlabel('Ratefluencer Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.set_facecolor('#f9f9f9')
        st.pyplot(fig)
    else:
        st.error("⚠️ Run `python track1_influencer/train.py` first!")

# ══════════════════════════════════════════════════════════
# TAB 2 — SCORE ONE INFLUENCER
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Score Any Influencer")

    # Option A: pick from dataset
    if df is not None:
        names = df['channel_info'].tolist()
        picked = st.selectbox("Pick an influencer from the dataset:", ["-- Enter manually --"] + names)
    else:
        picked = "-- Enter manually --"

    st.markdown("**Or enter details manually:**")
    c1, c2 = st.columns(2)
    with c1:
        followers       = st.number_input("👥 Followers", 1000, 500_000_000, 500000, step=10000)
        avg_likes       = st.number_input("❤️ Avg Likes/Post", 0, 50_000_000, 25000, step=1000)
        new_post_likes  = st.number_input("🆕 New Post Avg Likes", 0, 50_000_000, 20000, step=1000)
        posts           = st.number_input("📝 Total Posts", 1, 100000, 500, step=10)
    with c2:
        influence_score = st.slider("🌟 Influence Score (0-100)", 0, 100, 75)
        eng_rate        = st.number_input("📊 Engagement Rate (%)", 0.0, 50.0, 2.5, step=0.1)
        niche           = st.selectbox("🏷️ Niche", ["sports fitness athlete",
                                                     "fashion beauty lifestyle",
                                                     "music entertainment celebrity",
                                                     "technology innovation",
                                                     "food cooking culinary",
                                                     "fitness gym workout"])
        country         = st.text_input("🌍 Country", "India")

    # Auto-fill if user picked from dataset
    if picked != "-- Enter manually --" and df is not None:
        row = df[df['channel_info'] == picked].iloc[0]
        followers       = int(row.get('followers', followers))
        avg_likes       = int(row.get('avg_likes', avg_likes))
        new_post_likes  = int(row.get('new_post_avg_like', new_post_likes))
        posts           = int(row.get('posts', posts))
        influence_score = float(row.get('influence_score', influence_score))
        eng_rate        = float(row.get('engagement_rate', eng_rate))
        niche           = guess_niche(picked)
        st.info(f"📌 Auto-filled data for **{picked}**")

    if st.button("⚡ Generate Score", use_container_width=True):
        if model is None:
            st.error("❌ Run `python track1_influencer/train.py` first!")
        else:
            posting_freq = (posts / 365)
            auth_score   = max(0, min(100,
                100 - (30 if eng_rate < 0.3 else 15 if eng_rate < 1.0 else 0)
                    - (15 if (new_post_likes / (avg_likes+1)) < 0.3 else 0)
            ))
            growth_score = min(100, eng_rate * 4 + posting_freq * 3 + influence_score * 0.3)

            data = {
                'followers': followers,
                'engagement_rate': eng_rate,
                'posting_frequency': posting_freq,
                'avg_likes': avg_likes,
                'new_post_avg_like': new_post_likes,
                'influence_score': influence_score,
                'authenticity_score': auth_score,
                'growth_potential_score': growth_score,
                'niche': niche,
            }

            report = get_influencer_report(data, model)

            st.markdown("---")
            st.markdown(f"## Report: **{picked if picked != '-- Enter manually --' else 'Custom Influencer'}**")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("⚡ Ratefluencer Score", f"{report['ratefluencer_score']}/100")
            m2.metric("🛡️ Authenticity",       f"{report['authenticity_score']}/100")
            m3.metric("📈 Growth Potential",    f"{report['growth_potential_score']}/100")
            m4.metric("💡 Engagement Rate",     f"{eng_rate:.2f}%")

            st.markdown(f"### {report['verdict']}")
            st.info(report['recommendation'])

            st.markdown("### 🤝 Best Brand Matches")
            for bname, bscore in report['top_brand_matches']:
                st.progress(int(bscore), text=f"**{bname}** — {bscore:.1f}% match")

            # bar chart
            fig, ax = plt.subplots(figsize=(7,3))
            labels = ['Ratefluencer','Authenticity','Growth']
            vals   = [report['ratefluencer_score'],
                      report['authenticity_score'],
                      report['growth_potential_score']]
            colors = ['#4A90D9','#27AE60','#E67E22']
            bars   = ax.barh(labels, vals, color=colors)
            ax.set_xlim(0,100)
            ax.set_xlabel('Score (0-100)')
            ax.set_title('Score Breakdown')
            for bar, v in zip(bars, vals):
                ax.text(v+1, bar.get_y()+bar.get_height()/2,
                        f'{v:.1f}', va='center', fontweight='bold')
            ax.set_facecolor('#f9f9f9')
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Dataset Analytics")
    if df is not None:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Influencers", len(df))
        c2.metric("Avg Ratefluencer",  f"{df['ratefluencer_score'].mean():.1f}")
        c3.metric("Avg Authenticity",  f"{df['authenticity_score'].mean():.1f}")
        c4.metric("Avg Growth",        f"{df['growth_potential_score'].mean():.1f}")

        hi  = len(df[df['ratefluencer_score']>=75])
        mid = len(df[(df['ratefluencer_score']>=50)&(df['ratefluencer_score']<75)])
        lo  = len(df[df['ratefluencer_score']<50])
        st.markdown("### 🏆 Influencer Tiers")
        t1,t2,t3 = st.columns(3)
        t1.metric("🟢 High Value (75+)", hi)
        t2.metric("🟡 Mid Tier (50-74)", mid)
        t3.metric("🔴 Low Value (<50)",  lo)

        # scatter
        fig, ax = plt.subplots(figsize=(10,5))
        sc = ax.scatter(
            df['engagement_rate'].clip(0,20),
            df['ratefluencer_score'],
            c=df['authenticity_score'], cmap='RdYlGn',
            alpha=0.6, s=50
        )
        plt.colorbar(sc, ax=ax, label='Authenticity Score')
        ax.set_xlabel('Engagement Rate (%)')
        ax.set_ylabel('Ratefluencer Score')
        ax.set_title('Engagement Rate vs Ratefluencer Score')
        ax.set_facecolor('#f9f9f9')
        st.pyplot(fig)

        # top 10 chart
        top10 = df.nlargest(10,'ratefluencer_score')[['channel_info','ratefluencer_score']]
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.barh(top10['channel_info'][::-1], top10['ratefluencer_score'][::-1], color='#4A90D9')
        ax2.set_xlabel('Ratefluencer Score')
        ax2.set_title('Top 10 Influencers')
        ax2.set_facecolor('#f9f9f9')
        st.pyplot(fig2)
    else:
        st.error("⚠️ Run `python track1_influencer/train.py` first!")
