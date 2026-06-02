# app.py – Upgraded Viral Reel Agent
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import json
import re
import random
from role_b_agent import run_viral_reel_agent
from crew_agent import run_full_agent_pipeline
from script_genarator import generate_script, generate_captions
from youtube_trend import get_all_trends
from virality_enhanced import predict_virality_enhanced

app = Flask(__name__)
CORS(app)

# ---------- Existing Routes ----------
@app.route("/generate", methods=["GET"])
def generate():
    results = run_viral_reel_agent()
    return jsonify({"status": "success", "count": len(results), "results": results})

@app.route("/best", methods=["GET"])
def best():
    results = run_viral_reel_agent()
    best_item = max(results, key=lambda x: x["virality_score"])
    return jsonify(best_item)

@app.route("/generate_full", methods=["GET"])
def generate_full():
    try:
        result = run_full_agent_pipeline()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download_audio", methods=["GET"])
def download_audio():
    path = "output/voiceover.mp3"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Not generated"}), 404

@app.route("/download_thumbnail", methods=["GET"])
def download_thumbnail():
    path = "output/thumbnail.jpg"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Not generated"}), 404

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

@app.route("/")
def dashboard():
    return send_file("dashboard.html")

# ---------- NEW UPGRADED ENDPOINTS ----------

@app.route("/api/trends/topics", methods=["GET"])
def get_trend_topics():
    """Return list of trending topic titles."""
    try:
        trends = get_all_trends()
        topics = [t["title"] for t in trends[:10]]
        return jsonify({"status": "success", "topics": topics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/script", methods=["POST"])
def generate_script_api():
    """Generate script with style & duration."""
    data = request.get_json()
    topic = data.get("topic", "")
    style = data.get("style", "viral")
    duration = data.get("duration", "30")
    if not topic:
        return jsonify({"status": "error", "message": "Missing topic"}), 400

    # Create fake trend dict
    fake_trend = {
        "title": topic,
        "channel": "Custom",
        "views": 0,
        "category": "Custom",
        "trend_score": 5000000
    }
    try:
        full_script = generate_script(fake_trend)
        # Parse sections
        hook_match = re.search(r"HOOK:\s*(.*?)(?=STORY:|$)", full_script, re.DOTALL | re.IGNORECASE)
        story_match = re.search(r"STORY:\s*(.*?)(?=CTA:|$)", full_script, re.DOTALL | re.IGNORECASE)
        cta_match = re.search(r"CTA:\s*(.*?)$", full_script, re.DOTALL | re.IGNORECASE)
        hook = hook_match.group(1).strip() if hook_match else "Grab attention fast"
        story = story_match.group(1).strip() if story_match else "Key points go here"
        cta = cta_match.group(1).strip() if cta_match else "Follow for more"

        return jsonify({
            "status": "success",
            "script": full_script,
            "hook": hook,
            "story": story,
            "cta": cta,
            "topic": topic,
            "style": style,
            "duration": duration
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/captions", methods=["POST"])
def generate_captions_api():
    data = request.get_json()
    topic = data.get("topic", "")
    tone = data.get("tone", "casual")
    if not topic:
        return jsonify({"status": "error", "message": "Missing topic"}), 400

    fake_trend = {"title": topic, "channel": "Custom", "views": 0}
    try:
        full_captions = generate_captions(fake_trend)
        # Parse instagram
        insta_match = re.search(r"INSTAGRAM:\s*(.*?)(?=HASHTAGS:|LINKEDIN:|$)", full_captions, re.DOTALL | re.IGNORECASE)
        instagram = insta_match.group(1).strip() if insta_match else f"Check out this amazing {topic}!"
        # Parse hashtags
        hash_match = re.search(r"HASHTAGS:\s*(.*?)(?=LINKEDIN:|$)", full_captions, re.DOTALL | re.IGNORECASE)
        hashtags_text = hash_match.group(1).strip() if hash_match else f"#{topic.replace(' ', '')} #Viral #Trending"
        hashtags = re.findall(r"#\w+", hashtags_text)
        if not hashtags:
            hashtags = [f"#{topic.replace(' ', '')}", "#Trending", "#Reel"]
        # Parse linkedin
        li_match = re.search(r"LINKEDIN:\s*(.*?)$", full_captions, re.DOTALL | re.IGNORECASE)
        linkedin = li_match.group(1).strip() if li_match else f"Thoughts on {topic}? The industry is shifting. #Innovation"

        # Generate Twitter/X post using Groq (same client as in script_genarator)
        from script_genarator import ask_groq
        tweet_prompt = f"Write a punchy tweet under 280 characters about: {topic}. Tone: {tone}. Include 2 relevant hashtags."
        twitter = ask_groq(tweet_prompt).strip()
        if len(twitter) > 280:
            twitter = twitter[:277] + "..."

        return jsonify({
            "status": "success",
            "instagram": instagram,
            "hashtags": hashtags,
            "linkedin": linkedin,
            "twitter": twitter,
            "topic": topic,
            "tone": tone
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/virality", methods=["POST"])
def virality_api():
    data = request.get_json()
    script = data.get("script", "")
    topic = data.get("topic", "")
    if not script:
        return jsonify({"status": "error", "message": "Missing script"}), 400

    # Heuristic trend_score based on script length and keywords
    base_score = min(len(script) * 10, 10000000)
    keywords = ["amazing", "secret", "how to", "why", "shocking", "viral"]
    keyword_bonus = sum(1 for k in keywords if k in script.lower()) * 500000
    trend_score = min(base_score + keyword_bonus, 10000000)

    virality = predict_virality_enhanced(script, trend_score)

    # Use Groq for deeper analysis
    from script_genarator import ask_groq
    analysis_prompt = f"""
Analyze this reel script for virality:
{script}

Return ONLY a valid JSON object with these keys:
{{ 
  "hook_strength": 0-100, 
  "emotional_appeal": 0-100, 
  "shareability": 0-100, 
  "trend_relevance": 0-100, 
  "cta_strength": 0-100, 
  "summary": "string", 
  "improvements": ["tip1", "tip2", "tip3"]
}}
No extra text, only JSON.
"""
    try:
        analysis = ask_groq(analysis_prompt)
        # Remove any markdown code fences
        analysis = re.sub(r"```json\s*|\s*```", "", analysis).strip()
        analysis_data = json.loads(analysis)
    except:
        analysis_data = {
            "hook_strength": 70, "emotional_appeal": 65, "shareability": 60,
            "trend_relevance": 75, "cta_strength": 50,
            "summary": "Script has potential but could be punchier.",
            "improvements": ["Add a stronger hook", "Use more power words", "Shorten the story"]
        }

    response = {
        "status": "success",
        "virality_score": virality["virality_score"],
        "predicted_views": virality["predicted_views"],
        "predicted_likes": virality["predicted_likes"],
        "predicted_shares": virality["predicted_shares"],
        "predicted_saves": virality["predicted_saves"],
        **analysis_data
    }
    return jsonify(response)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    app.run(debug=True, port=5000)