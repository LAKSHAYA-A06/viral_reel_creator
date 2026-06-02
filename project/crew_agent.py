# crew_agent.py
import json
import os
from crewai import Agent, Task, Crew, Process
from trend_fetcher_enhanced import get_all_trends_enhanced
from script_genarator import generate_script, generate_captions
from voiceover import generate_voiceover
from thumbnail import generate_thumbnail
from virality_enhanced import predict_virality_enhanced

# Define CrewAI agents (each has a specific role)
trend_agent = Agent(
    role="Trend Spotter",
    goal="Find the hottest topics from YouTube and NewsAPI",
    backstory="Expert at scanning multiple sources for viral potential",
    allow_delegation=False,
    verbose=True
)

script_agent = Agent(
    role="Script Writer",
    goal="Write a punchy 30-60 sec reel script with HOOK/STORY/CTA",
    backstory="Top viral content creator",
    allow_delegation=False,
    verbose=True
)

voice_agent = Agent(
    role="Voiceover Artist",
    goal="Generate MP3 audio from script using gTTS",
    backstory="AI voice synthesizer",
    allow_delegation=False,
    verbose=True
)

thumbnail_agent = Agent(
    role="Thumbnail Designer",
    goal="Find or generate an eye-catching thumbnail image",
    backstory="Graphic designer using Unsplash",
    allow_delegation=False,
    verbose=True
)

virality_agent = Agent(
    role="Virality Analyst",
    goal="Predict views, likes, shares, saves",
    backstory="Data scientist specialized in social media metrics",
    allow_delegation=False,
    verbose=True
)

def run_full_agent_pipeline():
    """Orchestrate all agents to produce a complete viral reel package."""
    print("\n🚀 Starting CrewAI Viral Reel Pipeline...")
    
    # 1. Trend discovery
    print("🔍 Trend Agent: fetching trends...")
    trends = get_all_trends_enhanced(limit=5)
    if not trends:
        return {"error": "No trends found"}
    best_trend = max(trends, key=lambda x: x.get("trend_score", 0))
    print(f"✅ Best trend: {best_trend['title']} (score: {best_trend['trend_score']})")
    
    # 2. Generate script
    print("✍️ Script Agent: writing script...")
    script = generate_script(best_trend)
    
    # 3. Generate voiceover
    print("🎤 Voiceover Agent: generating MP3...")
    audio_file = generate_voiceover(script)
    
    # 4. Generate thumbnail
    print("🖼️ Thumbnail Agent: fetching image...")
    thumb_file = generate_thumbnail(best_trend["title"])
    
    # 5. Enhanced virality prediction
    print("📊 Virality Agent: predicting metrics...")
    virality = predict_virality_enhanced(script, best_trend["trend_score"])
    
    # 6. Generate captions (Instagram/LinkedIn)
    print("📝 Generating captions...")
    captions = generate_captions(best_trend)
    
    # Assemble final result
    result = {
        "trend": best_trend,
        "script": script,
        "audio_path": audio_file,
        "thumbnail_path": thumb_file if thumb_file else None,
        "virality": virality,
        "captions": captions
    }
    
    # Save to JSON
    os.makedirs("output", exist_ok=True)
    with open("output/final_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print("✅ Pipeline complete! Result saved to output/final_result.json")
    return result