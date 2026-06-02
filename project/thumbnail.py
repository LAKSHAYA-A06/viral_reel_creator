# thumbnail.py
import requests
import os

def generate_thumbnail(topic, output_path="output/thumbnail.jpg"):
    """
    Generate a thumbnail using Pollinations.ai (free, no API key).
    The image is created from a prompt based on the trending topic.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a good prompt for thumbnail image
    prompt = f"trending social media reel thumbnail about {topic}, viral, eye-catching, bright colors, 16:9"
    
    # Pollinations image generation URL (no API key required)
    image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width=1280&height=720"
    
    try:
        print(f"🎨 Generating thumbnail from prompt: {prompt[:80]}...")
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Thumbnail saved to {output_path}")
            return output_path
        else:
            print(f"⚠️ Pollinations returned {response.status_code}, using fallback")
    except Exception as e:
        print(f"⚠️ Pollinations error: {e}")
    
    # Fallback: a simple colored placeholder (always works)
    fallback_url = "https://placehold.co/1280x720/7c6ef5/white?text=Viral+Reel+Thumbnail"
    img_data = requests.get(fallback_url).content
    with open(output_path, "wb") as f:
        f.write(img_data)
    return output_path