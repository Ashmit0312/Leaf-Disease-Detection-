"""
Crop Identifier using Groq Vision API
Classes: Sugarcane, Wheat, Maize, Rice, Soybean

Install:
    pip install groq

Usage:
    python crop_identify.py --image path/to/image.jpg
"""

import argparse, base64, os
from groq import Groq

# ─── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY = "your_groq_api_key_here"   # or set as env variable GROQ_API_KEY

CLASSES = ["sugarcane", "wheat", "maize", "rice", "soybean"]

SYSTEM_PROMPT = f"""You are an expert agricultural crop identifier.
Given an image, identify which crop is present from ONLY these 5 classes:
{", ".join(CLASSES)}.

Respond in this exact format:
Crop: <crop name>
Confidence: <High / Medium / Low>
Reason: <one line explanation>

If none of the 5 crops are visible, respond with:
Crop: unknown
Confidence: Low
Reason: <explanation>"""


# ─── Image Loader ─────────────────────────────────────────────────────────────

def load_image_base64(image_path: str) -> tuple[str, str]:
    ext = os.path.splitext(image_path)[-1].lower()
    mime = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, mime


# ─── Identify ─────────────────────────────────────────────────────────────────

def identify_crop(image_path: str) -> dict:
    api_key = GROQ_API_KEY if GROQ_API_KEY != "your_groq_api_key_here" \
              else os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Set GROQ_API_KEY in the script or as an environment variable.")

    client = Groq(api_key=api_key)

    print(f"[→] Loading image: {image_path}")
    img_data, mime_type = load_image_base64(image_path)

    print("[→] Sending to Groq vision model...")
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq vision model
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_data}"},
                    },
                    {
                        "type": "text",
                        "text": "Identify the crop in this image."
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()

    # Parse response
    result = {"raw": raw, "crop": "unknown", "confidence": "Low", "reason": ""}
    for line in raw.splitlines():
        if line.lower().startswith("crop:"):
            result["crop"] = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("confidence:"):
            result["confidence"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reason:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify crop using Groq Vision")
    parser.add_argument("--image", required=True, help="Path to crop image")
    args = parser.parse_args()

    result = identify_crop(args.image)

    print("\n── Result ───────────────────────────────")
    print(f"  Crop       : {result['crop'].upper()}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Reason     : {result['reason']}")
    print("─────────────────────────────────────────\n")
