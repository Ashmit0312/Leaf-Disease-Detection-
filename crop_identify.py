"""
Crop Identifier using Groq Vision API
Classes: Sugarcane, Wheat, Maize, Rice, Soybean

Install:
    pip install groq

Usage:
    python crop_identify.py --image path/to/image.jpg
"""

import argparse
import base64
import json
import os
import re

from groq import Groq


# ─── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY = "your_groq_api_key_here"

MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.1
MAX_TOKENS  = 100
USER_TEXT   = "Identify the crop in this image."

CLASSES = ["sugarcane", "wheat", "maize", "rice", "soybean"]

SYSTEM_PROMPT = f"""
You are an expert agricultural crop identifier.

Given an image, identify which crop is present from ONLY these classes:
{", ".join(CLASSES)}

Return STRICT JSON in this exact format:

{{
    "crop": "crop_name",
    "confidence": 0,
    "reason": "short explanation"
}}

Rules:
- confidence must be an integer from 0 to 100
- use ONLY the provided crop classes
- if uncertain, lower the confidence
- if none of the crops are visible, return:
{{
    "crop": "unknown",
    "confidence": 0,
    "reason": "crop not identifiable"
}}
"""


# ─── Image Loader ─────────────────────────────────────────────────────────────

def load_image_base64(image_path: str) -> tuple[str, str]:

    ext = os.path.splitext(image_path)[-1].lower()

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, mime


# ─── Identify Crop ────────────────────────────────────────────────────────────

def get_client() -> Groq:

    api_key = (
        GROQ_API_KEY
        if GROQ_API_KEY != "your_groq_api_key_here"
        else os.environ.get("GROQ_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY in the script or as an environment variable."
        )

    return Groq(api_key=api_key.strip())


def query_vlm(client: Groq, image_path: str) -> str:
    """Send one image with the fixed prompt and return the raw model text."""

    img_data, mime_type = load_image_base64(image_path)

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_data}"
                        },
                    },

                    {
                        "type": "text",
                        "text": USER_TEXT
                    }
                ]
            }
        ],

        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response.choices[0].message.content.strip()


def parse_response(raw: str) -> dict:
    """Parse the model's JSON answer into a validated {crop, confidence, reason}."""

    # Models sometimes wrap the JSON in ```json fences or add text around it
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    try:
        result = json.loads(match.group(0) if match else raw)

    except Exception:
        result = {
            "crop": "unknown",
            "confidence": 0,
            "reason": "Failed to parse model output"
        }

    # ─── Validation ───────────────────────────────────────────────────────────

    crop = str(result.get("crop", "unknown")).strip().lower()

    if crop not in CLASSES:
        crop = "unknown"

    confidence = result.get("confidence", 0)

    try:
        confidence = int(confidence)

    except Exception:
        confidence = 0

    confidence = max(0, min(100, confidence)) if crop != "unknown" else 0

    reason = result.get("reason", "")

    return {
        "crop": crop,
        "confidence": confidence,
        "reason": reason,
    }


def identify_crop(image_path: str) -> dict:

    client = get_client()

    print(f"[→] Loading image: {image_path}")

    raw = query_vlm(client, image_path)

    print("\n[DEBUG] Raw Model Output:")
    print(raw)

    return parse_response(raw)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Identify crop type from image"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to crop image"
    )

    args = parser.parse_args()

    result = identify_crop(args.image)

    print("\n── Result ───────────────────────────────")

    print(f"  Crop       : {result['crop'].upper()}")
    print(f"  Confidence : {result['confidence']}%")

    