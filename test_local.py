"""
test_local.py
-----------------------------------------
Quick local test script for the GLUU BLIP2 Hybrid FastAPI backend.

Usage:
    python test_local.py --image sample_image.jpg
or just:
    python test_local.py
    (if you have sample_image.jpg in the same folder)
"""

import requests
import argparse
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8080/analyze-images"  # adjust if using different port

def run_test(image_path: Path):
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found.")
        return

    print(f"📤 Sending {image_path.name} to {API_URL} ...")
    with open(image_path, "rb") as img_file:
        files = {"files": (image_path.name, img_file, "image/jpeg")}
        try:
            response = requests.post(API_URL, files=files, timeout=120)
            print(f"✅ Status: {response.status_code}")
            try:
                data = response.json()
                print(json.dumps(data, indent=4))
            except Exception:
                print("⚠️  Non-JSON response:")
                print(response.text)
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GLUU BLIP2 FastAPI backend")
    parser.add_argument("--image", type=str, default="sample_image.jpg",
                        help="Path to image file for testing")
    args = parser.parse_args()

    run_test(Path(args.image))
