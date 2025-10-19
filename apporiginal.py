from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from optimum.bettertransformer import Bettertransformer
import torch
import httpx
import json
import io
import time
import logging

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Setup FastAPI logging if not already done
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ollama-client")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP2
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

blip_model = Bettertransformer.transformer(blip_model)

# Labels
OBJECT_LABELS = ["Shoe", "Watch", "Wallet", "Bag", "Purse", "Glasses", "Hat", "Phone", "Jacket", "Belt"]
MATERIAL_LABELS = ["Leather", "Canvas", "Suede", "Synthetic", "Rubber", "Metal", "Plastic", "Glass", "Wood", "Fabric"]

# Condition tags
PRODUCT_TAGS = {
    "Shoe": ["creased toe box", "worn sole", "heel wear", "misaligned heel", "sole intact", "upper intact", "visible scuff marks", "clean surface", "smooth finish", "like new"],
    "Watch": ["shiny bezel", "scratched glass", "buckle rusted", "strap discolored", "dial clean", "metal polished", "clean surface", "dial dirty", "well maintained"],
    "Wallet": ["soft leather", "strap cracked", "zipper broken", "corners frayed", "lining clean", "logo faded"],
    "Bag": ["strap cracked", "zipper broken", "lining clean", "corners frayed", "logo faded", "torn material"],
    "Purse": ["zipper broken", "soft leather", "strap wrinkled", "pristine condition"],
    "Glasses": ["frame intact", "lens scratched", "hinge loose", "nose pad clean", "lens spotless"],
    "Hat": ["brim firm", "fabric stretched", "collar worn", "fabric smooth"],
    "Phone": ["screen pristine", "cracked screen", "back cover intact", "body dented", "buttons responsive", "frame scuffed"],
    "Jacket": ["buttons intact", "fabric smooth", "collar worn", "lining damaged"],
    "Belt": ["buckle polished", "buckle rusted", "holes stretched", "strap wrinkled", "leather supple", "strap smooth"]
}

class ReportResponse(BaseModel):
    success: bool
    product_type: str
    material: str
    condition_score: int
    condition_report: str
    tags: List[str]
    timetaken: float

def analyze_with_blip(image: Image.Image, label_list: List[str], label_type: str = "object") -> str:
    best_label = "Unknown"
    best_confidence = 0
    for label in label_list:
        prompt = f"What is shown in the image? Is the {label_type} a {label}?"
        inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(blip_model.device)
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=15)
            answer = blip_processor.decode(output[0], skip_special_tokens=True).lower()
        if any(word in answer for word in ["yes", "definitely", "clearly"]):
            return label
        elif any(word in answer for word in ["probably", "maybe", "likely"]):
            best_label = label
    return best_label

def extract_tags_batch(image: Image.Image, product_type: str) -> List[str]:
    tags = PRODUCT_TAGS.get(product_type, [])
    if not tags:
        return []

    prompt = (
        f"Analyze this {product_type.lower()}. From the following condition tags, which are visible: "
        f"{', '.join(tags)}. Return only the relevant tags."
    )
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(blip_model.device)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=100)
        response = blip_processor.decode(output[0], skip_special_tokens=True).lower()

    selected = [tag for tag in tags if tag.lower() in response]
    return selected

def map_condition_to_score(tags: List[str]) -> int:
    damage_tags = {
        "creased toe box", "worn sole", "heel wear", "misaligned heel", "visible scuff marks",
        "scratched glass", "buckle rusted", "strap discolored", "strap cracked", "zipper broken", "dial dirty",
        "corners frayed", "logo faded", "torn material", "strap wrinkled", "lens scratched",
        "hinge loose", "fabric stretched", "collar worn", "lining damaged", "cracked screen",
        "body dented", "frame scuffed", "holes stretched"
    }
    count = sum(1 for tag in tags if tag in damage_tags)
    return 10 if count == 0 else 8 if count == 1 else 6 if count == 2 else 4

async def call_phi4(product_type: str, material: str, tags: List[str]) -> str:
    tag_string = ", ".join(tags) if tags else "no visible flaws"
    prompt = (
        f"Write a 50-word condition summary for a {material} {product_type} based on these observations: {tag_string}. "
        f"Describe flaws naturally if present, or highlight good condition otherwise. Avoid exaggeration."
    )
    
    logger.info(f"[call_phi4] Sending prompt: {prompt}")
    
    timeout = httpx.Timeout(
        connect=60.0,
        read=120.0,
        write=120.0,
        pool=60.0,
    )
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi4:14b-q4_K_M", "prompt": prompt},
            )

            summary = ""

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                logger.info(f"[call_phi4] Raw line: {line}")

                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    done = data.get("done", False)

                    logger.info(f"[call_phi4] Chunk: {chunk!r} | Done: {done}")

                    summary += chunk

                except json.JSONDecodeError:
                    logger.warning(f"[call_phi4] JSON parse failed for line: {line!r}")
                    continue

            logger.info(f"[call_phi4] Final summary: {summary.strip()}")
            return summary.strip()

        except Exception as e:
            logger.exception(f"[call_phi4] Ollama request failed: {e}")
            print(f"Exception type: {type(e)}, details: {str(e)}")
            raise RuntimeError(f"Ollama request failed: {e}")

@app.on_event("startup")
async def startup_warmup():
    dummy = Image.new("RGB", (224, 224))
    analyze_with_blip(dummy, OBJECT_LABELS)

@app.post("/analyze-images", response_model=ReportResponse)
async def analyze_images(files: List[UploadFile] = File(...)):
    start_time = time.time()
    
    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    all_tags = set()
    images = []

