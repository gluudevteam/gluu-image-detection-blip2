# app.py
# ============================================================
# GLUU Hybrid FastAPI Backend - BLIP2 Core Logic & Product Report API
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from PIL import Image
import time
import io
import os
import logging

from sagemaker_inference import (
    classify_with_sagemaker,
    extract_tags_with_sagemaker,
    local_blip_available
)
from tag_logic import (
    OBJECT_LABELS,
    MATERIAL_LABELS,
    map_condition_to_score
)
from summarizer import summarize_result

# ------------------------------------------------------------
# App Setup
# ------------------------------------------------------------
app = FastAPI(title="GLUU BLIP2 Hybrid Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gluu-blip2-backend")

# ------------------------------------------------------------
# Response Model
# ------------------------------------------------------------
class ReportResponse(BaseModel):
    success: bool
    product_type: str
    material: str
    condition_score: int
    condition_report: str
    tags: List[str]
    timetaken: float

# ------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------
@app.get("/")
async def root():
    """
    Root endpoint for quick health check.
    """
    return {"message": "GLUU BLIP2 Hybrid Inference API is running"}

# ------------------------------------------------------------
# Analyze Images Endpoint
# ------------------------------------------------------------
@app.post("/analyze-images", response_model=ReportResponse)
async def analyze_images(files: List[UploadFile] = File(...)):
    """
    Core BLIP2 Product Report API
    - Accepts uploaded product images
    - Runs BLIP2 model via SageMaker (or local fallback)
    - Classifies product type, material, and condition
    - Generates condition summary report
    """
    start_time = time.time()

    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    logger.info(f"Processing {len(files)} uploaded images...")

    # --------------------------------------------------------
    # Read and Validate Images
    # --------------------------------------------------------
    images = []
    for file in files:
        try:
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if img.width < 100 or img.height < 100:
                raise ValueError("Image too small for analysis.")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # --------------------------------------------------------
    # Inference Path Decision
    # --------------------------------------------------------
    use_local = os.getenv("LOCAL_BLIP2", "false").lower() == "true" and local_blip_available()
    inference_type = "local BLIP2" if use_local else "SageMaker"
    logger.info(f"Using {inference_type} for inference pipeline")

    # --------------------------------------------------------
    # BLIP2 Inference – Classification & Tag Extraction
    # --------------------------------------------------------
    try:
        product_type = classify_with_sagemaker(images[0], OBJECT_LABELS, "object", use_local=use_local)
        material = classify_with_sagemaker(images[0], MATERIAL_LABELS, "material", use_local=use_local)
        logger.info(f"Detected: {product_type=} | {material=}")

        all_tags = set()
        for img in images:
            tags = extract_tags_with_sagemaker(img, product_type, use_local=use_local)
            all_tags.update(tags)

        unique_tags = sorted(all_tags)
        logger.info(f"Extracted tags: {unique_tags}")

    except Exception as e:
        logger.exception("BLIP2 inference failed")
        raise HTTPException(status_code=500, detail=f"BLIP2 inference failed: {e}")

    # --------------------------------------------------------
    # Condition Scoring & Summarization
    # --------------------------------------------------------
    try:
        condition_score = map_condition_to_score(unique_tags)
        condition_report = summarize_result(
            product_type,
            material,
            unique_tags,
            openai_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as e:
        logger.exception("Condition report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Report completed in {elapsed}s")

    # --------------------------------------------------------
    # Return Structured Response
    # --------------------------------------------------------
    return ReportResponse(
        success=True,
        product_type=product_type,
        material=material,
        condition_score=condition_score,
        condition_report=condition_report,
        tags=unique_tags,
        timetaken=elapsed
    )

# ------------------------------------------------------------
# Warmup Event (Optional)
# ------------------------------------------------------------
@app.on_event("startup")
async def warmup_model():
    """
    Optional warmup on API startup to reduce cold-start latency.
    """
    try:
        dummy_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        classify_with_sagemaker(dummy_img, OBJECT_LABELS, "object", use_local=False)
        logger.info("Warmup inference completed successfully.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
