# app.py
# ============================================================
# GLUU Hybrid FastAPI Backend - BLIP2 Core Logic & Async Product Report API
# ============================================================

import os, io, time, json, logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports
from sagemaker_inference import (
    classify_with_sagemaker,
    extract_tags_with_sagemaker,
    local_blip_available
)
from tag_logic import PRODUCT_TAGS

# ------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------
app = FastAPI(title="GLUU BLIP2 Async Inference API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gluu-blip2")

# ------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
S3_BUCKET = os.getenv("S3_BUCKET", "gluu-project")
S3_INPUT_PREFIX = os.getenv("S3_INPUT_PREFIX", "input/")
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX", "output/")

# ------------------------------------------------------------
# Label Definitions
# ------------------------------------------------------------
OBJECT_LABELS = ["Shoe", "Watch", "Wallet", "Bag", "Purse", "Glasses", "Hat", "Phone", "Jacket", "Belt"]
MATERIAL_LABELS = ["Leather", "Canvas", "Suede", "Synthetic", "Rubber", "Metal", "Plastic", "Glass", "Wood", "Fabric"]

# ------------------------------------------------------------
# Response Model
# ------------------------------------------------------------
class AsyncReportResponse(BaseModel):
    success: bool
    message: str
    product_type_job_uri: str = None
    material_job_uri: str = None
    tags_job_uri: str = None
    timetaken: float


# ------------------------------------------------------------
# Startup Event
# ------------------------------------------------------------
@app.on_event("startup")
async def startup_log():
    logger.info("✅ GLUU BLIP2 Async FastAPI service started and ready for inference.")


# ------------------------------------------------------------
# Analyze Images Endpoint
# ------------------------------------------------------------
@app.post("/analyze-images", response_model=AsyncReportResponse)
async def analyze_images(files: List[UploadFile] = File(...)):
    """
    Accept uploaded images, trigger SageMaker async inference,
    and return the S3 URIs where results will appear.
    """
    start = time.time()

    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    images = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Use first image for classification & tagging
    image = images[0]

    try:
        product_type_uri = classify_with_sagemaker(image, OBJECT_LABELS, SAGEMAKER_ENDPOINT)
        material_uri = classify_with_sagemaker(image, MATERIAL_LABELS, SAGEMAKER_ENDPOINT)
        tags_uri = extract_tags_with_sagemaker(image, "Generic", SAGEMAKER_ENDPOINT)

        logger.info(f"🧠 Async Jobs Submitted:\n"
                    f"  Product Type → {product_type_uri}\n"
                    f"  Material → {material_uri}\n"
                    f"  Tags → {tags_uri}")

        elapsed = round(time.time() - start, 2)

        return AsyncReportResponse(
            success=True,
            message="Async inference jobs submitted successfully.",
            product_type_job_uri=product_type_uri,
            material_job_uri=material_uri,
            tags_job_uri=tags_uri,
            timetaken=elapsed
        )

    except Exception as e:
        logger.exception("❌ Failed to submit SageMaker async inference job.")
        raise HTTPException(status_code=500, detail=str(e))
