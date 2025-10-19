# app.py
# ============================================================
# GLUU Hybrid FastAPI Backend - BLIP2 Core Logic & Product Report API
# ============================================================

import os, io, time, json, logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

# Import local inference helpers
from sagemaker_inference import (
    classify_with_sagemaker,
    extract_tags_with_sagemaker,
    local_blip_available
)
from tag_logic import PRODUCT_TAGS

# FastAPI setup
app = FastAPI(title="GLUU BLIP2 Async Inference API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gluu-blip2")

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
S3_INPUT_BUCKET = os.getenv("S3_INPUT_BUCKET", "")
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET", "")

# Classes and tags
OBJECT_LABELS = ["Shoe", "Watch", "Wallet", "Bag", "Purse", "Glasses", "Hat", "Phone", "Jacket", "Belt"]
MATERIAL_LABELS = ["Leather", "Canvas", "Suede", "Synthetic", "Rubber", "Metal", "Plastic", "Glass", "Wood", "Fabric"]

# Response model
class AsyncReportResponse(BaseModel):
    success: bool
    message: str
    product_type_job_uri: str = None
    material_job_uri: str = None
    tags_job_uri: str = None
    timetaken: float


@app.on_event("startup")
async def startup_log():
    logger.info("GLUU BLIP2 Async FastAPI service started.")


@app.post("/analyze-images", response_model=AsyncReportResponse)
async def analyze_images(files: List[UploadFile] = File(...)):
    """
    Accepts uploaded images, triggers SageMaker async inference, 
    and returns S3 URIs where outputs will be written.
    """
    start = time.time()

    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    images = []
    for file in files:
        try:
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if img.width < 100 or img.height < 100:
                raise ValueError("Image too small for analysis.")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Use the first image for product and material classification
    product_type_job_uri = classify_with_sagemaker(
        images[0],
        OBJECT_LABELS,
        endpoint_name=SAGEMAKER_ENDPOINT,
        region_name=AWS_REGION,
        input_bucket=S3_INPUT_BUCKET,
        output_bucket=S3_OUTPUT_BUCKET,
        use_local=False
    )

    material_job_uri = classify_with_sagemaker(
        images[0],
        MATERIAL_LABELS,
        endpoint_name=SAGEMAKER_ENDPOINT,
        region_name=AWS_REGION,
        input_bucket=S3_INPUT_BUCKET,
        output_bucket=S3_OUTPUT_BUCKET,
        use_local=False
    )

    # Tag extraction (async)
    tags_job_uri = extract_tags_with_sagemaker(
        images[0],
        "Shoe",  # placeholder; in production, use actual product_type result if synchronous
        endpoint_name=SAGEMAKER_ENDPOINT,
        region_name=AWS_REGION,
        input_bucket=S3_INPUT_BUCKET,
        output_bucket=S3_OUTPUT_BUCKET,
        use_local=False
    )

    elapsed = round(time.time() - start, 2)

    return AsyncReportResponse(
        success=True,
        message="Async inference jobs submitted successfully. Retrieve results from S3 output URIs.",
        product_type_job_uri=product_type_job_uri,
        material_job_uri=material_job_uri,
        tags_job_uri=tags_job_uri,
        timetaken=elapsed,
    )


@app.get("/status")
def get_status():
    """
    Simple health check endpoint.
    """
    return {"status": "running", "endpoint": SAGEMAKER_ENDPOINT, "region": AWS_REGION}


@app.get("/")
def root():
    return {"message": "GLUU BLIP2 Async Inference API is live"}
