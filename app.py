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
