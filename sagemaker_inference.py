# sagemaker_inference.py (Async SageMaker version)
import boto3, json, logging, os, io, time
from PIL import Image
from typing import List

logger = logging.getLogger("sagemaker-inference")

# Try to import local BLIP2 components for fallback
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    from optimum.bettertransformer import Bettertransformer
    _LOCAL_AVAILABLE = True
except Exception:
    _LOCAL_AVAILABLE = False


def local_blip_available() -> bool:
    return _LOCAL_AVAILABLE and torch.cuda.is_available() or _LOCAL_AVAILABLE


# Helper to make a textual prompt for SageMaker
def build_classification_prompt(candidate_labels: List[str], label_type: str) -> str:
    labels_joined = ", ".join(candidate_labels)
    prompt = (
        f"Given the image, choose the best {label_type} from: {labels_joined}. "
        f"Answer with a single label only."
    )
    return prompt


def build_tags_prompt(product_type: str, tags: List[str]) -> str:
    joined = ", ".join(tags)
    prompt = (
        f"Analyze this {product_type.lower()} in the image. From the following condition tags, "
        f"which are visible: {joined}. Return only the tags that apply (comma separated)."
    )
    return prompt


# --- Async SageMaker Invocation ---
def invoke_sagemaker_async(image: Image.Image, prompt: str, endpoint_name: str, region_name: str,
                           input_bucket: str, output_bucket: str, prefix: str = "async-jobs") -> str:
    """
    Submit an asynchronous inference job to SageMaker.
    The image is uploaded to S3, and SageMaker writes output to an S3 output location.
    Returns the output S3 URI where the prediction will be written.
    """
    s3 = boto3.client("s3", region_name=region_name)
    sm = boto3.client("sagemaker-runtime", region_name=region_name)

    # Save image to S3 for async processing
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    input_key = f"{prefix}/inputs/{int(time.time())}.jpg"
    s3.upload_fileobj(buf, input_bucket, input_key)
    input_s3_uri = f"s3://{input_bucket}/{input_key}"

    logger.info(f"[invoke_sagemaker_async] Uploaded image to {input_s3_uri}")

    # Submit asynchronous request
    response = sm.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_s3_uri,
        CustomAttributes=prompt,
        OutputLocation=f"s3://{output_bucket}/{prefix}/outputs/"
    )

    output_uri = response.get("OutputLocation", f"s3://{output_bucket}/{prefix}/outputs/")
    logger.info(f"[invoke_sagemaker_async] Async job submitted. Output will appear at {output_uri}")
    return output_uri


# --- Public helpers used by app.py ---
def classify_with_sagemaker(image, candidates: List[str], endpoint_name: str,
                            region_name: str, input_bucket: str, output_bucket: str, use_local=False):
    """
    Return a single label string from candidates.
    For async SageMaker inference, returns the output S3 URI for monitoring.
    If use_local=True, attempt to run local BLIP2 model (fallback).
    """
    if use_local and local_blip_available():
        return _local_classify(image, candidates)

    prompt = build_classification_prompt(candidates, label_type=("type" if len(candidates) > 0 else "label"))
    output_uri = invoke_sagemaker_async(image, prompt, endpoint_name, region_name, input_bucket, output_bucket)
    logger.info(f"[classify_with_sagemaker] Async inference initiated for classification.")
    return output_uri


def extract_tags_with_sagemaker(image, product_type: str, endpoint_name: str,
                                region_name: str, input_bucket: str, output_bucket: str, use_local=False):
    """
    Return a list of selected tags (subset of PRODUCT_TAGS[product_type])
    For async SageMaker inference, returns the output S3 URI for monitoring.
    """
    from tag_logic import PRODUCT_TAGS
    tags = PRODUCT_TAGS.get(product_type, [])
    if not tags:
        return []

    if use_local and local_blip_available():
        return _local_extract_tags(image, product_type)

    prompt = build_tags_prompt(product_type, tags)
    output_uri = invoke_sagemaker_async(image, prompt, endpoint_name, region_name, input_bucket, output_bucket)
    logger.info(f"[extract_tags_with_sagemaker] Async inference initiated for tag extraction.")
    return output_uri


# --- Local BLIP2 fallback implementations (unchanged) ---
def _load_local_blip():
    global _local_processor, _local_model
    if not _LOCAL_AVAILABLE:
        raise RuntimeError("Local BLIP2 packages not available.")
    try:
        _local_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
        _local_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        try:
            _local_model = Bettertransformer.transformer(_local_model)
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(f"Failed to load local BLIP2: {e}")


def _local_classify(image: Image.Image, candidates: List[str]) -> str:
    if '_local_model' not in globals():
        _load_local_blip()
    for c in candidates:
        prompt = f"What is shown in the image? Is it a {c}?"
        inputs = _local_processor(images=image, text=prompt, return_tensors="pt").to(_local_model.device)
        with torch.no_grad():
            output = _local_model.generate(**inputs, max_new_tokens=16)
            answer = _local_processor.decode(output[0], skip_special_tokens=True).lower()
        if any(w in answer for w in ["yes", "definitely", "clearly"]):
            return c
    return candidates[0] if candidates else "Unknown"


def _local_extract_tags(image: Image.Image, product_type: str) -> List[str]:
    if '_local_model' not in globals():
        _load_local_blip()
    from tag_logic import PRODUCT_TAGS
    tags = PRODUCT_TAGS.get(product_type, [])
    if not tags:
        return []
    prompt = (
        f"Analyze this {product_type.lower()}. From the following condition tags, which are visible: "
        f"{', '.join(tags)}. Return only the relevant tags."
    )
    inputs = _local_processor(images=image, text=prompt, return_tensors="pt").to(_local_model.device)
    with torch.no_grad():
        output = _local_model.generate(**inputs, max_new_tokens=120)
        response = _local_processor.decode(output[0], skip_special_tokens=True).lower()
    selected = [t for t in tags if t.lower() in response]
    return selected
