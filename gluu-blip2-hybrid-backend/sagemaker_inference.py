# sagemaker_inference.py
import boto3, json, logging, os, io
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

# helper to make a textual prompt we will send to SageMaker
def build_classification_prompt(candidate_labels: List[str], label_type: str) -> str:
    # instruct BLIP2 to answer whether each label applies; model should answer in plain text
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

def invoke_sagemaker_for_prompt(image: Image.Image, prompt: str, endpoint_name: str, region_name: str):
    """
    Invoke the SageMaker real-time endpoint with a prompt and image.
    The endpoint is expected to accept binary image body and `x-prompt` header or accept multipart JSON;
    here we send raw image bytes and include prompt in 'X-Amzn-SageMaker-Custom-Attributes' header if supported.
    Implementation may vary depending on your SageMaker container signature — adjust as needed.
    
    The invoke_sagemaker_for_prompt uses CustomAttributes to pass the prompt (some SageMaker containers accept it); 
    adjust this to match your SageMaker endpoint contract. If your SageMaker wrapper expects a JSON payload with 
    "inputs": {image: ..., "prompt": "..."}, modify invoke_sagemaker_for_prompt to send JSON 
    (e.g., multipart/form-data or application/json) per your container.
    """
    client = boto3.client("sagemaker-runtime", region_name=region_name)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    # many Hugging Face / custom endpoints take just image bytes and prompt in 'ContentType' or 'X-Amzn-...'
    # We'll use a simple approach: send prompt in 'X-Amzn-SageMaker-Custom-Attributes' header.
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=image_bytes,
        CustomAttributes=prompt   # some endpoints support this; if not, change to your endpoint contract
    )
    # response['Body'] is a streaming body
    result = response['Body'].read().decode('utf-8')
    return result

# Public helpers used by app.py
def classify_with_sagemaker(image, candidates: List[str], endpoint_name: str, region_name: str, use_local=False):
    """
    Return a single label string from candidates.
    If use_local=True attempt to run local BLIP2 model (fallback).
    """
    if use_local and local_blip_available():
        # local inference path
        return _local_classify(image, candidates)
    prompt = build_classification_prompt(candidates, label_type=("type" if len(candidates)>0 else "label"))
    raw = invoke_sagemaker_for_prompt(image, prompt, endpoint_name, region_name)
    # raw may be free-form; try to match to candidate labels (case-insensitive)
    lower = raw.lower()
    for c in candidates:
        if c.lower() in lower:
            return c
    # fallback: return first candidate or 'Unknown'
    return candidates[0] if candidates else "Unknown"

def extract_tags_with_sagemaker(image, product_type: str, endpoint_name: str, region_name: str, use_local=False):
    """
    Return a list of selected tags (subset of PRODUCT_TAGS[product_type])
    """
    from tag_logic import PRODUCT_TAGS
    tags = PRODUCT_TAGS.get(product_type, [])
    if not tags:
        return []
    if use_local and local_blip_available():
        return _local_extract_tags(image, product_type)
    prompt = build_tags_prompt(product_type, tags)
    raw = invoke_sagemaker_for_prompt(image, prompt, endpoint_name, region_name)
    # raw should contain comma separated tags; parse defensively
    raw_lower = raw.lower()
    selected = []
    for t in tags:
        if t.lower() in raw_lower:
            selected.append(t)
    return selected

# --- Local BLIP2 fallback implementations (only used if LOCAL_BLIP2 true and transformer libs installed) ---
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
