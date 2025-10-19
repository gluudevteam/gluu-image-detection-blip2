# GLUU BLIP2 Async API – Postman Testing Guide

## Overview
This Postman collection allows you to test the FastAPI backend that integrates with the SageMaker Async BLIP2 inference pipeline.

## Setup
1. Open Postman and import:
   - `GLUU_BLIP2_Async_API.postman_collection.json`
   - Either `Local.postman_environment.json` or `AWS.postman_environment.json`
2. Select your environment.
3. Run `Health Check` to verify the backend is live.
4. Use `Analyze Images (Async)` to upload an image for analysis.

## Expected Response
```json
{
  "success": true,
  "message": "Async inference jobs submitted successfully. Retrieve results from S3 output URIs.",
  "product_type_job_uri": "s3://gluu-blip2-output/async-jobs/outputs/...json",
  "material_job_uri": "s3://gluu-blip2-output/async-jobs/outputs/...json",
  "tags_job_uri": "s3://gluu-blip2-output/async-jobs/outputs/...json",
  "timetaken": 2.15
}
