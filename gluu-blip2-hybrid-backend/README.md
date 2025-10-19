# GLUU BLIP2 Hybrid Product Summary Inference Backend

This repository contains the hybrid FastAPI backend for the GLUU BLIP2 Product Summary Inference project.  
It integrates BLIP2-based visual analysis (running via AWS SageMaker or locally) and OpenAI language summarization to automatically generate product condition reports from uploaded images.

---

## Overview

The system performs:
1. **Image Analysis** — Uses BLIP2 (via SageMaker or locally) to classify product type and material.
2. **Condition Tag Extraction** — Maps image features to product-specific condition tags.
3. **Score Mapping** — Assigns a normalized condition score based on detected flaws.
4. **Condition Summary Generation** — Calls OpenAI GPT model to produce a natural language summary.
5. **API Delivery** — Exposes `/analyze-images` endpoint via FastAPI for both local and cloud use.

The project aligns with the **GLUU - BLIP2 Product Summary Inference Architecture & Deployment Document**, supporting multi-stage deployment across Supabase, API Gateway, and AWS ECS.

---

## Architecture Summary

[Frontend / Supabase Edge]
↓
[API Gateway or ALB]
↓
[ECS Cluster running FastAPI container]
↓
[SageMaker Endpoint → BLIP2 Model] ←→ [OpenAI Summarization Service]

## Create Virtual Environment

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

## Install Dependencies
pip install -r requirements.txt

## Environment Configuration

Copy the example file:

cp .env.example .env


Then fill in your values:

OPENAI_API_KEY=your-openai-api-key
SAGEMAKER_ENDPOINT=your-sagemaker-endpoint
AWS_REGION=us-east-1
LOCAL_BLIP2=false
LOG_LEVEL=INFO


If you want to test locally without AWS:

LOCAL_BLIP2=true

## Running the Application
Local Development
uvicorn app:app --reload --port 8080


Visit:

http://127.0.0.1:8080/docs
 → Swagger UI

http://127.0.0.1:8080/
 → Health check

Upload test images under /analyze-images.

 Testing & Validation

## Option 1: Command-line Test

Run:

python test_local.py


This script sends a sample image to the local API and prints out:

Product type

Material

Condition tags

Score

Summary text

## Option 2: Postman Tests

Open Postman.

Import the collection and environments from /postman/:

GLUU_BLIP2_API_Tests.postman_collection.json

Local.postman_environment.json

AWS.postman_environment.json

Select the Local environment for local FastAPI testing or AWS for ECS deployment.

Run the Health Check and Analyze Single Image requests.

Verify:

Status code: 200 OK

JSON includes product_type, condition_report, and condition_score.

## AWS Deployment (CI/CD via GitHub)

Deployment is automated through GitHub Actions → .github/workflows/deploy.yml.

Steps

Push to the main branch in the Gluu Dev Team GitHub organization.

GitHub Actions:

Installs dependencies

Builds Docker image

Pushes image to AWS ECR

Updates ECS Service with new task definition

Secrets (OPENAI_API_KEY, SAGEMAKER_ENDPOINT) are loaded securely from AWS Secrets Manager.

## AWS Infrastructure Notes

ECS Cluster hosts the FastAPI container.

ALB (Application Load Balancer) routes /analyze-images requests to ECS tasks.

Route 53 + ACM (optional) provide HTTPS custom domain support.

Supabase Edge → API Gateway → ECS routing ensures frontend → backend connectivity.

## Project Structure
gluu-blip2-hybrid-backend/
│
├── app.py
├── sagemaker_inference.py
├── tag_logic.py
├── summarizer.py
├── test_local.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── secretsmanager_setup.sh
│
├── postman/
│   ├── GLUU_BLIP2_API_Tests.postman_collection.json
│   ├── Local.postman_environment.json
│   └── AWS.postman_environment.json
│
├── README.md
├── MANIFEST.txt
├── .github/workflows/deploy.yml
└── docs/
    ├── GLUU - BLIP2 Product Summary Inference Architecture & Deployment Document.pdf
    ├── ProjectHandoffDocument_Team6_Week10_Draft.pdf
    ├── Updated Potential Tasks.pdf
    └── README_REFERENCE.txt

## Local Secrets Setup (Optional)

You can automate AWS Secrets Manager setup using:

bash secretsmanager_setup.sh


This script will:

Prompt for SAGEMAKER_ENDPOINT and OPENAI_API_KEY

Create or update gluu-backend-secrets in AWS Secrets Manager

Ensure your GitHub Actions workflow retrieves secrets automatically

## Contributors
Name	Role	Responsibility
Joel Palmateer	Developer	FastAPI backend, CI/CD, AWS integration
Karen Smith	PM / Client Liaison	Coordination with Gluu Dev Team
Ayaan Gouse	Client Contact	GitHub org setup, AWS integration
Team 6 (Capella University)	Contributors	BLIP2 inference logic, documentation

## Version & Maintenance

Version: 1.0.0 (Hybrid BLIP2 API Release)

Maintainer: Joel Palmateer

Organization: Gluu Dev Team GitHub

Last Updated: October 2025



