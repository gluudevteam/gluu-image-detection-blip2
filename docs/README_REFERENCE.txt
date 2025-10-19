- Hybrid inference architecture: SageMaker for model processing, FastAPI for logic & summarization

gluu-blip2-hybrid-backend/
│
├── app.py                                 # Hybrid FastAPI entrypoint (SageMaker + logic + OpenAI)
├── sagemaker_inference.py                 # SageMaker inference + local BLIP2 fallback
├── tag_logic.py                           # Object/material/tag definitions and scoring
├── summarizer.py                          # GPT-4o summarization for product reports
│
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Container definition for ECS deployment
│
├── .env.example                           # Example environment variables
├── .env                                   # (Optional local dev copy, not committed)
│
├── secretsmanager_setup.sh                # Script to set up AWS Secrets Manager secrets
│
├── postman_collection.json                # Postman API tests (GET /, POST /analyze-images)
│
├── README.md                              # Full documentation: setup, deployment, architecture
│
├── .github/
│   └── workflows/
│       └── deploy.yml                     # GitHub Actions CI/CD pipeline (build, push, deploy)
│
└── docs/
    ├── GLUU - BLIP2 Product Summary Inference Architecture & Deployment Document.pdf
    ├── ProjectHandoffDocument_Team6_Week10_Draft.pdf
    ├── Updated Potential Tasks.pdf
    └── README_REFERENCE.txt               # Optional metadata reference
