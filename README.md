# Agentic Era Hackathon

Hackathon project for AI-driven processing of Spanish public administration documents. The system uses LLMs to extract structured data from certificates and administrative procedures, integrating with Google Cloud services.

## Overview

An AI agent built with LangChain and Google Vertex AI (Gemini 2.5 Flash) that:

- Parses Spanish administrative documents (certificates, procedures)
- Extracts structured data using Pydantic models
- Stores results in a Cloud SQL database with tables for procedures (`TRAMITE`), documentation (`DOCUMENTACION`), cases (`EXPEDIENTE`), and more
- Exposes functionality via a FastAPI backend

## Structure

```
├── aid_agent/
│   └── main.py              # AI agent using Vertex AI + LangChain
├── src/
│   ├── test.py              # LangChain structured output tests
│   └── create_database.sql  # Cloud SQL schema
├── backend/
│   └── Dockerfile           # FastAPI service (Python 3.11, uvicorn)
├── Notes.md                 # Project notes
└── requirements.txt         # Python dependencies
```

## Tech Stack

- **Python** with LangChain and Google Vertex AI
- **Google Cloud** (Cloud SQL, Cloud Storage)
- **FastAPI** + Uvicorn for the backend
- **Pydantic** for structured data extraction
