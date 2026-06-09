# AgenticEraHackathon

Agentic Era hackathon project (Google + Deloitte). A LangChain agent on Vertex AI Gemini 2.5 Flash that extracts structured data (Pydantic models) from Spanish public-administration documents and persists it to Cloud SQL.

## Architecture

- [aid_agent/main.py](aid_agent/main.py) — main agent. Pulls field definitions from Cloud SQL (`psycopg2`), reads preprocessed text from a GCS bucket (`google.cloud.storage`), builds a dynamic prompt, and calls `ChatVertexAI(model_name="gemini-2.5-flash")` via `langchain_google_vertexai`. Wraps the flow as a Google ADK `Agent` deployed through `vertexai.agent_engines`.
- [src/test.py](src/test.py) — standalone LangChain `PydanticOutputParser` experiments (`Certificado`, `ApplicationData`) used to iterate on the structured-output schema.
- [src/create_database.sql](src/create_database.sql) — Cloud SQL schema for `TRAMITE`, `DOCUMENTACION`, `EXPEDIENTE`, `ESTRUCTURA`, etc.
- [backend/](backend/) — FastAPI + Uvicorn service ([backend/Dockerfile](backend/Dockerfile), Python 3.11). `main.py` is currently empty — backend is scaffolding only.
- [data/](data/) — sample Spanish documents (DNI, certificado de hacienda, MOVES III, presupuesto).

## Build and Test

Python 3.11. [requirements.txt](requirements.txt) is a stub (only `python-dotenv`, `pandas`); install the real deps explicitly:

```bash
pip install langchain langchain-google-vertexai google-cloud-aiplatform google-cloud-storage google-adk psycopg2-binary fastapi uvicorn pydantic
python aid_agent/main.py
```

Backend container: `docker build -t aid-backend backend/ && docker run -p 8080:8080 aid-backend` (won't serve anything until `backend/main.py` is implemented).

## Conventions

- All prompts, field names, and log messages are in Spanish — keep that language when extending the agent.
- Structured output flows through Pydantic models; field metadata (`nombre_campo`, `descripcion`) is read from the `ESTRUCTURA`/`DOCUMENTACION` join, not hardcoded.

## Agent skills

Installable skills live under `.agents/skills/` (gitignored; restore with `make skills-restore`). Pinned versions are in [skills-lock.json](skills-lock.json).

- **deep-agents-memory** — consult when designing agent state, memory, or multi-step planning patterns for the ADK `Agent`.
- **fastapi-templates** — consult when implementing endpoints in [backend/](backend/) (currently scaffolding-only).
- **gemini-api-dev** — consult when integrating Vertex AI Gemini calls in [aid_agent/](aid_agent/) or [src/](src/) (prompts, structured output, retries).

## Pitfalls

- Requires Google Cloud auth (`GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth application-default login`) plus Vertex AI + Cloud SQL access in the hackathon project.
- [aid_agent/main.py](aid_agent/main.py) contains hardcoded Cloud SQL host, user, and password (`qwiklabs-gcp-...`) — hackathon throwaway creds; do not reuse and do not commit new secrets.
- `requirements.txt` pins versions that don't exist on PyPI (`pandas==3.0.2`) — treat it as a placeholder, not a lockfile.
- Design context (architecture diagram, participants guide) lives in [docs/](docs/) as `.pptx`/`.pdf`.

See [README.md](README.md) and [Notes.md](Notes.md) for hackathon context and full setup.
