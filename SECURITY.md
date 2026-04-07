# Security Policy — AgenticEraHackathon

## Reporting a Vulnerability
If you discover a security vulnerability, please email polcg10@gmail.com. Do not open a public issue.

## Security Considerations

### GCP Credential Management
- Google Cloud credentials (service account keys, OAuth tokens) must never be committed to version control.
- Store service account JSON files outside the repository. Reference them via `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
- Use Workload Identity Federation instead of service account keys when possible.
- Apply least-privilege IAM roles — grant only the specific GCP APIs this project needs.

### Cloud SQL Access
- Cloud SQL instances should use private IP or Cloud SQL Auth Proxy — never expose them to the public internet.
- Use IAM database authentication where supported instead of static username/password.
- Store database connection strings and passwords in environment variables or GCP Secret Manager, not in code.
- Restrict database user permissions to only the tables and operations needed.

### API Endpoint Security
- FastAPI endpoints process government documents. Ensure proper input validation on all request bodies.
- Add authentication (API keys, OAuth2, or JWT) before any deployment beyond local development.
- Enable rate limiting to prevent abuse.
- Use Pydantic models for strict request validation — reject malformed or unexpected input shapes.

### Document Data Handling
- Government documents may contain sensitive or personally identifiable information (PII).
- Do not log document contents. Redact PII from any logs or error reports.
- Implement data retention policies — delete processed documents when no longer needed.
- Ensure document uploads are validated (file type, size limits) before processing.

### Recommendations
- Use `.env` for local development and GCP Secret Manager for deployed environments.
- Run `gcloud auth revoke` after development sessions on shared machines.
- Keep all Python dependencies updated; run `pip audit` periodically.
- Review AI agent outputs before acting on them — treat as untrusted content.
