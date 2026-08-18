from fastapi.testclient import TestClient

import main

REQUIRED = {
    ("GET", "/health"),
    ("GET", "/ready"),
    ("POST", "/api/v1/projects"),
    ("GET", "/api/v1/projects"),
    ("POST", "/api/v1/projects/{project_id}/documents"),
    ("GET", "/api/v1/projects/{project_id}/documents"),
    ("POST", "/api/v1/documents/{document_id}/ingest"),
    ("GET", "/api/v1/documents/{document_id}/status"),
    ("GET", "/api/v1/jobs/{job_id}"),
    ("POST", "/api/v1/conversations/{conversation_id}/messages"),
    ("POST", "/api/v1/projects/{project_id}/retrieve"),
    ("POST", "/api/v1/projects/{project_id}/evaluations"),
}

routes = {(method, route.path) for route in main.app.routes for method in route.methods or set()}
missing = REQUIRED - routes
assert not missing, f"missing routes: {sorted(missing)}"

with TestClient(main.app) as client:
    health = client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code == 200, ready.text
    assert ready.json()["status"] in {"ready", "not_ready"}

    created = client.post("/api/v1/projects", json={"name": "Smoke project", "description": "test"})
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]

    listed = client.get("/api/v1/projects")
    assert listed.status_code == 200, listed.text
    assert any(row["id"] == project_id for row in listed.json())

    documents = client.get(f"/api/v1/projects/{project_id}/documents")
    assert documents.status_code == 200 and documents.json() == [], documents.text

    invalid_upload = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert invalid_upload.status_code == 400, invalid_upload.text

    retrieve = client.post(
        f"/api/v1/projects/{project_id}/retrieve",
        json={"query": "What is the recommendation?", "k": 5, "method": "hybrid"},
    )
    assert retrieve.status_code == 400, retrieve.text

    evaluation = client.post(
        f"/api/v1/projects/{project_id}/evaluations",
        json={"query": "test", "expected_chunk_ids": []},
    )
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["grounded"] is False

print("backend smoke test: OK")
