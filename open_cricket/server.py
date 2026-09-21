"""Optional HTTP service. Run: uvicorn open_cricket.server:app --host 127.0.0.1

Default backend: local HF model specified by OPEN_CRICKET_MODEL.
Custom backend: create_app(any compatible classification Runnable).
"""

import hmac
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi import Request as HTTPRequest

from .systemone import ModelsResponse, SystemOneRequest, SystemOneResponse


def create_app(classifier=None, *, model_name=None):
    served_model = model_name or (
        os.getenv("OPEN_CRICKET_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        if classifier is None
        else "open-cricket-custom"
    )

    def authenticate(authorization: str | None = Header(default=None)):
        secret = os.getenv("OPEN_CRICKET_API_KEY")
        if secret and not hmac.compare_digest(
            (authorization or "").encode("utf-8"), ("Bearer " + secret).encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @asynccontextmanager
    async def lifespan(app):
        if classifier is None:
            from .sdk import LocalClient

            app.state.classifier = LocalClient(
                model=served_model,
                runtime=os.getenv("OPEN_CRICKET_BACKEND", "hf"),
                device=os.getenv("OPEN_CRICKET_DEVICE", "auto"),
                revision=os.getenv("OPEN_CRICKET_REVISION"),
                mode=os.getenv("OPEN_CRICKET_MODE", "answer_codes"),
            )
        else:
            app.state.classifier = classifier
        yield

    app = FastAPI(title="Open Cricket", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_metadata(request: HTTPRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            response.headers["x-request-id"] = str(uuid4())
            response.headers["x-open-cricket-confidence-method"] = "normalized-entropy"
        return response

    @app.get("/v1/models", response_model=ModelsResponse, dependencies=[Depends(authenticate)])
    def models():
        return {
            "models": [
                {
                    "name": name,
                    "description": f"Open Cricket local inference using {served_model}; structured decision API.",
                    "release_date": "2026-09-21",
                }
                for name in (served_model,)
            ]
        }

    @app.post(
        "/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(authenticate)]
    )
    async def system_one(request: SystemOneRequest):
        if request.model != served_model:
            raise HTTPException(status_code=422, detail="Unknown model; see GET /v1/models")
        try:
            return await app.state.classifier.ainvoke(request.model_dump(mode="json"))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/health")
    def health():
        return {"status": "ready"}

    return app


app = create_app()
