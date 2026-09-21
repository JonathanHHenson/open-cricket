"""Optional HTTP service. Run: uvicorn labeljudge.server:app --host 127.0.0.1

Default backend: local HF model specified by LABELJUDGE_MODEL.
Custom backend: create_app(any compatible classification Runnable).
"""
import os
import hmac
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, ConfigDict, Field
from .integrations import ProbabilityUnavailableError


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=1)


def create_app(classifier=None):
    @asynccontextmanager
    async def lifespan(app):
        if classifier is None:
            from .hf import HuggingFaceBackend
            from .integrations import as_runnable
            app.state.classifier = as_runnable(HuggingFaceBackend(
                os.getenv("LABELJUDGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
                device=os.getenv("LABELJUDGE_DEVICE", "auto"),
                revision=os.getenv("LABELJUDGE_REVISION")))
        else:
            app.state.classifier = classifier
        yield

    app = FastAPI(title="LabelJudge", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.post("/classify")
    async def classify_http(request: Request, authorization: str | None = Header(default=None)):
        secret = os.getenv("LABELJUDGE_API_KEY")
        if secret and not hmac.compare_digest(authorization or "", "Bearer " + secret):
            raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            return await app.state.classifier.ainvoke(request.model_dump())
        except ProbabilityUnavailableError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()
