"""Local demo API for inference"""
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.model import MAX_LENGTH, load_model_and_tokenizer, predict


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=256)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


@asynccontextmanager
async def lifespan(app):
    directory = Path(os.environ["MODEL_DIR"])
    if not directory.is_dir():
        raise ValueError("MODEL_DIR must be a saved local checkpoint")
    app.state.model, app.state.tokenizer = load_model_and_tokenizer(str(directory), os.environ.get("MODEL_DEVICE", "cpu"))
    # Load once at startup, not once per HTTP request.
    app.state.inference_lock = threading.Lock()
    yield


app = FastAPI(title="Japanese OCR Post-Correction", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/correct")
def correct_text(request: CorrectionRequest):
    # Reject overlong demo inputs rather than silently truncating a company name.
    token_ids = app.state.tokenizer(request.text)["input_ids"]
    if len(token_ids) > MAX_LENGTH:
        raise HTTPException(status_code=422, detail="Input exceeds 64 tokens")
    # Reject concurrent work instead of building an unbounded GPU/CPU queue.
    if not app.state.inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Model busy; try again", headers={"Retry-After": "1"})
    try:
        text = predict(app.state.model, app.state.tokenizer, request.text)
        return {"suggestion": text,
                "note": "Model suggestion; human review required"}
    except ValueError:
        raise HTTPException(status_code=503, detail="The model could not produce a complete suggestion")
    finally:
        app.state.inference_lock.release()
