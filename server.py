# BenTech VoiceAI - OpenVoice V2 Backend v1.2.1 (Render-safe version)

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# OpenVoice imports
try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS
except ImportError:
    raise SystemExit(
        "OpenVoice or MeloTTS not installed.\n"
        "Install with:\n"
        "pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
        "pip install git+https://github.com/myshell-ai/OpenVoice.git"
    )

# Config

VOICES_DIR = Path("voices")
CHECKPOINTS_DIR = Path("checkpoints/converter")
KEYS_FILE = Path("keys.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bentech")

# API key system

class ApiKey(BaseModel):
    key: str
    name: str
    tier: str
    created_at: str
    revoked: bool = False


_keys: dict[str, ApiKey] = {}


def load_keys():
    global _keys

    if KEYS_FILE.exists():
        try:
            raw = json.loads(KEYS_FILE.read_text())

            for k, v in raw.items():
                _keys[k] = ApiKey(**v)

        except Exception as e:
            log.error(f"Failed to load keys.json: {e}")


load_keys()

bearer_scheme = HTTPBearer(auto_error=False)


def validate_key(raw_key: str) -> ApiKey:

    ak = _keys.get(raw_key)

    if ak is None:
        raise HTTPException(401, "Invalid API key")

    if ak.revoked:
        raise HTTPException(401, "Key revoked")

    return ak


async def require_key(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):

    raw = None

    if creds:
        raw = creds.credentials

    else:
        raw = request.query_params.get("api_key")

    if not raw:
        raise HTTPException(401, "API key required")

    return validate_key(raw)


# FastAPI app

app = FastAPI(title="BenTech VoiceAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "BenTech VoiceAI",
        "device": DEVICE,
        "time": datetime.now(timezone.utc),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def voices():
    return {
        "voices": [
            "Stephanie",
            "Ben"
        ]
    }


# Render-compatible startup

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False
)
