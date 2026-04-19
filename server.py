"""
BenTech VoiceAI — OpenVoice V2 Backend  v1.2.0
================================================
Now with full API key authentication + rate limiting.

KEY MANAGEMENT (CLI):
  python manage_keys.py create --name "Acme Corp" --tier pro
  python manage_keys.py list
  python manage_keys.py revoke btk_xxxx

HOW AUTH WORKS:
  Every request must include the key in the Authorization header:
    Authorization: Bearer btk_<key>
  Or as a query param (less preferred):
    ?api_key=btk_<key>

  Three tiers:
    admin  — unlimited, can manage other keys, access /admin/* routes
    pro    — 500 requests/day, all voices, upload allowed
    basic  — 100 requests/day, Stephanie + Ben only, no upload

KEYS FILE:
  Stored in keys.json (gitignore this file in production).
  In Modal/Docker, mount it as a secret volume or use the env var approach below.

ENV VAR OVERRIDE:
  Set BENTECH_ADMIN_KEY=btk_yourkey to always have one hardcoded admin key,
  even if keys.json is missing. Useful for first boot.
"""

import os
import io
import json
import time
import uuid
import shutil
import hashlib
import logging
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Annotated

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── OpenVoice imports ──────────────────────────────────────────────────────────
try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS
except ImportError:
    raise SystemExit(
        "\n❌ OpenVoice / MeloTTS not installed.\n"
        "Run: pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
        "     pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
    )

# ── Config ─────────────────────────────────────────────────────────────────────
VOICES_DIR      = Path("voices")
CHECKPOINTS_DIR = Path("checkpoints/converter")
CACHE_DIR       = Path(".cache/bentech/audio")
KEYS_FILE       = Path("keys.json")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_MAX_MB    = 500

# Tier daily request limits  (None = unlimited)
TIER_LIMITS: dict[str, Optional[int]] = {
    "admin": None,
    "pro":   500,
    "basic": 100,
}

# Which tiers can upload new voice clones
UPLOAD_ALLOWED_TIERS = {"admin", "pro"}

# Which tiers can access admin management routes
ADMIN_TIERS = {"admin"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("bentech")

# ── Global model state ─────────────────────────────────────────────────────────
_tone_converter: Optional[ToneColorConverter] = None
_melo_models:    dict[str, MeloTTS] = {}
_speaker_embeddings: dict[str, torch.Tensor] = {}

SPEAKER_CONFIG = {
    "Stephanie": {
        "reference_wav": VOICES_DIR / "stephanie_reference.wav",
        "melo_language": "EN",
        "melo_speaker":  "EN-US",
        "description":   "Bright, expressive female voice",
    },
    "Ben": {
        "reference_wav": VOICES_DIR / "ben_reference.wav",
        "melo_language": "EN",
        "melo_speaker":  "EN-Default",
        "description":   "Deep, clear male voice",
    },
}

EMOTION_SPEED = {
    "Natural": 0.88, "Warm": 0.84, "Professional": 0.92,
    "Excited": 1.10, "Calm": 0.80, "Sad": 0.78,
}

ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


# ══════════════════════════════════════════════════════════════════════════════
# API KEY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class ApiKey(BaseModel):
    key:         str            # btk_<32 random hex chars>
    name:        str            # human label, e.g. "Acme Corp – prod"
    tier:        str            # admin | pro | basic
    created_at:  str            # ISO 8601
    last_used:   Optional[str]  # ISO 8601 or None
    revoked:     bool = False
    usage_today: int  = 0       # requests today (resets at UTC midnight)
    usage_date:  str  = ""      # YYYY-MM-DD of last increment
    total_usage: int  = 0       # lifetime request count


def _load_keys() -> dict[str, ApiKey]:
    """Load keys from keys.json. Returns dict keyed by key string."""
    keys: dict[str, ApiKey] = {}

    # Always honour the hardcoded admin env var
    env_key = os.environ.get("BENTECH_ADMIN_KEY")
    if env_key:
        keys[env_key] = ApiKey(
            key=env_key,
            name="env-admin",
            tier="admin",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_used=None,
        )

    if KEYS_FILE.exists():
        try:
            raw = json.loads(KEYS_FILE.read_text())
            for k, v in raw.items():
                keys[k] = ApiKey(**v)
        except Exception as e:
            log.error(f"Failed to parse {KEYS_FILE}: {e}")

    return keys


def _save_keys(keys: dict[str, ApiKey]) -> None:
    """Persist keys to keys.json, skipping env-admin."""
    to_save = {k: v.model_dump() for k, v in keys.items() if v.name != "env-admin"}
    KEYS_FILE.write_text(json.dumps(to_save, indent=2))


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# In-memory key store — loaded at startup, mutated on create/revoke
_keys: dict[str, ApiKey] = {}


def _validate_key(raw_key: str) -> ApiKey:
    """
    Validate a key string. Raises HTTPException on any failure.
    On success, updates last_used + usage counters and persists.
    """
    ak = _keys.get(raw_key)

    if ak is None:
        log.warning(f"Auth failed: unknown key {raw_key[:12]}…")
        raise HTTPException(401, "Invalid API key")

    if ak.revoked:
        log.warning(f"Auth failed: revoked key '{ak.name}'")
        raise HTTPException(401, "API key has been revoked")

    # Daily usage reset
    today = _today_utc()
    if ak.usage_date != today:
        ak.usage_today = 0
        ak.usage_date  = today

    # Rate limit check (None = unlimited)
    limit = TIER_LIMITS.get(ak.tier)
    if limit is not None and ak.usage_today >= limit:
        log.warning(f"Rate limit hit: '{ak.name}' tier={ak.tier} usage={ak.usage_today}")
        raise HTTPException(
            429,
            f"Daily limit of {limit} requests reached for {ak.tier} tier. "
            "Resets at UTC midnight."
        )

    # Update counters
    ak.last_used    = datetime.now(timezone.utc).isoformat()
    ak.usage_today += 1
    ak.total_usage += 1
    _save_keys(_keys)

    return ak


# ── FastAPI dependency ─────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


async def require_key(
    request: Request,
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
) -> ApiKey:
    """
    Dependency injected into protected routes.
    Accepts key via:
      1. Authorization: Bearer btk_xxx   (preferred)
      2. ?api_key=btk_xxx                (fallback for browser testing)
    """
    raw = None
    if creds and creds.credentials:
        raw = creds.credentials
    else:
        raw = request.query_params.get("api_key")

    if not raw:
        raise HTTPException(
            401,
            detail="API key required. Pass it as: Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _validate_key(raw)


async def require_admin(key: Annotated[ApiKey, Depends(require_key)]) -> ApiKey:
    """Extra dependency for admin-only routes."""
    if key.tier not in ADMIN_TIERS:
        raise HTTPException(403, "Admin tier required for this endpoint")
    return key


async def require_upload_permission(key: Annotated[ApiKey, Depends(require_key)]) -> ApiKey:
    """Extra dependency for voice upload routes."""
    if key.tier not in UPLOAD_ALLOWED_TIERS:
        raise HTTPException(
            403,
            f"Voice upload requires pro or admin tier. Your tier: {key.tier}"
        )
    return key


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(text: str, voice: str, emotion: str, speed: float) -> str:
    raw = f"{text}|{voice}|{emotion}|{speed:.3f}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[bytes]:
    path = CACHE_DIR / f"{key}.wav"
    if path.exists():
        path.touch()
        log.info(f"Cache HIT {key[:12]}…")
        return path.read_bytes()
    return None


def _cache_put(key: str, wav_bytes: bytes) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.wav").write_bytes(wav_bytes)
    _evict_cache_if_needed()


def _evict_cache_if_needed() -> None:
    files = sorted(CACHE_DIR.glob("*.wav"), key=lambda f: f.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    limit = CACHE_MAX_MB * 1024 * 1024
    if total <= limit:
        return
    for f in files:
        sz = f.stat().st_size
        f.unlink()
        total -= sz
        log.info(f"Cache evicted: {f.name}")
        if total <= limit * 0.8:
            break


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    global _tone_converter
    log.info(f"Loading models on: {DEVICE}")

    if CHECKPOINTS_DIR.exists():
        _tone_converter = ToneColorConverter(
            f"{CHECKPOINTS_DIR}/config.json", device=DEVICE
        )
        _tone_converter.load_ckpt(f"{CHECKPOINTS_DIR}/checkpoint.pth")
        log.info("✓ ToneColorConverter loaded")
    else:
        log.warning(
            f"Checkpoints missing at '{CHECKPOINTS_DIR}'. Download with:\n"
            "  python -c \"import huggingface_hub; "
            "huggingface_hub.snapshot_download('myshell-ai/OpenVoiceV2', "
            "local_dir='checkpoints/converter')\""
        )

    langs_seen = set()
    for cfg in SPEAKER_CONFIG.values():
        lang = cfg["melo_language"]
        if lang not in langs_seen:
            _melo_models[lang] = MeloTTS(language=lang, device=DEVICE)
            langs_seen.add(lang)
            log.info(f"✓ MeloTTS [{lang}] loaded")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in SPEAKER_CONFIG.items():
        _try_load_embedding(name, cfg["reference_wav"])

    log.info(f"🚀 Server ready — voices: {list(_speaker_embeddings)}")


def _try_load_embedding(name: str, wav_path: Path) -> bool:
    if _tone_converter is None:
        return False
    if not wav_path.exists():
        log.warning(f"⚠️  No reference WAV for '{name}' at {wav_path}")
        return False
    try:
        log.info(f"Extracting embedding for {name}…")
        with tempfile.TemporaryDirectory() as tmp:
            se, _ = se_extractor.get_se(
                str(wav_path), _tone_converter, target_dir=tmp, vad=True
            )
        _speaker_embeddings[name] = se
        log.info(f"✓ Embedding ready for {name}")
        return True
    except Exception as e:
        log.error(f"Failed to extract embedding for {name}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="BenTech VoiceAI",
    description="OpenVoice V2 cloned-voice synthesis — authenticated",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text:    str
    voice:   str   = "Stephanie"
    emotion: str   = "Natural"
    speed:   float = 1.0
    pitch:   float = 1.0


class VoiceInfo(BaseModel):
    name:        str
    description: str
    ready:       bool


class CreateKeyRequest(BaseModel):
    name: str
    tier: str = "basic"   # admin | pro | basic


class KeyResponse(BaseModel):
    key:         str
    name:        str
    tier:        str
    created_at:  str
    usage_today: int
    total_usage: int
    revoked:     bool


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTES  (no auth required)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health check — public, no auth needed."""
    return {
        "status":  "ok",
        "service": "BenTech VoiceAI",
        "version": "1.2.0",
        "device":  DEVICE,
        "auth":    "required",
    }


@app.get("/voices", response_model=list[VoiceInfo])
def list_voices():
    """Public voice list — no auth so the frontend can show voices before login."""
    return [
        VoiceInfo(name=n, description=c["description"], ready=n in _speaker_embeddings)
        for n, c in SPEAKER_CONFIG.items()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED ROUTES  (API key required)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/synthesize")
async def synthesize(
    req: SynthesizeRequest,
    key: Annotated[ApiKey, Depends(require_key)],
):
    """Synthesize speech in a cloned voice. Requires valid API key."""
    text  = req.text.strip()
    voice = req.voice.strip()

    if not text:
        raise HTTPException(400, "text must not be empty")
    if len(text) > 2000:
        raise HTTPException(400, "text exceeds 2000 character limit")
    if voice not in SPEAKER_CONFIG:
        raise HTTPException(400, f"Unknown voice '{voice}'. Options: {list(SPEAKER_CONFIG)}")
    if voice not in _speaker_embeddings:
        raise HTTPException(503, f"Voice '{voice}' not ready. Upload a reference WAV.")
    if _tone_converter is None:
        raise HTTPException(503, "ToneColorConverter not loaded.")

    emotion = req.emotion
    for pfx in ["🎤 ", "🤗 ", "💼 ", "⚡ ", "🌿 ", "😢 "]:
        emotion = emotion.replace(pfx, "")
    emotion = emotion.strip()

    speed = EMOTION_SPEED.get(emotion, 0.88) * req.speed

    # Cache check
    ck     = _cache_key(text, voice, emotion, speed)
    cached = _cache_get(ck)
    if cached:
        return StreamingResponse(
            io.BytesIO(cached), media_type="audio/wav",
            headers={
                "X-Voice": voice, "X-Emotion": emotion, "X-Cache": "HIT",
                "X-Key-Tier": key.tier,
                "Content-Disposition": f'attachment; filename="bentech-{voice.lower()}.wav"',
            },
        )

    # Synthesis
    cfg     = SPEAKER_CONFIG[voice]
    melo    = _melo_models.get(cfg["melo_language"])
    if melo is None:
        raise HTTPException(500, "MeloTTS not loaded")

    speaker_ids = melo.hps.data.spk2id
    speaker = cfg["melo_speaker"]
    if speaker not in speaker_ids:
        speaker = next(iter(speaker_ids))

    t0 = time.time()
    log.info(f"[{key.name}|{key.tier}] Synthesizing {voice}/{emotion} speed={speed:.2f}")

    with tempfile.TemporaryDirectory() as tmp:
        base_wav  = os.path.join(tmp, "base.wav")
        clone_wav = os.path.join(tmp, "clone.wav")

        melo.tts_to_file(text, speaker_ids[speaker], base_wav, speed=speed)

        with tempfile.TemporaryDirectory() as se_tmp:
            se_src, _ = se_extractor.get_se(
                base_wav, _tone_converter, target_dir=se_tmp, vad=False
            )

        _tone_converter.convert(
            audio_src_path=base_wav,
            src_se=se_src,
            tgt_se=_speaker_embeddings[voice],
            output_path=clone_wav,
            message="@BenTechVoiceAI",
        )

        waveform, sr = torchaudio.load(clone_wav)
        buf = io.BytesIO()
        torchaudio.save(buf, waveform, sr, format="wav")
        buf.seek(0)
        wav_bytes = buf.read()

    elapsed = time.time() - t0
    log.info(f"✓ Done in {elapsed:.2f}s — {len(wav_bytes)//1024} KB")
    _cache_put(ck, wav_bytes)

    limit = TIER_LIMITS.get(key.tier)
    remaining = (limit - key.usage_today) if limit else None

    return StreamingResponse(
        io.BytesIO(wav_bytes), media_type="audio/wav",
        headers={
            "X-Voice": voice, "X-Emotion": emotion,
            "X-Latency": f"{elapsed:.2f}s", "X-Cache": "MISS",
            "X-Key-Tier": key.tier,
            "X-Requests-Remaining": str(remaining) if remaining is not None else "unlimited",
            "Content-Disposition": f'attachment; filename="bentech-{voice.lower()}.wav"',
        },
    )


@app.post("/clone-voice")
async def clone_voice(
    file: UploadFile = File(...),
    name: str        = Form(...),
    key: ApiKey      = Depends(require_upload_permission),
):
    """Upload a reference audio file to clone a new voice. Requires pro/admin tier."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(400, f"File type '{suffix}' not allowed.")

    data    = await file.read()
    size_kb = len(data) // 1024

    if len(data) < 8_000:
        raise HTTPException(400, "File too small — need at least 5–10 seconds of audio")
    if len(data) > 60_000_000:
        raise HTTPException(400, "File too large (max 60 MB)")

    name = name.strip()
    if not name or not name.replace(" ", "").isalnum():
        raise HTTPException(400, "name must be alphanumeric")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    dest = VOICES_DIR / f"{name.lower()}_reference.wav"
    dest.write_bytes(data)
    log.info(f"[{key.name}] Saved reference for '{name}' ({size_kb} KB)")

    is_new = name not in SPEAKER_CONFIG
    if is_new:
        SPEAKER_CONFIG[name] = {
            "reference_wav": dest,
            "melo_language": "EN",
            "melo_speaker":  "EN-Default",
            "description":   f"Custom voice: {name}",
        }

    ok = _try_load_embedding(name, dest)
    if not ok:
        raise HTTPException(500, f"Embedding extraction failed for '{name}'. Check audio quality.")

    return JSONResponse({
        "status":  "ready",
        "voice":   name,
        "is_new":  is_new,
        "size_kb": size_kb,
        "message": f"✓ {name}'s voice is live",
    })


@app.delete("/clone-voice/{name}")
def delete_voice(
    name: str,
    key: Annotated[ApiKey, Depends(require_admin)],
):
    """Delete a voice clone. Admin only."""
    if name in {"Stephanie", "Ben"}:
        raise HTTPException(403, f"Cannot delete protected voice '{name}'")
    if name not in SPEAKER_CONFIG:
        raise HTTPException(404, f"Voice '{name}' not found")

    ref = SPEAKER_CONFIG[name]["reference_wav"]
    if ref.exists():
        ref.unlink()
    _speaker_embeddings.pop(name, None)
    SPEAKER_CONFIG.pop(name, None)
    log.info(f"[{key.name}] Deleted voice: {name}")
    return {"status": "deleted", "voice": name}


# ── My key info ────────────────────────────────────────────────────────────────
@app.get("/me")
def me(key: Annotated[ApiKey, Depends(require_key)]):
    """Return info about the authenticated key — useful for the frontend."""
    limit = TIER_LIMITS.get(key.tier)
    return {
        "name":        key.name,
        "tier":        key.tier,
        "usage_today": key.usage_today,
        "limit_today": limit,
        "remaining":   (limit - key.usage_today) if limit else None,
        "total_usage": key.total_usage,
        "last_used":   key.last_used,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES  (admin tier only)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/keys", response_model=KeyResponse)
def create_key(
    req: CreateKeyRequest,
    key: Annotated[ApiKey, Depends(require_admin)],
):
    """Create a new API key. Admin only."""
    if req.tier not in TIER_LIMITS:
        raise HTTPException(400, f"Unknown tier '{req.tier}'. Options: {list(TIER_LIMITS)}")

    new_key = ApiKey(
        key        = "btk_" + secrets.token_hex(24),
        name       = req.name.strip(),
        tier       = req.tier,
        created_at = datetime.now(timezone.utc).isoformat(),
        last_used  = None,
    )
    _keys[new_key.key] = new_key
    _save_keys(_keys)
    log.info(f"[{key.name}] Created key '{new_key.name}' tier={new_key.tier}: {new_key.key[:16]}…")
    return KeyResponse(**new_key.model_dump())


@app.get("/admin/keys", response_model=list[KeyResponse])
def list_keys(key: Annotated[ApiKey, Depends(require_admin)]):
    """List all API keys. Admin only."""
    return [KeyResponse(**k.model_dump()) for k in _keys.values() if k.name != "env-admin"]


@app.delete("/admin/keys/{target_key}")
def revoke_key(
    target_key: str,
    key: Annotated[ApiKey, Depends(require_admin)],
):
    """Revoke an API key. Admin only."""
    ak = _keys.get(target_key)
    if not ak:
        raise HTTPException(404, "Key not found")
    if ak.key == key.key:
        raise HTTPException(400, "Cannot revoke your own key")
    ak.revoked = True
    _save_keys(_keys)
    log.info(f"[{key.name}] Revoked key '{ak.name}'")
    return {"status": "revoked", "name": ak.name}


@app.get("/admin/stats")
def admin_stats(key: Annotated[ApiKey, Depends(require_admin)]):
    """Usage stats overview. Admin only."""
    active  = [k for k in _keys.values() if not k.revoked and k.name != "env-admin"]
    today   = _today_utc()
    return {
        "total_keys":          len(active),
        "requests_today":      sum(k.usage_today for k in active if k.usage_date == today),
        "requests_total":      sum(k.total_usage for k in active),
        "voices_loaded":       list(_speaker_embeddings),
        "cache_entries":       len(list(CACHE_DIR.glob("*.wav"))) if CACHE_DIR.exists() else 0,
        "device":              DEVICE,
    }


@app.get("/cache/stats")
def cache_stats(key: Annotated[ApiKey, Depends(require_admin)]):
    if not CACHE_DIR.exists():
        return {"entries": 0, "size_mb": 0}
    files = list(CACHE_DIR.glob("*.wav"))
    total = sum(f.stat().st_size for f in files)
    return {"entries": len(files), "size_mb": round(total / 1024 / 1024, 2)}


@app.delete("/cache")
def clear_cache(key: Annotated[ApiKey, Depends(require_admin)]):
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    return {"status": "cleared"}


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def on_startup():
    global _keys
    _keys = _load_keys()
    log.info(f"Loaded {len(_keys)} API key(s)")
    load_models()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

