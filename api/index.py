"""FastAPI server: English text -> IPA (dựa trên CMU Pronouncing Dictionary)."""
import os
import re
import secrets
import threading
from functools import lru_cache
from typing import Optional

import cmudict
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MAX_LEN = 5000

app = FastAPI(title="English → IPA API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def restore_original_path(request, call_next):
    """Trên Vercel, rewrite có thể làm function nhận path /api/index thay vì path gốc.
    vercel.json gửi path gốc qua query `__path`; ở đây khôi phục lại để FastAPI route đúng.
    Vẫn chạy bình thường khi không có `__path` (local, Docker)."""
    path = request.scope["path"]
    forwarded = request.query_params.get("__path")
    if forwarded is not None:
        request.scope["path"] = "/" + forwarded.lstrip("/")
    elif path == "/api/index" or path.startswith("/api/index/"):
        request.scope["path"] = path[len("/api/index"):] or "/"
    return await call_next(request)


security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """Basic auth; user/password lấy từ ENV BASIC_AUTH_USER / BASIC_AUTH_PASSWORD."""
    user = os.getenv("BASIC_AUTH_USER")
    password = os.getenv("BASIC_AUTH_PASSWORD")
    if not user or not password:  # thiếu cấu hình -> từ chối, không mở public
        raise HTTPException(status_code=500, detail="Chưa cấu hình BASIC_AUTH_USER / BASIC_AUTH_PASSWORD")
    ok_user = secrets.compare_digest(credentials.username.encode(), user.encode())
    ok_pass = secrets.compare_digest(credentials.password.encode(), password.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=401,
            detail="Sai tên đăng nhập hoặc mật khẩu",
            headers={"WWW-Authenticate": "Basic"},
        )


# Load 1 lần khi cold start
CMU = cmudict.dict()

ARPA_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ",
    "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
    "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

TOKEN_RE = re.compile(r"([A-Za-z]+(?:'[A-Za-z]+)*)|(\s+)|(.)", re.DOTALL)
PHONE_RE = re.compile(r"([A-Z]+)([012])?")


def arpa_to_ipa(phones: list[str], stress: bool = True) -> str:
    out = []
    for p in phones:
        base, s = PHONE_RE.fullmatch(p).groups()
        stressed = s in ("1", "2")
        if base == "AH":
            ipa = "ʌ" if stressed else "ə"
        elif base == "ER":
            ipa = "ɝ" if stressed else "ɚ"
        else:
            ipa = ARPA_TO_IPA[base]
        if stress and s == "1":
            ipa = "ˈ" + ipa
        elif stress and s == "2":
            ipa = "ˌ" + ipa
        out.append(ipa)
    return "".join(out)


def word_to_ipa(word: str, stress: bool = True) -> Optional[str]:
    prons = CMU.get(word.lower())
    return arpa_to_ipa(prons[0], stress) if prons else None


class WordIPA(BaseModel):
    word: str
    ipa: Optional[str]
    found: bool


class IPAResponse(BaseModel):
    text: str
    ipa: str
    words: list[WordIPA]
    not_found: list[str]


class IPARequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LEN)
    stress: bool = True


def convert(text: str, stress: bool) -> IPAResponse:
    pieces, words, missing = [], [], []
    for m in TOKEN_RE.finditer(text):
        word, space, other = m.groups()
        if word:
            ipa = word_to_ipa(word, stress)
            words.append(WordIPA(word=word, ipa=ipa, found=ipa is not None))
            if ipa is None:
                missing.append(word)
            pieces.append(ipa or word)
        else:
            pieces.append(space or other)
    return IPAResponse(text=text, ipa="".join(pieces), words=words, not_found=missing)


@app.get("/")
def root():
    return {
        "service": "English → IPA",
        "usage": {
            "POST /espeak": {"text": "hello world", "lang": "en-us (tuỳ chọn)", "stress": True},
            "GET /ipa?text=hello world": "tra từ điển CMU",
            "auth": "HTTP Basic",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ipa", response_model=IPAResponse, dependencies=[Depends(require_auth)])
def ipa_get(text: str = Query(..., min_length=1, max_length=MAX_LEN), stress: bool = True):
    return convert(text, stress)


@app.post("/ipa", response_model=IPAResponse, dependencies=[Depends(require_auth)])
def ipa_post(body: IPARequest):
    return convert(body.text, body.stress)


# ---------------------------------------------------------------------------
# eSpeak NG
#  - Vercel : dùng thư viện đóng gói sẵn trong wheel `espeakng-loader`
#  - Docker : ESPEAK_USE_SYSTEM=1 -> dùng espeak-ng cài bằng apt
# ---------------------------------------------------------------------------
_espeak_lock = threading.Lock()  # libespeak-ng không thread-safe
_espeak_ready = False


def _init_espeak():
    global _espeak_ready
    if _espeak_ready:
        return
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if os.getenv("ESPEAK_USE_SYSTEM") != "1":
        import espeakng_loader

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.data_path = espeakng_loader.get_data_path()
    _espeak_ready = True


@lru_cache(maxsize=16)
def _backend(lang: str, stress: bool):
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(lang, preserve_punctuation=True, with_stress=stress)


class EspeakResponse(BaseModel):
    text: str
    lang: str
    ipa: str


class EspeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LEN)
    lang: str = "en-us"
    stress: bool = True


@app.post("/espeak", response_model=EspeakResponse, dependencies=[Depends(require_auth)])
def espeak(body: EspeakRequest):
    with _espeak_lock:
        try:
            _init_espeak()
            backend = _backend(body.lang, body.stress)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=f"Ngôn ngữ không hỗ trợ: {body.lang} ({e})")
        ipa = backend.phonemize([body.text], strip=True)[0]
    return EspeakResponse(text=body.text, lang=body.lang, ipa=ipa.strip())
