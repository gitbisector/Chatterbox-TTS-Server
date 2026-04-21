import importlib.metadata
import os
from os import path
from typing import Annotated, List, Optional, Union
from urllib.parse import quote

import click
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

asr_model = ASRModelFactory.create_asr_model()
asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata.get("Home-page", projectMetadata.get("Homepage", ""))},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    beam_size: int = Query(
        default=5,
        ge=1,
        le=10,
        description="Beam size for faster_whisper. Lower = faster, higher = slightly better accuracy. 1 = greedy.",
    ),
):
    result = asr_model.transcribe(
        load_audio(audio_file.file, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "beam_size": beam_size,
        },
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post("/asr/batch", tags=["Endpoints"])
async def asr_batch(
    audio_files: List[UploadFile] = File(..., description="One or more audio files"),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
    task: str = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Optional[str] = Query(
        default=None,
        enum=LANGUAGE_CODES,
        description="Source language. Required for multilingual models; selection-scoring callers should always set it.",
    ),
    beam_size: int = Query(default=1, ge=1, le=10),
):
    """Transcribe N audio files in a single request.

    Optimized for TTS candidate selection: skips VAD, temperature fallback,
    segmentation, and word timestamps. Bypasses ``WhisperModel.transcribe``
    and calls ``ctranslate2.models.Whisper.generate`` directly with a batched
    feature tensor. Saves ~2× per-call Python/FastAPI overhead vs looping
    ``/asr`` N times; GPU compute remains linear in N (encoder is
    compute-bound per item at this model size).

    Response: JSON list of ``{"text": str, "language": str}``, in input order.
    """
    if CONFIG.ASR_ENGINE != "faster_whisper":
        return JSONResponse(
            status_code=501,
            content={"detail": "/asr/batch is only implemented for the faster_whisper engine"},
        )
    if not audio_files:
        return JSONResponse(status_code=400, content={"detail": "audio_files is empty"})

    # Lazy import to avoid pulling these at module load on other engines.
    from ctranslate2 import StorageView
    from faster_whisper.tokenizer import Tokenizer

    import time
    t0 = time.perf_counter()

    # Load + feature-extract each audio under a single lock. The feature
    # extractor is CPU-only and thread-safe, but we keep it under the lock
    # to preserve the single-queue semantics of the existing ASR endpoint
    # and avoid stepping on in-flight /asr work.
    with asr_model.model_lock:
        if asr_model.model is None:
            asr_model.load_model()
        wm = asr_model.model  # faster_whisper.WhisperModel
        mels = []
        max_frames = wm.feat_kwargs["chunk_length"] * wm.frames_per_second  # 30s → 3000 @ 16kHz/160-hop
        for uf in audio_files:
            audio = load_audio(uf.file, encode)
            feat = wm.feature_extractor(audio)  # [n_mels, T]
            if feat.shape[1] < max_frames:
                pad = ((0, 0), (0, max_frames - feat.shape[1]))
                feat = np.pad(feat, pad)
            else:
                feat = feat[:, :max_frames]
            mels.append(feat)
        batch = np.stack(mels).astype(np.float32)
        features_sv = StorageView.from_array(batch)

        tok = Tokenizer(
            wm.hf_tokenizer,
            wm.model.is_multilingual,
            task=task,
            language=language if wm.model.is_multilingual else "en",
        )
        prompt = list(tok.sot_sequence) + [tok.no_timestamps]
        prompts = [prompt] * len(audio_files)

        t_prep = time.perf_counter() - t0
        t1 = time.perf_counter()
        results = wm.model.generate(
            features_sv,
            prompts,
            beam_size=beam_size,
            max_length=wm.max_length,
            return_scores=False,
            suppress_blank=True,
        )
        t_gen = time.perf_counter() - t1

    texts = [tok.decode(r.sequences_ids[0]).strip() for r in results]
    t_total = time.perf_counter() - t0

    return JSONResponse(
        content={
            "results": [
                {"text": text, "language": language}
                for text in texts
            ],
            "timings_ms": {
                "prep": round(t_prep * 1000, 1),
                "generate": round(t_gen * 1000, 1),
                "total": round(t_total * 1000, 1),
            },
            "batch_size": len(audio_files),
        },
        headers={"Asr-Engine": CONFIG.ASR_ENGINE},
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
