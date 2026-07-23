"""추론 제어 + 모델 가중치 관리 라우터 (/api/inference/*).

deepeye-lite ipcam.py 의 inference_router 차용. main.py 에서 include_router(inference_router).
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.inference import models_dir
from app.streaming.manager import manager as stream_manager

logger = logging.getLogger("rtsp-streaming.inference")


# ─── 추론 제어 (모델 토글, ON/OFF, conf threshold) ───


class InferenceConfig(BaseModel):
    enabled: bool
    model: str
    conf_threshold: float
    device: str
    gpu_util_target: float
    gpu_util_duty: float


class InferenceConfigUpdate(BaseModel):
    enabled: bool | None = None
    model: str | None = None
    conf_threshold: float | None = None
    gpu_util_target: float | None = None


inference_router = APIRouter(prefix="/api/inference", tags=["inference"])


@inference_router.get("/config", response_model=InferenceConfig)
def get_inference_config() -> dict:
    """현재 추론 워커 상태."""
    return stream_manager.get_inference_config()


@inference_router.put("/config", response_model=InferenceConfig)
def update_inference_config(body: InferenceConfigUpdate) -> dict:
    """추론 ON/OFF · 모델 · conf threshold 변경. 부분 업데이트 지원."""
    if body.enabled is not None:
        stream_manager.set_inference_enabled(body.enabled)
    if body.model is not None:
        stream_manager.set_inference_model(body.model)
    if body.conf_threshold is not None:
        stream_manager.set_inference_conf_threshold(body.conf_threshold)
    if body.gpu_util_target is not None:
        stream_manager.set_gpu_util_target(body.gpu_util_target)
    return stream_manager.get_inference_config()


# ─── 모델 목록 + 클래스 메타 (preset-only — custom 업로드 제거, pickle RCE 차단) ───


class ModelInfo(BaseModel):
    name: str
    type: str  # 항상 "preset"
    size_mb: float | None = None


@inference_router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[dict]:
    """preset(YOLO26 5종) 목록. custom 업로드는 제거됨."""
    return models_dir.list_all_models()


_classes_cache: dict[str, dict[int, str]] = {}


@inference_router.get("/models/{name}/classes")
def get_model_classes(name: str) -> list[dict]:
    """주어진 preset 모델의 클래스 ID→이름 목록.

    preset allowlist 통과 시에만 lazy load → `model.names` → 캐시. 임의 `.pt` 로드 금지
    (pickle RCE 차단). preset 가중치는 immutable 이라 캐시 stale 문제가 없다.
    """
    if not models_dir.is_preset(name):
        raise HTTPException(status_code=404, detail="알 수 없는 모델 (preset 만 가능)")

    if name not in _classes_cache:
        try:
            from ultralytics import YOLO
            path = models_dir.resolve_model_path(name)
            m = YOLO(path)
            names = dict(m.names) if getattr(m, "names", None) else {}
            _classes_cache[name] = {int(k): str(v) for k, v in names.items()}
        except Exception as e:
            logger.exception("모델 클래스 조회 실패: %s", name)
            raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}")

    names = _classes_cache[name]
    return [{"id": k, "name": v} for k, v in sorted(names.items())]
