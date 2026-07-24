"""Preset model class metadata must be available without weight downloads."""

import sys
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.inference import models_dir
from app import inference_api


def test_preset_classes_do_not_load_yolo_weights(monkeypatch) -> None:
    fake_ultralytics = ModuleType("ultralytics")

    class ForbiddenYOLO:
        def __init__(self, _path: str) -> None:
            raise AssertionError("class metadata must not load model weights")

    fake_ultralytics.YOLO = ForbiddenYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    app = FastAPI()
    app.include_router(inference_api.inference_router)
    client = TestClient(app)

    for model_name in models_dir.PRESET_MODELS:
        response = client.get(f"/api/inference/models/{model_name}/classes")
        assert response.status_code == 200
        classes = response.json()
        assert len(classes) == 80
        assert classes[0] == {"id": 0, "name": "person"}
        assert classes[-1] == {"id": 79, "name": "toothbrush"}
