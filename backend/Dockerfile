# syntax=docker/dockerfile:1.7
# rtsp-detection backend — FastAPI + mediamtx control plane + YOLO 추론(별도 프로세스) + RTSP 캡처.
# 베이스: python:3.11-slim (Debian slim), Linux 전용.
# 의존성: pip + requirements.txt (torch+CUDA cu12 휠 포함 → 이미지 ~3GB, GPU/CPU 양용).

FROM python:3.11-slim

# -----------------------------------------------------------------------------
# 시스템 패키지
#   - libglib2.0-0 : opencv-python-headless 런타임 의존 (GLib)
#   - libgomp1     : torch / ultralytics 런타임 의존 (OpenMP)
#   - ffmpeg       : ffprobe 로 등록 시점 카메라 코덱 감지 (h264=passthrough/else=libx264 결정)
# -----------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# -----------------------------------------------------------------------------
# 의존성 설치 (레이어 캐시 극대화)
#   ultralytics 가 X11 의존 opencv-python 을 함께 끌어와 우리가 명시한
#   opencv-python-headless 의 cv2 를 덮어쓴다(headless 이미지에서 libxcb ImportError).
#   → opencv-python 제거 후 headless 강제 재설치로 정리(deepeye Dockerfile 패턴).
# -----------------------------------------------------------------------------
COPY requirements.txt ./
RUN pip install -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-deps --force-reinstall opencv-python-headless==4.13.0.92

# YOLO preset 가중치 캐시(/root/.cache/ultralytics — gpu compose 의 yolo_weights 볼륨 대상)
# 사전 생성(권한 회피). custom .pt 업로드는 제거됨(pickle RCE 차단) — /app/models 불필요.
RUN mkdir -p /root/.cache/ultralytics

# 애플리케이션 소스 + 마이그레이션 (alembic 이 스키마 정본 — RULES §9)
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY --chmod=0755 docker-entrypoint.sh /app/docker-entrypoint.sh
# Git 설정을 무시한 Windows checkout도 복구한다. CRLF/CR과 UTF-8 BOM을
# 이미지 안에서 제거하고, COPY --chmod로 호스트 파일 모드에 의존하지 않는다.
RUN python -c "from pathlib import Path; p=Path('/app/docker-entrypoint.sh'); data=p.read_bytes().removeprefix(b'\xef\xbb\xbf').replace(b'\r\n', b'\n').replace(b'\r', b'\n'); p.write_bytes(data)"

EXPOSE 8000

# 절대경로의 /bin/sh로 실행해 shebang·호스트 실행권한에 의존하지 않는다.
# 스크립트의 exec "$@"가 uvicorn을 PID 1로 교체한다.
ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
