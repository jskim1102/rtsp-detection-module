import { useEffect, useRef, useState } from "react";
import { resolveClassColor } from "../utils/colors";
import type { ModelSettings } from "./ModelSettingsModal";

/**
 * WHEP `<video>` 위에 절대 위치 `<canvas>` 로 bbox 오버레이.  ← 하이브리드 SEAM
 *
 * deepeye 원본은 같은 JPEG `<img>` 위에 그려 좌표 변환이 0 이었지만, 여기선 영상(WHEP)과
 * 추론 프레임(백엔드 별도 캡처)이 **서로 다른 한 장**이라 두 좌표계를 정합해야 한다:
 *
 * - canvas internal width/height = 부모 `<video>` 의 natural 크기(video.videoWidth/Height)
 *   — `loadedmetadata` · `resize` 이벤트로 갱신(WHEP 는 srcObject 라 `load` 가 없음).
 * - detections.xyxy 는 추론 캡처 프레임(frameW×frameH) 좌표계 → `videoNatural / detFrame`
 *   스케일로 그린다. **두 해상도가 같으면 sx=sy=1 (identity), 다르면 자동 보정.**
 *   (frameW/H 미상=0 이면 identity 가정.)
 * - canvas 는 video 와 동일 박스에 절대배치(inset:0, 100%)하되 `object-fit:contain` 으로
 *   video 와 똑같이 레터/필러박싱 → 비 16:9 카메라(예 4:3)에서도 화면상 정렬.
 * - settings (per-model) 가 주어지면 클래스 필터링 + conf threshold + 색상 override 적용.
 *
 * 새 detection 이 도착할 때 최신 박스를 그대로 그리고 다음 결과까지 유지한다.
 * 빈 detection 이 오면 canvas 를 즉시 비워 사라진 객체를 유령 박스로 남기지 않는다.
 */

export interface Detection {
  class_id: number;
  name: string;
  conf: number;
  xyxy: [number, number, number, number];
  model: string;
}

interface Props {
  // 부모 WhepPlayer 의 <video> ref — 이 위에 절대배치 canvas 를 겹친다.
  videoRef: React.RefObject<HTMLVideoElement | null>;
  detections: Detection[];
  // WS hook/CameraGrid 계약 호환용. latest-only 렌더링은 시간축 보정에 사용하지 않는다.
  captureTs?: number | null;
  // YOLO 가 본 추론 캡처 프레임 치수 (WS frame:{w,h}). 0 이면 스케일 = identity.
  frameW: number;
  frameH: number;
  // 모델별 설정 — class filter (enabledClasses) + conf threshold + color override.
  // undefined 또는 해당 모델 키 없음 → 모든 클래스 표시 + 기본 팔레트 색.
  settings?: Record<string, ModelSettings>;
}

interface ScaleVars {
  scale: number;
  lineWidth: number;
  fontPx: number;
}

// settings 기반 필터 — class filter(enabledClasses) + conf threshold. 원본과 동일 로직.
function filterVisible(
  dets: Detection[],
  settings?: Record<string, ModelSettings>,
): Detection[] {
  return dets.filter((det) => {
    const ms = settings?.[det.model];
    // class filter (enabledClasses)
    const en = ms?.classes;
    if (en !== undefined && en !== null && !en.includes(det.class_id)) return false;
    // conf threshold — UI 에서 override 한 값이 있으면 그 미만은 숨김
    if (ms?.conf !== undefined && det.conf < ms.conf) return false;
    return true;
  });
}

// 한 박스 그리기 — 원본 per-det 드로잉을 그대로 추출(재사용). box 는 추론 프레임 좌표계라
// 여기서 sx/sy 를 곱해 video natural 좌표로 변환한다(원본과 동일 SEAM 적용).
function drawDet(
  ctx: CanvasRenderingContext2D,
  box: readonly number[],
  det: Detection,
  sx: number,
  sy: number,
  sv: ScaleVars,
  showModelPrefix: boolean,
  settings?: Record<string, ModelSettings>,
) {
  // SEAM 스케일 적용 — 추론 프레임 좌표 → video natural 좌표
  const x1 = box[0] * sx;
  const y1 = box[1] * sy;
  const x2 = box[2] * sx;
  const y2 = box[3] * sy;
  // 색상 — per-model override 가 있으면 그것, 없으면 기본 팔레트
  const colorOverride = settings?.[det.model]?.colors;
  const color = resolveClassColor(det.class_id, colorOverride);

  // 사각형
  ctx.lineWidth = sv.lineWidth;
  ctx.strokeStyle = color;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // 라벨 텍스트
  const short =
    showModelPrefix && det.model
      ? det.model.endsWith(".pt")
        ? det.model.slice(0, -3)
        : det.model
      : "";
  const label = short
    ? `[${short}] ${det.name} ${det.conf.toFixed(2)}`
    : `${det.name} ${det.conf.toFixed(2)}`;

  const padX = 4 * sv.scale;
  const padY = 2 * sv.scale;
  const textW = ctx.measureText(label).width;
  const labelH = sv.fontPx + padY * 2;
  const labelY = Math.max(0, y1 - labelH);

  // 라벨 배경
  ctx.fillStyle = color;
  ctx.fillRect(x1, labelY, textW + padX * 2, labelH);

  // 라벨 글자 (흰색)
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x1 + padX, labelY + padY);
}

function BboxOverlay({ videoRef, detections, frameW, frameH, settings }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  // video natural 크기 — loadedmetadata + resize 시 갱신. (원본 그대로 유지 — size state 설정)
  // WHEP 는 srcObject(MediaStream) 라 <img> 의 onload 가 없다. 트랙 해상도 확정/변경은
  // video 의 'resize'(intrinsic 크기 변동) + 'loadedmetadata' 로 통지된다.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => {
      if (video.videoWidth && video.videoHeight) {
        setSize((prev) => {
          if (prev?.w === video.videoWidth && prev?.h === video.videoHeight) return prev;
          return { w: video.videoWidth, h: video.videoHeight };
        });
      }
    };
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    update(); // 이미 메타데이터 로드됐을 수 있음
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
    };
  }, [videoRef]);

  // 최신 detection 을 그대로 한 번 그린다. 다음 결과가 올 때까지 canvas 픽셀을 유지한다.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (detections.length === 0) return;

    // SEAM 좌표 스케일 — xyxy(추론 프레임 좌표계) → canvas internal(video natural).
    // 두 해상도 같으면 identity. (frameW/H 미상=0 이면 identity.)
    const sx = frameW > 0 ? size.w / frameW : 1;
    const sy = frameH > 0 ? size.h / frameH : 1;
    const visible = filterVisible(detections, settings);

    // 원본 좌표계 기준 적정 사이즈 (해상도 클수록 선 두께/폰트 비율 보정)
    const scale = Math.max(1, Math.min(size.w, size.h) / 600);
    const lineWidth = Math.max(1.5, 2 * scale);
    const fontPx = Math.max(11, Math.round(13 * scale));
    ctx.font = `${fontPx}px sans-serif`;
    ctx.textBaseline = "top";
    const sv: ScaleVars = { scale, lineWidth, fontPx };

    const distinctModels = new Set(visible.map((d) => d.model).filter(Boolean));
    const showModelPrefix = distinctModels.size >= 2;
    for (const det of visible) {
      drawDet(ctx, det.xyxy, det, sx, sy, sv, showModelPrefix, settings);
    }
  }, [detections, frameH, frameW, settings, size]);

  // video 메타데이터가 아직이면(size 미상) 그릴 대상 없음 — 오버레이 생략.
  if (!size) return null;

  return (
    <canvas
      ref={canvasRef}
      width={size.w}
      height={size.h}
      style={canvasStyle}
    />
  );
}

const canvasStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "contain",
  pointerEvents: "none",
};

export default BboxOverlay;
