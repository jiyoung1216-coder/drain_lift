#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------
# 0) 공통 유틸리티 함수
# ------------------------------------------------
def masks_to_polys(result, W, H, hole_cls=0, sd_cls=1):
    """ultralytics result -> (sd_polys, hole_polys) in pixel coords"""
    sd_polys, hole_polys = [], []

    if result.masks is None or result.masks.data is None:
        # seg가 없으면 bbox 기반으로 약식으로 만든다
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls  = result.boxes.cls.cpu().numpy().astype(int)
            for box, c in zip(xyxy, cls):
                x1, y1, x2, y2 = box
                poly = [(int(x1), int(y1)), (int(x2), int(y1)),
                        (int(x2), int(y2)), (int(x1), int(y2))]
                if c == sd_cls:
                    sd_polys.append(poly)
                elif c == hole_cls:
                    hole_polys.append(poly)
        return sd_polys, hole_polys

    boxes = result.boxes
    classes = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []

    # r.masks.xyn -> list of (N,2) in 0~1
    for i, poly_norm in enumerate(result.masks.xyn):
        if i >= len(classes):
            continue
        c = classes[i]
        poly_norm = np.asarray(poly_norm, dtype=np.float32)
        poly_pix = np.stack([poly_norm[:, 0] * W, poly_norm[:, 1] * H], axis=1)
        poly_pix = poly_pix.astype(int).tolist()
        if c == sd_cls:
            sd_polys.append(poly_pix)
        elif c == hole_cls:
            hole_polys.append(poly_pix)

    return sd_polys, hole_polys


def bbox_of_polygon(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    return x0, y0, x1 - x0, y1 - y0


def centroid_mean(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (int(np.mean(xs)), int(np.mean(ys)))


def draw_centerline(img, c, dir_vec, color=(255, 255, 0), thickness=2, length=2000):
    p0 = (int(c[0] - dir_vec[0] * length), int(c[1] - dir_vec[1] * length))
    p1 = (int(c[0] + dir_vec[0] * length), int(c[1] + dir_vec[1] * length))
    cv2.line(img, p0, p1, color, thickness)


def select_best_sd(sd_polys, hole_polys):
    """홀들에 가장 가까운 SD 하나 선택"""
    if not sd_polys:
        return None
    if not hole_polys:
        return sd_polys[0]

    centers = [centroid_mean(h) for h in hole_polys]
    mu = np.mean(np.array(centers), axis=0)

    best, best_d = None, 1e18
    for poly in sd_polys:
        c = centroid_mean(poly)
        d = np.hypot(c[0] - mu[0], c[1] - mu[1])
        if d < best_d:
            best, best_d = poly, d
            
    return best


# ------------------------------------------------
# 1) SD 중심 찾기 (빗물받이 정가운데 좌표 추출)
# ------------------------------------------------
def find_sd_center(sd_poly, vis):
    """빗물받이 폴리곤의 BBOX 중심을 구하고 시각화합니다."""
    
    # 1. BBOX 계산
    x0, y0, bw, bh = bbox_of_polygon(sd_poly)

    # 2. 중심 (BBOX 중심) 계산
    c = (x0 + bw // 2, y0 + bh // 2)

    # 3. 시각화 (중심점, BBOX, 중심선)
    
    # BBOX (마젠타색)
    cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (255, 0, 255), 1)

    # 중심점 (파란색)
    cv2.circle(vis, c, 6, (255, 0, 0), -1)

    # 중심선 (하늘색 - 긴 변에 평행하게)
    if bw > bh:
        # 가로가 더 길면 세로(y)축 중심선 (수직선)
        dir_vec = np.array([0.0, 1.0]) 
    else:
        # 세로가 더 길면 가로(x)축 중심선 (수평선)
        dir_vec = np.array([1.0, 0.0])
        
    draw_centerline(vis, c, dir_vec, (0, 255, 255), 2) # 하늘색 (Cyan)

    # 최종 Target 표시 (녹색)
    cv2.circle(vis, c, 10, (0, 255, 0), -1)
    cv2.putText(vis, "FINAL TARGET", (c[0] + 15, c[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return c, (x0, y0, bw, bh)


# ------------------------------------------------
# 2) 메인: YOLO → 중심점 추출
# ------------------------------------------------
def main():
    
    # --------------------------------------------------
    # 🎯 사용자 설정 필요: 이미지 및 모델 경로
    # --------------------------------------------------
    
    # 1. YOLO 모델 경로: 실제 best.pt 파일의 경로로 설정하세요.
    cand_model_paths = [
        "/home/sj-desktop/ros2_ws/src/grating_detector_py/resource/weights/best.pt",
        "/home/sj-desktop/ros2_ws/install/grating_detector_py/share/grating_detector_py/resource/weights/best.pt",
    ]
    model_path = None
    for p in cand_model_paths:
        if Path(p).exists():
            model_path = p
            break
            
    if model_path is None:
        print("ERROR: YOLO 모델(best.pt) 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
        return 
    
    # 2. 입력 이미지 경로: 실제 빗물받이 이미지 파일의 경로로 설정하세요.
    # 이전 오류의 원인이었으므로, 정확한 경로를 명시해야 합니다.
    img_path = "/home/sj-desktop/test.jpg" # 실제 파일 경로를 이곳에 입력
    out_dir = Path("/home/sj-desktop/debug")
    out_path = out_dir / "grating_debug_center_target.png"
    
    # --------------------------------------------------


    # 3) 이미지 읽기
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"ERROR: 이미지를 로드할 수 없습니다. 경로를 확인해주세요: {img_path}")
        return
        
    H, W = img.shape[:2]
    vis = img.copy() # 시각화용 이미지 복사
    
    
    # 4) YOLO 추론
    try:
        model = YOLO(model_path)
        r = model.predict(source=img, verbose=False)[0]
    except Exception as e:
        print(f"ERROR: YOLO 모델 로드/추론 오류: {e}")
        return
        
    # 5) YOLO → sd/hole 폴리곤 분리
    sd_polys, hole_polys = masks_to_polys(r, W, H, hole_cls=0, sd_cls=1)


    # 6) SD 하나 고르기 (가장 적절한 빗물받이 선택)
    sd_poly = select_best_sd(sd_polys, hole_polys)
    
    if sd_poly is None: 
        print("결과: SD가 하나도 안 나왔거나 선택할 SD가 없습니다.")
        return
    
    # SD 테두리 표시 (녹색)
    cv2.polylines(vis, [np.array(sd_poly, np.int32)], True, (0, 255, 0), 2)


    # 7) SD 중심 타깃 선택
    center_pt, bbox = find_sd_center(
        sd_poly, 
        vis
    )

    if center_pt is None:
        print("결과: SD 중심점을 찾지 못했습니다.")
    else:
        # 정규화 값도 같이 출력
        u_c, v_c = center_pt[0] / W, center_pt[1] / H
        print(f"✅ FINAL TARGET Center (px): {center_pt}, norm=({u_c:.4f}, {v_c:.4f})")
    
    
    # 8) 시각화 결과 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"[saved] {out_path}")
    
    # Note: Removed the display_image function as it was a dummy/placeholder.


if __name__ == "__main__":
    main()