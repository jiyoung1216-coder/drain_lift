#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# 메시지 인터페이스 (GratingHoles)
from grating_interfaces.msg import GratingHoles

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import torch  
import time  # 성능 측정용

class GratingDetectorNode(Node):
    def __init__(self):
        super().__init__('grating_detector')

        # ============= GPU 설정 =============
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"🔧 사용 디바이스: {self.device}")
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.get_logger().info(f"📊 GPU: {gpu_name}")
            self.get_logger().info(f"💾 GPU 메모리: {gpu_mem:.2f} GB")
        else:
            self.get_logger().warn("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        # =========================================

        # ----------------- 1) 모델 경로 -----------------
        cand = [
            "/home/sj-desktop/ros2_ws/install/grating_detector_py/share/grating_detector_py/resource/weights/best.pt (1)",
            "/home/sj-desktop/ros2_ws/src/grating_detector_py/resource/weights/best.pt (1)",
        ]
        model_path = None
        for p in cand:
            if Path(p).exists():
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError("best.pt 를 못 찾음")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)  # 모델을 GPU로 이동!
        self.get_logger().info(f"[grating_detector] loading model from: {model_path}")

        # ----------------- 2) 카메라 (해상도 축소) -----------------
        cam_id = self.declare_parameter('camera_id', 0).get_parameter_value().integer_value
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        
        # ============= 카메라 해상도 설정 =============
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # =================================================
        
        if not self.cap.isOpened():
            self.get_logger().error("camera open fail")
        else:
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(f"camera {cam_id} opened ({actual_w}x{actual_h})")

        # ----------------- 3) 퍼블리셔 -----------------
        self.pub = self.create_publisher(GratingHoles, '/grating/holes', 10)

        # ----------------- 4) 타이머 -----------------
        self.timer = self.create_timer(0.5, self.timer_cb)  # 2Hz

        # ============= 시각화 및 성능 모니터링 =============
        self.show = True
        self.display_scale = 0.7  # 표시 크기 비율 (70%)
        self.frame_count = 0
        self.inference_times = []
        # ============================================

    # ==================== 유틸리티 함수들 ====================

    def _bbox_of_polygon(self, pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        return x0, y0, x1 - x0, y1 - y0

    def _centroid_mean(self, pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _draw_centerline(self, img, c, dir_vec, color=(0, 255, 255), thickness=2, length=2000):
        p0 = (int(c[0] - dir_vec[0] * length), int(c[1] - dir_vec[1] * length))
        p1 = (int(c[0] + dir_vec[0] * length), int(c[1] + dir_vec[1] * length))
        cv2.line(img, p0, p1, color, thickness)

    def yolo_result_to_polys(self, r, W, H):
        """
        YOLO seg 결과 -> (storm_drain polys, hole polys)
        class 1 : storm_drain, class 0 : hole
        """
        sd_polys, hole_polys = [], []
        if r.masks is None or r.boxes is None:
            return sd_polys, hole_polys

        boxes = r.boxes
        # CPU로 이동 및 NumPy 변환
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        
        for i, poly in enumerate(r.masks.xyn):
            cls_id = classes[i]
            # 정규화 좌표 -> 픽셀
            poly = np.asarray(poly, dtype=np.float32)
            poly_px = []
            for (u, v) in poly:
                x = int(u * W)
                y = int(v * H)
                poly_px.append((x, y))

            if cls_id == 1:
                sd_polys.append(poly_px)
            elif cls_id == 0:
                hole_polys.append(poly_px)

        return sd_polys, hole_polys

    # ==================== 핵심 로직: 중앙 타깃 추출 (Target 1 대신 사용) ====================
    def _find_center_target(self, sd_poly, vis):
        """빗물받이 폴리곤의 BBOX 중심을 구하고 시각화합니다."""
        
        H, W = vis.shape[:2]
        # 1. BBOX 계산
        x0, y0, bw, bh = self._bbox_of_polygon(sd_poly)

        # 2. 중심 (BBOX 중심) 계산
        c = (x0 + bw // 2, y0 + bh // 2)
        
        # 3. 시각화
        
        # BBOX (마젠타색)
        cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (255, 0, 255), 1)

        # 중심점 (파란색)
        cv2.circle(vis, c, 6, (255, 0, 0), -1)

        # 중심선 (하늘색 - 긴 변에 평행하게)
        if bw > bh:
            dir_vec = np.array([0.0, 1.0]) # 수직선
        else:
            dir_vec = np.array([1.0, 0.0]) # 수평선
            
        self._draw_centerline(vis, c, dir_vec, (0, 255, 255), 2) # 하늘색 (Cyan)

        # 최종 Target 표시 (녹색)
        cv2.circle(vis, c, 10, (0, 255, 0), -1)
        cv2.putText(vis, "CENTER TARGET", (c[0] + 15, c[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 정규화 좌표 반환
        u_c = c[0] / W
        v_c = c[1] / H
        return u_c, v_c


    # ==================== 타이머 콜백 (메인 루프) ====================
    def timer_cb(self):
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("camera frame read fail")
            return

        H, W = frame.shape[:2]
        
        # ============= GPU에서 추론 실행 =============
        start_time = time.time()
        # predict 함수의 device 인수를 사용하여 GPU 메모리 오류 경고 방지 및 디바이스 지정
        r = self.model.predict(source=frame, device=self.device, verbose=False)[0]
        inference_time = time.time() - start_time
        
        self.inference_times.append(inference_time)
        self.frame_count += 1
        # ===========================================

        # 기본 캔버스
        vis = frame.copy()

        # YOLO → 폴리곤
        sd_polys, hole_polys = self.yolo_result_to_polys(r, W, H)

        # ROS 2 메시지 초기화
        msg = GratingHoles()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = "logitech_cam"

        if not sd_polys:
            # 그레이팅 못찾으면 fallback (기존 로직 유지)
            msg.num_holes = 2
            msg.holes_uv = [0.45, 0.3, 0.45, 0.7] # 더미 좌표
            self.pub.publish(msg)
            self.get_logger().info("publish(no_sd): [0.45,0.3,0.45,0.7]")
        else:
            # 빗물받이 선택 (가장 적절한 SD)
            sd_poly = self._select_best_sd(sd_polys, hole_polys)
            if sd_poly is None:
                 msg.num_holes = 2
                 msg.holes_uv = [0.45, 0.3, 0.45, 0.7] # 폴백
            else:
                # 빗물받이 테두리 그리기
                cv2.polylines(vis, [np.array(sd_poly, np.int32)], True, (0,255,0), 2)
                
                # ⭐️⭐️⭐️ 정중앙 좌표 추출 ⭐️⭐️⭐️
                u_c, v_c = self._find_center_target(sd_poly, vis)

                # 토픽 발행: 정중앙 좌표를 Target 1, Target 2 위치에 반복해서 발행하여 메시지 타입 유지
                msg.num_holes = 2
                msg.holes_uv = [float(u_c), float(v_c), float(u_c), float(v_c)]
            
            self.pub.publish(msg)
            
            # ============= 성능 정보 로깅 =============
            avg_time = np.mean(self.inference_times[-10:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            if self.frame_count % 10 == 0:
                log_msg = f"✅ TARGET: [{u_c:.4f},{v_c:.4f}] | FPS: {fps:.1f} | 추론시간: {inference_time*1000:.1f}ms"
                if self.device == 'cuda':
                    gpu_mem = torch.cuda.memory_allocated() / 1024**2
                    log_msg += f" | GPU메모리: {gpu_mem:.0f}MB"
                self.get_logger().info(log_msg)
            # ========================================

            
        # ============= 실시간 표시 =============
        if self.show:
            fps = 1.0 / inference_time if inference_time > 0 else 0
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, f"Device: {self.device.upper()}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if self.device == 'cuda':
                gpu_mem = torch.cuda.memory_allocated() / 1024**2
                cv2.putText(vis, f"GPU: {gpu_mem:.0f}MB", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            display_h = int(H * self.display_scale)
            display_w = int(W * self.display_scale)
            vis_resized = cv2.resize(vis, (display_w, display_h))
            
            cv2.imshow("grating_debug", vis_resized)
            if cv2.waitKey(1) & 0xFF == 27:
                self.show = False
                cv2.destroyAllWindows()
        # ====================================================

    def _select_best_sd(self, sd_polys, hole_polys):
        """홀들에 가장 가까운 SD 하나 선택 (유틸 함수를 클래스 메서드로 변환)"""
        if not sd_polys:
            return None
        if not hole_polys:
            return sd_polys[0]

        centers = [self._centroid_mean(h) for h in hole_polys]
        mu = np.mean(np.array(centers), axis=0)

        best, best_d = None, 1e18
        for poly in sd_polys:
            c = self._centroid_mean(poly)
            d = np.hypot(c[0] - mu[0], c[1] - mu[1])
            if d < best_d:
                best, best_d = poly, d
                
        return best


    def __del__(self):
        """소멸자: GPU 메모리 정리 및 카메라 해제"""
        if hasattr(self, 'device') and self.device == 'cuda':
            torch.cuda.empty_cache()
            self.get_logger().info("✅ GPU 메모리 정리 완료")
        if hasattr(self, 'cap') and self.cap.isOpened():
             self.cap.release()
             self.get_logger().info("📷 카메라 해제 완료")


def main(args=None):
    rclpy.init(args=args)
    node = GratingDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Detector stopped manually.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()