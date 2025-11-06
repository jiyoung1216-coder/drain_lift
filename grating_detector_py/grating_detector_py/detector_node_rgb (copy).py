#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from grating_interfaces.msg import GratingHoles

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class GratingDetectorNode(Node):
    def __init__(self):
        super().__init__('grating_detector')

        # ----------------- 1) 모델 경로 -----------------
        cand = [
            "/home/sj-desktop/ros2_ws/install/grating_detector_py/share/grating_detector_py/resource/weights/best.pt",
            "/home/sj-desktop/ros2_ws/src/grating_detector_py/resource/weights/best.pt",
        ]
        model_path = None
        for p in cand:
            if Path(p).exists():
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError("best.pt 를 못 찾음")
        self.model = YOLO(model_path)
        self.get_logger().info(f"[grating_detector] loading model from: {model_path}")

        # ----------------- 2) 카메라 -----------------
        cam_id = self.declare_parameter('camera_id', 0).get_parameter_value().integer_value
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("camera open fail")
        else:
            self.get_logger().info(f"camera {cam_id} opened")

        # ----------------- 3) 퍼블리셔 -----------------
        self.pub = self.create_publisher(GratingHoles, '/grating/holes', 10)

        # ----------------- 4) 타이머 -----------------
        self.timer = self.create_timer(0.5, self.timer_cb)  # 2Hz

        # 시각화 on/off
        self.show = True

    # ==================== 유틸들 ====================

    def _bbox_of_polygon(self, pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        return x0, y0, x1 - x0, y1 - y0

    def _centroid_mean(self, pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _contour_area(self, pts):
        if len(pts) < 3:
            return 0.0
        cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        return float(cv2.contourArea(cnt))

    def _draw_centerline(self, img, c, dir_vec, color=(255, 255, 0), thickness=2, length=2000):
        p0 = (int(c[0] - dir_vec[0] * length), int(c[1] - dir_vec[1] * length))
        p1 = (int(c[0] + dir_vec[0] * length), int(c[1] + dir_vec[1] * length))
        cv2.line(img, p0, p1, color, thickness)

    def yolo_result_to_polys(self, r, W, H):
        """
        YOLO seg 결과 -> (storm_drain polys, hole polys)
        class 1 : storm_drain
        class 0 : hole
        """
        sd_polys, hole_polys = [], []
        if r.masks is None or r.boxes is None:
            return sd_polys, hole_polys

        boxes = r.boxes
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confs   = boxes.conf.detach().cpu().numpy()

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

    def pick_targets_like_local(self, sd_poly, hole_polys, vis):
        """
        너가 윈도우에서 로컬로 테스트하던 그 로직을
        frame 한 장 기준으로 단순화해서 넣은 버전
        SD 1개 기준.
        return: (u1, v1, u2, v2) normalized
        """
        H, W = vis.shape[:2]
        x0, y0, bw, bh = self._bbox_of_polygon(sd_poly)
        c = (x0 + bw // 2, y0 + bh // 2)   # bbox 중심

        # 짧은변/긴변 결정
        if bw < bh:
            s = np.array([1.0, 0.0]); l = np.array([0.0, 1.0])
            Ll = bh / 2.0
            Ls = bw / 2.0
        else:
            s = np.array([0.0, 1.0]); l = np.array([1.0, 0.0])
            Ll = bw / 2.0
            Ls = bh / 2.0

        # 표시
        self._draw_centerline(vis, c, s, (255, 255, 0), 2)
        cv2.circle(vis, c, 6, (255, 0, 0), -1)

        margin_ratio = 0.01
        d_min_ratio  = 0.15
        d_max_ratio  = 0.55
        d_pref_ratio = 0.30
        eps_s_ratio  = 0.10
        eps_l_ratio  = 0.10

        margin = margin_ratio * min(bw, bh)
        d_min  = d_min_ratio  * Ll
        d_max  = d_max_ratio  * Ll
        d_pref = d_pref_ratio * Ll
        eps_s  = eps_s_ratio  * Ls
        eps_l  = eps_l_ratio  * Ll

        # 후보 뽑기
        cand = []
        for hp in hole_polys:
            area = self._contour_area(hp)
            hc = self._centroid_mean(hp)

            # bbox 안쪽
            if not ((x0 + margin) <= hc[0] <= (x0 + bw - margin) and
                    (y0 + margin) <= hc[1] <= (y0 + bh - margin)):
                continue

            v = np.array([hc[0] - c[0], hc[1] - c[1]], dtype=float)
            ts = float(np.dot(v, s))  # 짧은변 방향 좌우
            tl = float(np.dot(v, l))  # 긴변 방향 위아래

            if not (d_min <= abs(tl) <= d_max):
                continue

            cand.append({"pt": hc, "ts": ts, "tl": tl, "area": area})
            cv2.circle(vis, hc, 3, (0, 0, 255), -1)

        # 사분면 분류
        Q1, Q2, Q3, Q4 = [], [], [], []
        for h in cand:
            if   h["ts"] > 0 and h["tl"] > 0: Q1.append(h)
            elif h["ts"] < 0 and h["tl"] > 0: Q2.append(h)
            elif h["ts"] < 0 and h["tl"] < 0: Q3.append(h)
            else:                             Q4.append(h)

        # 대칭 페어 찾기
        pairs = []
        for A_list, B_list in [(Q1, Q3), (Q2, Q4)]:
            for A in A_list:
                for B in B_list:
                    C1 = abs(abs(A["tl"]) - abs(B["tl"]))
                    C2 = abs(A["ts"] + B["ts"])
                    if C1 > eps_l or C2 > eps_s:
                        continue
                    J = 0.6 * C1 + 0.4 * C2 + abs(abs(A["tl"]) - d_pref) + abs(abs(B["tl"]) - d_pref)
                    pairs.append((J, A, B))

        if not pairs:
            # 페어 못찾으면 SD 중심 기준 더미 좌표
            u1 = (c[0] - 0.1 * bw) / W
            v1 = (c[1] - 0.2 * bh) / H
            u2 = (c[0] + 0.1 * bw) / W
            v2 = (c[1] + 0.2 * bh) / H
            return u1, v1, u2, v2

        _, A, B = min(pairs, key=lambda x: x[0])

        # 왼쪽/오른쪽 순서 고정
        if A["ts"] < 0:
            p1 = A["pt"]; p2 = B["pt"]
        else:
            p1 = B["pt"]; p2 = A["pt"]

        # 그리기
        for (name, p, col) in [("Target1", p1, (0,255,0)), ("Target2", p2, (0,255,0))]:
            cv2.circle(vis, tuple(p), 8, col, -1)
            cv2.putText(vis, name, (p[0]+5, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        u1, v1 = p1[0] / W, p1[1] / H
        u2, v2 = p2[0] / W, p2[1] / H
        return u1, v1, u2, v2

    # ==================== 타이머 콜백 ====================
    def timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("camera frame read fail")
            return

        H, W = frame.shape[:2]
        r = self.model.predict(source=frame, verbose=False)[0]

        # 기본 캔버스
        vis = frame.copy()

        # YOLO → 폴리곤
        sd_polys, hole_polys = self.yolo_result_to_polys(r, W, H)

        msg = GratingHoles()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = "logitech_cam"

        if not sd_polys:
            # 그레이팅 못찾으면 fallback
            msg.num_holes = 2
            msg.holes_uv = [0.45, 0.3, 0.45, 0.7]
            self.pub.publish(msg)
            self.get_logger().info("publish(no_sd): [0.45,0.3,0.45,0.7]")
        else:
            # 가장 첫번째 SD에 대해서만 처리
            u1, v1, u2, v2 = self.pick_targets_like_local(sd_polys[0], hole_polys, vis)
            msg.num_holes = 2
            msg.holes_uv = [float(u1), float(v1), float(u2), float(v2)]
            self.pub.publish(msg)
            self.get_logger().info(f"publish(ok): [{u1:.4f},{v1:.4f},{u2:.4f},{v2:.4f}]")

            # SD 윤곽선도 그려주자
            cv2.polylines(vis, [np.array(sd_polys[0], np.int32)], True, (0,255,0), 2)

        # ------------ 여기! 실시간 표시 ------------
        if self.show:
            cv2.imshow("grating_debug", vis)
            # 1을 안 주면 창 멈춘다
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 끔
                self.show = False
                cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = GratingDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
