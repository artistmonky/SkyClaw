#!/usr/bin/env python3
"""
Green Object Tracker — publishes real‑time guidance vectors for PX4‑based
precision alignment.
"""

import cv2
import numpy as np
import argparse
import math
import socket, struct
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped

# ------------------ CONFIGURATION ------------------ #
LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([80, 255, 255])

MIN_AREA = 500
ASPECT_RATIO_MIN = 0.75
ASPECT_RATIO_MAX = 1.50
MATCH_DISTANCE = 50
PERSISTENCE_THRESHOLD = 5
PIXEL_TO_METER = 0.005
# --------------------------------------------------- #

class VectorPublisher(Node):
    def __init__(self):
        super().__init__('vector_publisher')
        self.pub = self.create_publisher(Vector3Stamped, 'landing_vector', 10)

    def publish(self, dx_px: float, dy_px: float):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(dx_px * PIXEL_TO_METER)
        msg.vector.y = float(dy_px * PIXEL_TO_METER)
        msg.vector.z = 0.0
        self.pub.publish(msg)

class Track:
    def __init__(self, centroid: tuple[int, int]):
        self.centroids = deque([centroid], maxlen=PERSISTENCE_THRESHOLD)
        self.updated = True

    def mark_new_frame(self):
        self.updated = False

    def update(self, centroid: tuple[int, int]):
        self.centroids.append(centroid)
        self.updated = True

    def is_persistent(self) -> bool:
        return len(self.centroids) == self.centroids.maxlen

    @property
    def last(self) -> tuple[int, int]:
        return self.centroids[-1]

def euclidean(p: tuple[int, int], q: tuple[int, int]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_x', type=int, help='Target pixel X')
    ap.add_argument('--target_y', type=int, help='Target pixel Y')
    ap.add_argument('--camera_id', type=int, default=0, help='Camera index')
    args = ap.parse_args()

    PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", PORT))

    rclpy.init(args=None)
    vector_pub = VectorPublisher()

    tracks: list[Track] = []
    target_point: tuple[int, int] | None = None
    kernel = np.ones((5, 5), np.uint8)

    try:
        while True:
            pkt, _ = sock.recvfrom(65535)
            if len(pkt) < 2:
                continue

            size = struct.unpack("H", pkt[:2])[0]
            frame = cv2.imdecode(np.frombuffer(pkt[2:2+size], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            half_w = w // 2
            left_img = frame[:, :half_w]
            right_img = frame[:, half_w:]
            combined_view = np.hstack((left_img.copy(), right_img.copy()))

            left_mask = None
            right_mask = None
            
            guidance_vectors = []

            for side, eye_img, x_offset in [('left', left_img, 0), ('right', right_img, half_w)]:
                hsv = cv2.cvtColor(eye_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                if side == 'left':
                    left_mask = mask
                else:
                    right_mask = mask

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_mask = np.zeros_like(mask)

                for cnt in contours:
                    cv2.drawContours(valid_mask, [cnt], -1, 255, -1)  # include all blobs

                if cv2.countNonZero(valid_mask) == 0:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(valid_mask)
                cx = x + w_box // 2
                cy = y + h_box // 2

                for tr in tracks:
                    tr.mark_new_frame()

                matched = False
                for tr in tracks:
                    if euclidean((cx + x_offset, cy), tr.last) < MATCH_DISTANCE:
                        tr.update((cx + x_offset, cy))
                        matched = True
                        break
                if not matched:
                    tracks.append(Track((cx + x_offset, cy)))

                for tr in tracks:
                    if not tr.updated or not tr.is_persistent():
                        continue
                    cx_tr, cy_tr = tr.last
                    if target_point is None:
                        target_point = (w // 2, h // 2)
                    dx_px = target_point[0] - cx_tr
                    dy_px = target_point[1] - cy_tr

                    cv2.rectangle(combined_view,
                                  (x + x_offset, y),
                                  (x + x_offset + w_box, y + h_box),
                                  (0, 255, 255), 2)
                    cv2.circle(combined_view, (cx_tr, cy_tr), 5, (0, 255, 0), -1)
                    cv2.arrowedLine(combined_view, (cx_tr, cy_tr), target_point, (255, 0, 0), 2, tipLength=0.2)
                    cv2.putText(combined_view, f'{side.upper()} dX:{dx_px} dY:{dy_px}', (cx_tr + 10, cy_tr - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    guidance_vectors.append((dx_px, dy_px))

            if guidance_vectors:
                avg_dx = sum(v[0] for v in guidance_vectors) / len(guidance_vectors)
                avg_dy = sum(v[1] for v in guidance_vectors) / len(guidance_vectors)
                vector_pub.publish(avg_dx, avg_dy)

                # Draw averaged correction vector from center of combined view
                center = (w // 2, h // 2)
                end_point = (int(center[0] + avg_dx), int(center[1] + avg_dy))
                cv2.arrowedLine(combined_view, center, end_point, (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(combined_view, f'Avg dX:{int(avg_dx)} dY:{int(avg_dy)}',
                            (center[0] + 10, center[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if target_point is not None:
                cv2.circle(combined_view, target_point, 6, (0, 0, 255), -1)
                cv2.putText(combined_view, 'TARGET', (target_point[0] + 8, target_point[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Show windows
            cv2.imshow('Tracker (Left+Right)', combined_view)
            if left_mask is not None:
                cv2.imshow('Mask - Left', left_mask)
            if right_mask is not None:
                cv2.imshow('Mask - Right', right_mask)

            if cv2.waitKey(1) in (27, ord('q')):
                break

    finally:
        cv2.destroyAllWindows()
        vector_pub.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

