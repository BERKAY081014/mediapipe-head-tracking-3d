#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Proje: MediaPipe ile Head-Tracking & Dinamik Paralaks 3D Simülasyonu
Geliştirici: Berkay Bilgin (https://github.com/BERKAY081014)
Kütüphaneler: OpenCV, MediaPipe, NumPy
Amaç: Web kamera üzerinden yüzün 3D uzaydaki (X, Y, Z) konumunu gerçek zamanlı
      takip ederek ekranda derinlik (hologram/paralaks) hissi oluşturan simülasyon.
===============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class HeadTracking3D:
    def __init__(self, camera_id=0, screen_w=1280, screen_h=720):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Kamera Başlat
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # MediaPipe Face Mesh Başlat
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # 3D Küp Köşe Noktaları (Dünya Koordinat Sistemi)
        self.cube_size = 180
        s = self.cube_size // 2
        self.cube_vertices = np.array([
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s]
        ], dtype=np.float32)

        # Küp Çizgileri Bağlantı İndeksleri
        self.cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Filtreleme (Yumuşak Hareket - EMA Smoothing)
        self.smooth_x = screen_w // 2
        self.smooth_y = screen_h // 2
        self.smooth_z = 500.0
        self.alpha = 0.25 # Düşük değer = daha pürüzsüz

    def calculate_head_position(self, landmarks, frame_w, frame_h):
        """
        İki göz bebeği ve burun ucu referansıyla kafa merkezini ve derinliği (Z) hesaplar.
        """
        left_eye = np.array([landmarks[33].x * frame_w, landmarks[33].y * frame_h])
        right_eye = np.array([landmarks[263].x * frame_w, landmarks[263].y * frame_h])
        nose_tip = np.array([landmarks[1].x * frame_w, landmarks[1].y * frame_h])

        # Gözler arası piksel mesafesi (Göz mesafesi büyüdükçe kullanıcı kameraya yakındır)
        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance < 1.0:
            eye_distance = 1.0

        # Yaklaşık Z derinlik formülü (Fokal uzunluk simülasyonu)
        estimated_z = (60.0 * 550.0) / eye_distance

        head_center_x = (left_eye[0] + right_eye[0]) / 2.0
        head_center_y = (left_eye[1] + right_eye[1]) / 2.0

        return head_center_x, head_center_y, estimated_z

    def project_3d_to_2d(self, vertices, camera_pos, focal_length=600):
        """
        Perspektif izdüşüm formülü: X_proj = (X * f) / (Z + dist) + C
        """
        cam_x, cam_y, cam_z = camera_pos
        projected_points = []

        for vertex in vertices:
            x, y, z = vertex
            rel_x = x - cam_x
            rel_y = y - cam_y
            rel_z = z + cam_z

            if rel_z <= 10.0:
                rel_z = 10.0 # Sıfıra bölmeyi engelle

            screen_x = int((rel_x * focal_length) / rel_z + (self.screen_w // 2))
            screen_y = int((rel_y * focal_length) / rel_z + (self.screen_h // 2))
            projected_points.append((screen_x, screen_y))

        return projected_points

    def run(self):
        prev_time = time.time()
        print("[BAŞLADI] Head-Tracking penceresi açıldı. Çıkış için 'ESC' tuşuna basın.")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1) # Ayna etkisi
            fh, fw, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            # Siyah Render Sahnesi Oluştur
            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Izgara Zemin Çizimi (Derinlik Hissini Güçlendirir)
            for grid_x in range(100, self.screen_w, 100):
                cv2.line(canvas, (grid_x, self.screen_h - 120), (grid_x, self.screen_h), (20, 40, 70), 1)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                raw_x, raw_y, raw_z = self.calculate_head_position(landmarks, fw, fh)

                # Ekran oranına ölçekle
                norm_x = (raw_x - (fw / 2.0)) * 2.8
                norm_y = (raw_y - (fh / 2.0)) * 2.8

                # Üstel Düzeltme (Smoothing)
                self.smooth_x = self.smooth_x + self.alpha * (norm_x - self.smooth_x)
                self.smooth_y = self.smooth_y + self.alpha * (norm_y - self.smooth_y)
                self.smooth_z = self.smooth_z + self.alpha * (raw_z - self.smooth_z)

            # Paralaks İzdüşüm Noktaları
            cam_pos = (self.smooth_x, self.smooth_y, self.smooth_z)
            points_2d = self.project_3d_to_2d(self.cube_vertices, cam_pos)

            # 3D Küp Çizimi (Neon Mavi ve Cyan Çizgiler)
            for edge in self.cube_edges:
                pt1 = points_2d[edge[0]]
                pt2 = points_2d[edge[1]]
                cv2.line(canvas, pt1, pt2, (255, 180, 40), 2, cv2.LINE_AA)

            for pt in points_2d:
                cv2.circle(canvas, pt, 5, (0, 230, 255), -1, cv2.LINE_AA)

            # FPS ve Bilgi Paneli
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            cv2.putText(canvas, f"FPS: {int(fps)} | X: {int(self.smooth_x)} Y: {int(self.smooth_y)} Z: {int(self.smooth_z)}",
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            cv2.putText(canvas, "Berkay Bilgin | MediaPipe Head-Tracking Paralaks Demo",
                        (30, self.screen_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 160, 200), 1)

            # Küçük Kamera Önizlemesi (Sağ Üst Köşe)
            thumb = cv2.resize(frame, (180, 135))
            canvas[20:155, self.screen_w - 200:self.screen_w - 20] = thumb
            cv2.rectangle(canvas, (self.screen_w - 200, 20), (self.screen_w - 20, 155), (0, 200, 255), 1)

            cv2.imshow("3D Head-Tracking Simülasyonu", canvas)
            if cv2.waitKey(1) & 0xFF == 27: # ESC ile çıkış
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HeadTracking3D()
    app.run()\n