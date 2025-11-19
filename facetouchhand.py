import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2)

# Use the tesselation connections from MediaPipe
face_triangles = np.array(list(mp_face_mesh.FACEMESH_TESSELATION))

TOUCH_DIST_2D = 20  # pixels
TOUCH_DIST_Z = 0.02

cap = cv2.VideoCapture(1)  # Use 0 for Mac built-in camera
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    touch_points = []

    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        face_pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks])

        # Build a 2D Delaunay triangulation of face landmarks
        tri = Delaunay(face_pts[:, :2])

        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 1️⃣ Check each hand landmark (fingertips, joints, wrist)
            for h_lm in hand_landmarks.landmark:
                hx, hy, hz = h_lm.x * w, h_lm.y * h, h_lm.z
                point = np.array([hx, hy])

                simplex = tri.find_simplex(point)
                if simplex >= 0:
                    tri_vertices = tri.simplices[simplex]
                    tri_z = np.mean(face_pts[tri_vertices, 2])
                    if abs(hz - tri_z) < TOUCH_DIST_Z:
                        touch_points.append((int(hx), int(hy)))

            # 2️⃣ Check intermediate points along each hand connection (edges)
            for i1, i2 in HAND_CONNECTIONS:
                pt1 = hand_landmarks.landmark[i1]
                pt2 = hand_landmarks.landmark[i2]

                num_samples = 5  # How many points to sample along each edge
                for t in np.linspace(0, 1, num_samples):
                    hx = (pt1.x * (1 - t) + pt2.x * t) * w
                    hy = (pt1.y * (1 - t) + pt2.y * t) * h
                    hz = (pt1.z * (1 - t) + pt2.z * t)
                    point = np.array([hx, hy])

                    simplex = tri.find_simplex(point)
                    if simplex >= 0:
                        tri_vertices = tri.simplices[simplex]
                        tri_z = np.mean(face_pts[tri_vertices, 2])
                        if abs(hz - tri_z) < TOUCH_DIST_Z:
                            touch_points.append((int(hx), int(hy)))

    # Draw overlays
    if face_results.multi_face_landmarks:
        mp_draw.draw_landmarks(frame, face_results.multi_face_landmarks[0],
                               mp_face_mesh.FACEMESH_TESSELATION)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Apply red glow overlay at detected touch points
    if len(touch_points) > 0:
        overlay = frame.copy()
        for (tx, ty) in touch_points:
            cv2.circle(overlay, (tx, ty), 40, (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
