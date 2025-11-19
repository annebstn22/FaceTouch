import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

# Initialize MediaPipe components
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2)

TOUCH_DIST_Z = 0.02
COINCIDENCE_THRESHOLD = 2  # require detection in 2+ channels
cap = cv2.VideoCapture(1)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)


def detect_touches(face_results, hand_results, h, w):
    """Finds hand-face intersection points using geometric logic."""
    touches = []
    if not (face_results.multi_face_landmarks and hand_results.multi_hand_landmarks):
        return touches

    face_landmarks = face_results.multi_face_landmarks[0].landmark
    face_pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks])
    tri = Delaunay(face_pts[:, :2])

    for hand_landmarks in hand_results.multi_hand_landmarks:
        for h_lm in hand_landmarks.landmark:
            hx, hy, hz = h_lm.x * w, h_lm.y * h, h_lm.z
            simplex = tri.find_simplex(np.array([hx, hy]))
            if simplex >= 0:
                tri_vertices = tri.simplices[simplex]
                tri_z = np.mean(face_pts[tri_vertices, 2])
                if abs(hz - tri_z) < TOUCH_DIST_Z:
                    touches.append((int(hx), int(hy)))
    return touches


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Split into channels and make 3-channel grayscale copies for MediaPipe
    b, g, r = cv2.split(frame)
    r_img = cv2.merge([r, r, r])
    g_img = cv2.merge([g, g, g])
    b_img = cv2.merge([b, b, b])

    # Run MediaPipe on each color channel separately
    r_face = face_mesh.process(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
    g_face = face_mesh.process(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))
    b_face = face_mesh.process(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))

    # Hands only need to be detected once (color invariant)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    # Detect geometric touches for each channel
    r_touches = detect_touches(r_face, hand_results, h, w)
    g_touches = detect_touches(g_face, hand_results, h, w)
    b_touches = detect_touches(b_face, hand_results, h, w)

    # Merge and count coincidences
    all_touches = r_touches + g_touches + b_touches
    coincident_touches = []
    for (x, y) in set(all_touches):
        count = all_touches.count((x, y))
        if count >= COINCIDENCE_THRESHOLD:
            coincident_touches.append((x, y))

    # Visualization
    overlay = frame.copy()
    for (x, y) in coincident_touches:
        cv2.circle(overlay, (x, y), 40, (0, 0, 255), -1)

    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

