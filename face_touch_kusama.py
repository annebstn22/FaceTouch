import cv2
import mediapipe as mp
import numpy as np
import socket
import json

# UDP socket to overlay
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_dot(region):
    msg = json.dumps({"region": region})
    sock.sendto(msg.encode(), ("127.0.0.1", 5055))


# -------- Face region sets --------
# You can adjust these any time
FACE_REGIONS = {
    "orange": set(range(10, 110)),       # forehead
    "yellow": {1,2,3,4,5,197,6},         # nose bridge
    "pink": set(range(234, 253)),        # left cheek
    "red": set(range(350, 455)),         # right cheek
    "blue": set(range(152, 200)),        # chin
    "green": set(range(127, 152)),       # jawline
}

def get_triangle_region(i1, i2, i3):
    for region, ids in FACE_REGIONS.items():
        if (i1 in ids) or (i2 in ids) or (i3 in ids):
            return region
    return None


# ===== Your original MediaPipe setup =====
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# -- Your triangulation list stays unchanged --
FACE_MESH_TRIANGLES = [
    (127, 34, 139), (11, 0, 37), (232, 231, 120), (72, 37, 39), (128, 121, 47),
    (232, 121, 128), (104, 69, 67), (175, 171, 148), (118, 50, 101), (73, 39, 40),
    (9, 151, 108), (48, 115, 131), (194, 204, 211), (74, 40, 185), (80, 81, 82),
    (38, 122, 212), (62, 76, 77), (191, 80, 178), (61, 185, 40), (146, 91, 181),
    (84, 17, 314), (90, 77, 146), (156, 70, 63), (160, 159, 158), (127, 139, 34),
    (156, 63, 62), (21, 54, 103), (67, 109, 10), (38, 20, 212), (221, 222, 223),
    (162, 21, 54), (67, 69, 108), (37, 0, 267), (232, 120, 231), (72, 38, 37),
    (128, 232, 121), (104, 67, 69), (175, 148, 171), (118, 101, 50), (73, 40, 39),
    (151, 9, 108), (48, 131, 115), (194, 211, 204), (74, 185, 40), (80, 82, 81),
    (122, 38, 212), (62, 77, 76), (191, 178, 80), (185, 61, 40), (146, 181, 91),
    (17, 84, 314), (77, 90, 146), (70, 156, 63), (159, 160, 158), (139, 127, 34),
    (63, 156, 62), (54, 21, 103), (109, 67, 10), (20, 38, 212), (222, 221, 223),
    (21, 162, 54), (69, 67, 108), (0, 37, 267), (356, 454, 361), (279, 420, 360),
    (363, 281, 360), (456, 252, 253), (251, 252, 256), (341, 463, 263), (446, 265, 342),
    (342, 445, 446), (383, 300, 368), (374, 387, 373), (387, 374, 386), (466, 388, 387),
    (356, 361, 454), (420, 279, 360), (281, 363, 360), (252, 456, 253), (252, 251, 256),
    (463, 341, 263), (265, 446, 342), (445, 342, 446), (300, 383, 368), (387, 374, 373),
    (374, 387, 386), (388, 466, 387), (264, 447, 366), (368, 300, 264), (353, 265, 353),
    (376, 433, 352), (352, 345, 376), (280, 425, 411), (411, 425, 352), (352, 425, 280),
    (447, 264, 366), (300, 368, 264), (265, 353, 353), (433, 376, 352), (345, 352, 376),
    (425, 280, 411), (425, 411, 352), (425, 352, 280), (371, 266, 329), (266, 329, 330),
    (266, 330, 329), (329, 266, 371), (330, 266, 329), (330, 329, 266), (371, 329, 266),
    (329, 330, 266), (127, 162, 21), (234, 127, 162), (162, 127, 21), (21, 127, 162),
    (234, 162, 127), (162, 234, 127), (139, 127, 162), (127, 139, 162), (162, 127, 139),
    (356, 389, 368), (389, 356, 264), (264, 356, 389), (368, 389, 264), (389, 368, 356),
    (264, 389, 356), (454, 356, 323), (356, 454, 389), (389, 454, 356), (323, 356, 454),
    (356, 323, 454), (454, 323, 356), (251, 389, 388), (389, 251, 356), (356, 251, 389),
    (388, 389, 251), (389, 388, 251), (251, 388, 389), (6, 197, 195), (197, 5, 4),
    (5, 197, 195), (195, 197, 5), (4, 5, 197), (197, 4, 5), (164, 165, 167), (165, 164, 92),
    (92, 164, 165), (167, 165, 92), (165, 167, 164), (92, 165, 164), (40, 39, 37),
    (39, 40, 185), (185, 40, 39), (37, 39, 185), (39, 37, 40), (185, 39, 40),
    (269, 270, 409), (270, 269, 291), (291, 269, 270), (409, 270, 291), (270, 409, 269),
    (291, 270, 269), (84, 83, 18), (83, 84, 17), (17, 84, 83), (18, 83, 17), (83, 18, 84),
    (17, 83, 84), (314, 315, 17), (315, 314, 83), (83, 314, 315), (17, 315, 83),
    (315, 17, 314), (83, 315, 314), (78, 14, 317), (14, 78, 87), (87, 78, 14), (317, 14, 87),
    (14, 317, 78), (87, 14, 78), (78, 308, 324), (308, 78, 191), (191, 78, 308),
    (324, 308, 191), (308, 324, 78), (191, 308, 78), (78, 191, 80), (191, 78, 81),
    (81, 78, 191), (80, 191, 81), (191, 80, 78), (81, 191, 78)
]

TOUCH_DIST_THRESHOLD = 0.025  # Distance threshold in normalized 3D units

cap = cv2.VideoCapture(1)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

# ===== Helper geometry functions (unchanged from your code) =====
def is_front_facing(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal[2] < 0

def point_to_triangle_distance(p, a, b, c):
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-9:
        return np.inf
    normal = normal / normal_len
    ap = p - a
    dist_to_plane = np.dot(ap, normal)
    proj_p = p - dist_to_plane * normal
    v0 = c - a
    v1 = b - a
    v2 = proj_p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    if (u >= 0) and (v >= 0) and (u + v <= 1):
        return abs(dist_to_plane)
    return min(
        point_to_segment_distance(p, a, b),
        point_to_segment_distance(p, b, c),
        point_to_segment_distance(p, c, a)
    )

def point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-10:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# ===== Main loop =====
was_touching = False
touch_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    touching = False
    touched_regions = set()

    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        face_pts = np.array([[lm.x, lm.y, lm.z] 
                             for lm in face_results.multi_face_landmarks[0].landmark])

        for hand_landmarks in hand_results.multi_hand_landmarks:
            for h_lm in hand_landmarks.landmark:
                hand_pt = np.array([h_lm.x, h_lm.y, h_lm.z])
                min_dist = np.inf
                best_region = None

                for (i1, i2, i3) in FACE_MESH_TRIANGLES:
                    if i1 >= len(face_pts) or i2 >= len(face_pts) or i3 >= len(face_pts):
                        continue
                    v1, v2, v3 = face_pts[i1], face_pts[i2], face_pts[i3]
                    if not is_front_facing(v1, v2, v3):
                        continue

                    # bounding box reject
                    if not (min(v1[0], v2[0], v3[0]) - 0.05 <= hand_pt[0] <= max(v1[0], v2[0], v3[0]) + 0.05):
                        continue
                    if not (min(v1[1], v2[1], v3[1]) - 0.05 <= hand_pt[1] <= max(v1[1], v2[1], v3[1]) + 0.05):
                        continue

                    dist = point_to_triangle_distance(hand_pt, v1, v2, v3)
                    if dist < min_dist:
                        region = get_triangle_region(i1, i2, i3)
                        if region is not None:
                            min_dist = dist
                            best_region = region

                if min_dist < TOUCH_DIST_THRESHOLD and best_region is not None:
                    touching = True
                    touched_regions.add(best_region)

    # Trigger dot events *once per new touch*
    if touching and not was_touching:
        for region in touched_regions:
            print("Sending dot for region:", region)
            send_dot(region)
        touch_count += 1

    was_touching = touching

    # UI
    cv2.putText(frame,
                "TOUCHING FACE!" if touching else "No contact",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,0,255) if touching else (0,255,0),
                3)

    cv2.putText(frame, f"Face touches: {touch_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
