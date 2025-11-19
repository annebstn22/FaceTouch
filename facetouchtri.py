import cv2
import mediapipe as mp
import numpy as np

touch_count = 0  # Total face touches
was_touching = False  # Track if previous frame had touch


# --- Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# MediaPipe Face Mesh Triangulation
# These are the actual triangles from the canonical face mesh topology
# 468 landmarks form approximately 800+ triangles
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

# --- Geometry helpers ---
def is_front_facing(p1, p2, p3):
    """Check if a triangle faces the camera (positive Z normal component)."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    # MediaPipe: camera looks along -Z, so front-facing normals point toward -Z
    return normal[2] < 0

def point_to_triangle_distance(p, a, b, c):
    """
    Compute the closest distance from point p to triangle (a,b,c).
    This includes both perpendicular distance to the plane and distance to edges/vertices.
    """
    # First compute distance to the plane
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_len = np.linalg.norm(normal)
    
    if normal_len < 1e-9:
        # Degenerate triangle
        return np.inf
    
    normal = normal / normal_len
    
    # Project point onto plane
    ap = p - a
    dist_to_plane = np.dot(ap, normal)
    proj_p = p - dist_to_plane * normal
    
    # Check if projection is inside triangle using barycentric coordinates
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
    
    # Check if point is inside triangle
    if (u >= 0) and (v >= 0) and (u + v <= 1):
        return abs(dist_to_plane)
    
    # Point projection is outside triangle, find distance to closest edge/vertex
    # Distance to edges
    dist_ab = point_to_segment_distance(p, a, b)
    dist_bc = point_to_segment_distance(p, b, c)
    dist_ca = point_to_segment_distance(p, c, a)
    
    return min(dist_ab, dist_bc, dist_ca)

def point_to_segment_distance(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-10:
        return np.linalg.norm(ap)
    
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# --- Main loop ---
frame_count = 0
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
    touching = False
    frame_count += 1

    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        face_pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])

        for hand_landmarks in hand_results.multi_hand_landmarks:
            for h_idx, h_lm in enumerate(hand_landmarks.landmark):
                hand_pt = np.array([h_lm.x, h_lm.y, h_lm.z])
                
                min_dist = np.inf
                closest_triangle = None

                # Check against all face triangles
                for (i1, i2, i3) in FACE_MESH_TRIANGLES:
                    if i1 >= len(face_pts) or i2 >= len(face_pts) or i3 >= len(face_pts):
                        continue
                        
                    v1, v2, v3 = face_pts[i1], face_pts[i2], face_pts[i3]

                    # Skip back-facing triangles (optimization)
                    if not is_front_facing(v1, v2, v3):
                        continue
                    
                    # Quick XY bounding box check (optimization)
                    min_x = min(v1[0], v2[0], v3[0]) - 0.05
                    max_x = max(v1[0], v2[0], v3[0]) + 0.05
                    min_y = min(v1[1], v2[1], v3[1]) - 0.05
                    max_y = max(v1[1], v2[1], v3[1]) + 0.05
                    
                    if not (min_x <= hand_pt[0] <= max_x and min_y <= hand_pt[1] <= max_y):
                        continue

                    dist = point_to_triangle_distance(hand_pt, v1, v2, v3)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_triangle = (v1, v2, v3)

                # Check if touch detected
                if min_dist < TOUCH_DIST_THRESHOLD and closest_triangle is not None:
                    v1, v2, v3 = closest_triangle
                    # Ensure hand is in front of face (not behind)
                    avg_face_z = (v1[2] + v2[2] + v3[2]) / 3
                    
                    if hand_pt[2] <= avg_face_z + 0.05:
                        hx, hy = int(h_lm.x * w), int(h_lm.y * h)
                        touch_points.append((hx, hy))
                        touching = True

    if touching and not was_touching:
        touch_count += 1  # Increment only on new touch

    # --- Visualization ---
    if face_results.multi_face_landmarks:
        mp_draw.draw_landmarks(
            frame, 
            face_results.multi_face_landmarks[0], 
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1)
        )

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
            )

    # Apply red glow overlay on detected touch points
    if len(touch_points) > 0:
        overlay = frame.copy()
        for (tx, ty) in touch_points:
            cv2.circle(overlay, (tx, ty), 35, (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Status indicator
    status_text = "TOUCHING FACE!" if touching else "No contact"
    status_color = (0, 0, 255) if touching else (0, 255, 0)
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    # Debug info
    cv2.putText(frame, f"Touch points: {len(touch_points)}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()