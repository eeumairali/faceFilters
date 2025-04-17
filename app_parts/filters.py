import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

# Initialize MediaPipe models
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Original filters
def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def bitwise_not(frame):
    return cv2.bitwise_not(frame)


def sepia(frame):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    return cv2.transform(frame, kernel)


def make_cartoon(frame):
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 10)
    return cv2.bitwise_and(color, color, mask=edges)


def sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(gray, 255 - blur, scale=256)


def blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)


def negative(frame):
    return cv2.bitwise_not(frame)


def emboss(frame):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    return cv2.filter2D(frame, -1, kernel)


def edges(frame):
    return cv2.Canny(frame, 100, 200)


def stylize(frame):
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)


def enhance(frame):
    return cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)


def thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def glitch(frame):
    h, w, _ = frame.shape
    glitch = frame.copy()
    for i in range(5):
        y = np.random.randint(0, h)
        glitch[y:y+1, :] = np.roll(glitch[y:y+1, :], np.random.randint(-20, 20), axis=1)
    return glitch


# New filters using MediaPipe
def face_mesh_tesselation(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        annotated_frame = frame.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        return annotated_frame
    return frame


def face_contours(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        annotated_frame = frame.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # Try to draw irises if available
            try:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            except:
                pass  # If irises are not available in this version of MediaPipe
                
        return annotated_frame
    return frame


def pose_detection(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return annotated_frame
    return frame


def holistic_detection(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)
    
    annotated_frame = frame.copy()
    
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
    return annotated_frame


def disco_lights(frame):
    # Apply a color cycling effect
    h, w, _ = frame.shape
    
    # Create a cycling color overlay
    t = cv2.getTickCount() / cv2.getTickFrequency()
    r = np.sin(t * 2.0) * 127 + 128
    g = np.sin(t * 2.0 + 2.0) * 127 + 128
    b = np.sin(t * 2.0 + 4.0) * 127 + 128
    
    # Create a color mask
    color_mask = np.ones((h, w, 3), dtype=np.uint8)
    color_mask[:,:,0] = b
    color_mask[:,:,1] = g
    color_mask[:,:,2] = r
    
    # Apply the mask with alpha blending
    alpha = 0.3
    blended = cv2.addWeighted(frame, 1-alpha, color_mask, alpha, 0)
    
    # Apply holistic detection for added fun
    return holistic_detection(blended)


def rainbow_gradient(frame):
    h, w, _ = frame.shape
    
    # Create a rainbow gradient
    gradient = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        hue = int((i / h) * 180) % 180  # Hue ranges from 0 to 180 in OpenCV
        color = np.ones((1, w, 3), dtype=np.uint8) * 255
        color[0, :, 0] = hue
        color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        gradient[i, :] = color
    
    # Blend with original frame
    alpha = 0.5
    blended = cv2.addWeighted(frame, 1-alpha, gradient, alpha, 0)
    
    return blended


def pixel_sort(frame):
    # Simple pixel sorting effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Sort pixels in blocks
    block_size = 20
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = frame[i:min(i+block_size, h), j:min(j+block_size, w)]
            if np.random.random() > 0.7:  # Randomly choose blocks to sort
                # Sort by brightness
                flat_block = block.reshape(-1, 3)
                brightness = np.mean(flat_block, axis=1)
                sorted_idx = np.argsort(brightness)
                sorted_flat = flat_block[sorted_idx]
                frame[i:min(i+block_size, h), j:min(j+block_size, w)] = sorted_flat.reshape(block.shape)
    
    return frame


def mirror_effect(frame):
    h, w, _ = frame.shape
    
    # Create a mirror effect by flipping the left side
    left = frame[:, :w//2]
    flipped_left = cv2.flip(left, 1)
    
    # Create mirrored frame
    mirrored = frame.copy()
    mirrored[:, w//2:] = flipped_left
    
    return mirrored


def kaleidoscope(frame):
    h, w, _ = frame.shape
    
    # Create a triangular segment from the center
    center_x, center_y = w // 2, h // 2
    segment_size = min(center_x, center_y)
    
    # Create an output frame
    output = np.zeros_like(frame)
    
    # Create a triangular segment
    triangle = frame[center_y-segment_size:center_y, center_x-segment_size:center_x]
    
    # Rotate and copy the triangle to create a kaleidoscope effect
    for angle in range(0, 360, 60):
        M = cv2.getRotationMatrix2D((segment_size/2, segment_size/2), angle, 1)
        rotated = cv2.warpAffine(triangle, M, (segment_size, segment_size))
        
        # Place rotated triangle in output
        try:
            output[center_y:center_y+segment_size, center_x:center_x+segment_size] = rotated
        except:
            pass  # Handle size mismatch
    
    return output


def time_warp(frame):
    # Create a time warping effect with ripples
    h, w, _ = frame.shape
    
    # Create time-based distortion map
    t = cv2.getTickCount() / cv2.getTickFrequency()
    
    # Create distortion maps
    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Add ripple effect
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((map_x - center_x)**2 + (map_y - center_y)**2)
    
    # Calculate displacement
    displacement = np.sin(dist_from_center / 10.0 - t * 5) * 10
    
    # Apply displacement
    map_x = map_x + displacement * (map_x - center_x) / dist_from_center
    map_y = map_y + displacement * (map_y - center_y) / dist_from_center
    
    # Handle division by zero
    map_x[np.isnan(map_x)] = 0
    map_y[np.isnan(map_y)] = 0
    
    # Ensure maps are within bounds
    map_x = np.clip(map_x, 0, w-1)
    map_y = np.clip(map_y, 0, h-1)
    
    # Remap the frame
    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    
    return warped