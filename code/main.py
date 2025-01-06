# System and logging configuration
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import required libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    model_complexity=2
)

# Calculate the angle between three points
def calculate_angle(a, b, c):
    if not all([a, b, c]) or any(np.isnan([a.x, a.y, b.x, b.y, c.x, c.y])):
        return None
    
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Calculate the angle between a line and the vertical axis
def get_vertical_angle(a, b):
    if not all([a, b]) or any(np.isnan([a.x, a.y, b.x, b.y])):
        return None
    return np.degrees(np.arctan2(b.x - a.x, b.y - a.y))

# Check if specified landmarks are visible
def check_visibility(landmarks, indices):
    return all(landmarks[i].visibility > 0.5 for i in indices)


# Get relative vertical positions of key points
def get_vertical_position(landmarks):
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    knee_y = (landmarks[25].y + landmarks[26].y) / 2
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    
    return {
        'shoulder_to_hip': hip_y - shoulder_y,
        'hip_to_knee': knee_y - hip_y,
        'knee_to_ankle': ankle_y - knee_y
    }

# Classify poses with adjusted scoring system
def classify_pose(landmarks):    
    # Extract key landmarks
    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    elbow_left = landmarks[13]
    elbow_right = landmarks[14]
    wrist_left = landmarks[15]
    wrist_right = landmarks[16]
    hip_left = landmarks[23]
    hip_right = landmarks[24]
    knee_left = landmarks[25]
    knee_right = landmarks[26]
    ankle_left = landmarks[27]
    ankle_right = landmarks[28]
    
    # Calculate angles
    angles = {
        'left_knee': calculate_angle(hip_left, knee_left, ankle_left),
        'right_knee': calculate_angle(hip_right, knee_right, ankle_right),
        'left_hip': calculate_angle(shoulder_left, hip_left, knee_left),
        'right_hip': calculate_angle(shoulder_right, hip_right, knee_right),
        'left_elbow': calculate_angle(shoulder_left, elbow_left, wrist_left),
        'right_elbow': calculate_angle(shoulder_right, elbow_right, wrist_right),
        'spine': get_vertical_angle(hip_left, shoulder_left)
    }
    
    # Check visibility
    visibility = {
        'legs': check_visibility(landmarks, [23, 24, 25, 26, 27, 28]),
        'upper_body': check_visibility(landmarks, [11, 12, 13, 14])
    }
    
    # Get positions
    positions = get_vertical_position(landmarks)
    
    # Initialize pose scores
    pose_scores = {
        "standing": 0,
        "sitting": 0,
        "squatting": 0,
        "lunging": 0,
        "plank": 0,
        "push_up": 0,
        "yoga_warrior": 0,
        "t_pose": 0
    }
    
    # Enhanced standing detection with adjusted scoring
    if visibility['legs'] and visibility['upper_body']:
        # Check leg straightness
        if angles['left_knee'] and angles['right_knee']:
            if angles['left_knee'] > 170 and angles['right_knee'] > 170:
                pose_scores["standing"] += 1.5
            elif angles['left_knee'] > 160 and angles['right_knee'] > 160:
                pose_scores["standing"] += 1.0
        
        # Check hip angles
        if angles['left_hip'] and angles['right_hip']:
            if angles['left_hip'] > 160 and angles['right_hip'] > 160:
                pose_scores["standing"] += 0.5
        
        # Check vertical spine
        if angles['spine']:
            spine_angle_mod = abs(angles['spine']) % 180
            if spine_angle_mod > 170 or spine_angle_mod < 10:
                pose_scores["standing"] += 0.5
        
        # Check vertical alignment
        if (positions['shoulder_to_hip'] > 0 and
            positions['hip_to_knee'] > 0 and
            positions['knee_to_ankle'] > 0):
            pose_scores["standing"] += 0.5

    # Sitting detection
    if visibility['legs'] and visibility['upper_body']:
        if angles['left_knee'] and angles['right_knee']:
            if angles['left_knee'] < 120 and angles['right_knee'] < 120:
                pose_scores["sitting"] += 1.5
        if angles['left_hip'] and angles['right_hip']:
            if angles['left_hip'] < 120 and angles['right_hip'] < 120:
                pose_scores["sitting"] += 1.0
    
    # Get the pose with highest confidence
    detected_pose = max(pose_scores.items(), key=lambda x: x[1])
    
    # Return the detected pose if confidence is high enough
    if detected_pose[1] >= 1.5:
        return detected_pose[0].replace('_', ' ').title()
    else:
        return "Unknown Pose"

# Process image and identify the pose
def identify_pose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No pose detected in the image.")
        return

    landmarks = results.pose_landmarks.landmark
    pose_description = classify_pose(landmarks)
    print(f"Detected Pose: {pose_description}")

if __name__ == "__main__":
    image_path = r"C:\Users\krish\OneDrive\Desktop\image.jpg"
    identify_pose(image_path)