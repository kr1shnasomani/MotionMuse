<h1 align="center">MotionMuse</h1>
The code detects human poses in images using MediaPipe, calculates joint angles, checks visibility, and classifies poses like standing, sitting, or squatting based on body landmarks and positions.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install opencv-python mediapipe numpy
   ```

2. Enter the path of the image from where you want to detect the pose

3. Run the code and then it will show its prediction

## Model Prediction:

Input Image:

![image](https://github.com/user-attachments/assets/36b6e876-ca56-487d-9b6f-c84217f100e7)

Output:

`Detected Pose: Standing`

## Overview:
The code is designed to detect and classify human poses from an image using the MediaPipe library. Here's an overview of how it works:

1. **Configuration**:
   - **TensorFlow Logging**: The code suppresses TensorFlow warnings and logs, setting the environment variable `TF_CPP_MIN_LOG_LEVEL` to '3' to minimize unnecessary logs.
   - **Logging Setup**: TensorFlow’s logging level is set to `ERROR` to further reduce log clutter.

2. **Library Imports**:
   - **OpenCV (cv2)**: Used for image processing and reading images.
   - **MediaPipe**: A framework used for various computer vision tasks, here it is used for pose detection.
   - **NumPy**: Utilized for mathematical operations, specifically for calculating angles.

3. **MediaPipe Pose Initialization**: The `mp_pose.Pose` object is initialized with specific parameters:
     - `static_image_mode=True`: The pose model works on static images, not video.
     - `min_detection_confidence=0.5`: A detection confidence threshold for identifying poses.
     - `model_complexity=2`: The complexity of the pose model (higher values provide more accuracy but require more computational power).

4. **Angle Calculation Functions**:
   - `calculate_angle(a, b, c)`: Computes the angle between three key points (like joints). This is used for determining joint angles (e.g., knee, elbow).
   - `get_vertical_angle(a, b)`: Calculates the angle between a line and the vertical axis to assess the alignment of body parts like the spine.
   - `check_visibility(landmarks, indices)`: Checks if specific landmarks (like joints) are visible based on a visibility threshold.

5. **Pose Classification**:
   - `get_vertical_position(landmarks)`: Returns the relative vertical positions of certain body parts (e.g., shoulder, hip, knee, and ankle) based on their Y-coordinate in the image. This helps in classifying standing, sitting, or squatting poses.
   - `classify_pose(landmarks)`: This is the core function that performs pose classification. It:
     - Calculates joint angles using `calculate_angle`.
     - Checks the visibility of key body parts.
     - Analyzes the vertical positioning of body parts.
     - Classifies the pose as standing, sitting, squatting, lunging, etc., based on certain thresholds for joint angles and body part positions.
     - Scores various pose types and returns the pose with the highest score if it meets a confidence threshold.

6. **Pose Detection from Image**:
   - `identify_pose(image_path)`: This function processes an input image:
     - It reads the image using OpenCV.
     - Converts the image to RGB format (required by MediaPipe).
     - Processes the image to detect the pose landmarks using the MediaPipe Pose model.
     - If pose landmarks are detected, it calls `classify_pose` to classify the pose and prints the detected pose description.

7. **Main Execution**: The `main` block specifies an image file path (`image_path`) and calls `identify_pose` to detect and classify the pose from that image.

### Pose Classification Criteria:
- **Standing**: Based on the straightness of knees and hips, spine alignment, and vertical positioning of body parts.
- **Sitting**: Assessed by the angle between the knees and hips, indicating a sitting position.
- Other poses like squatting, lunging, and push-ups are also classified using specific angle thresholds and position checks.

### Summary:
This code leverages the MediaPipe Pose detection model to classify human poses in an image based on the angles and relative positions of key body landmarks. It includes logic to evaluate specific poses (like standing, sitting, squatting) and assigns a confidence score to detect the most likely pose.
