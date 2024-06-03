import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Face Mesh and Pose
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize drawing utilities for face and pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Feedback thresholds and buffer for natural movement
POSTURE_THRESHOLD = 0.008
POSTURE_BUFFER_FRAMES = 20  # Number of frames to allow bad posture before alert
GAZE_THRESHOLD = 0.0015
EMOTION_MOVEMENT_AVG_PERIOD = 30
GAZE_BUFFER_FRAMES = 30     # Number of frames to allow gaze outside before alert
LAPTOP_SCREEN_RATIO = (0.30, 0.7)  # Horizontal screen range considered as 'focused'

# Video capture setup
cap = cv2.VideoCapture(0)
previous_iris_position = None
iris_movement = []
start_time = time.time()

# Counters and buffers
total_frames = 0
correct_posture_frames = 0
focused_gaze_frames = 0
bad_posture_count = 0
out_of_focus_count = 0

def analyze_frame(image, previous_iris_position):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(image_rgb)
    results_pose = pose.process(image_rgb)
    image_with_annotations = image.copy()

    global correct_posture_frames, focused_gaze_frames, bad_posture_count, out_of_focus_count

    # Draw only face landmarks without connections
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image_with_annotations,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec
            )

    # Analyze shoulder posture using pose landmarks
    if results_pose.pose_landmarks:
        left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)

        if shoulder_slope < POSTURE_THRESHOLD:
            correct_posture_frames += 1
            bad_posture_count = 0  # reset the bad posture counter
        else:
            bad_posture_count += 1

        if bad_posture_count > POSTURE_BUFFER_FRAMES:
            posture_feedback = "Adjust posture: Sit straight"
        else:
            posture_feedback = "Posture is good"
        cv2.putText(image_with_annotations, posture_feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Gaze analysis
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            right_iris = [face_landmarks.landmark[i] for i in range(468, 473)]
            left_iris = [face_landmarks.landmark[i] for i in range(473, 478)]
            right_iris_center = np.mean([[p.x, p.y] for p in right_iris], axis=0)
            left_iris_center = np.mean([[p.x, p.y] for p in left_iris], axis=0)
            current_iris_position = np.mean([right_iris_center, left_iris_center], axis=0)

            if previous_iris_position is not None:
                iris_displacement = np.linalg.norm(current_iris_position - previous_iris_position)
                iris_movement.append(iris_displacement)

                if len(iris_movement) > EMOTION_MOVEMENT_AVG_PERIOD:
                    avg_movement = np.mean(iris_movement[-EMOTION_MOVEMENT_AVG_PERIOD:])
                    if avg_movement > GAZE_THRESHOLD:
                        out_of_focus_count += 1
                    else:
                        out_of_focus_count = 0  # reset the out of focus counter

                    if out_of_focus_count > GAZE_BUFFER_FRAMES:
                        emotion_feedback = "Please focus: Look at the screen"
                    else:
                        emotion_feedback = "Focused gaze"
                    cv2.putText(image_with_annotations, emotion_feedback, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            previous_iris_position = current_iris_position

    return image_with_annotations, previous_iris_position

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    processed_image, previous_iris_position = analyze_frame(frame, previous_iris_position)
    cv2.imshow('Real-time Analysis', processed_image)
    total_frames += 1

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate session summary
end_time = time.time()
session_duration = (end_time - start_time) / 60  # duration in minutes
correct_posture_percentage = (correct_posture_frames / total_frames) * 100
focused_gaze_percentage = (focused_gaze_frames / total_frames) * 100

print(f"Session Duration: {session_duration:.2f} minutes")
print(f"Correct Posture Percentage: {correct_posture_percentage:.2f}%")
print(f"Total Frames Analyzed: {total_frames}")
print(f"Frames with Correct Posture: {correct_posture_frames}")

