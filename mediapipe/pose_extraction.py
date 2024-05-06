import cv2
import mediapipe as mp
import csv
import math

## POSE EXTRACTION

def write_landmarks_to_csv(landmarks, frame_number, csv_data, landmark_type, norm_landmarks, empty=False):
    for idx, landmark in enumerate(landmarks):
        if landmark_type == "POSE":
            if (0 <= idx <= 14) or (idx == 23) or (idx ==24):
                if idx == 0:
                    norm_landmarks["nose_x"] = landmark.x
                    norm_landmarks["nose_y"] = landmark.y
                if idx == 3:
                    norm_landmarks["left_eye_outer_x"] = landmark.x
                if idx == 6:
                    norm_landmarks["right_eye_outer_x"] = landmark.x
                csv_data.append([frame_number, mp_holistic.PoseLandmark(idx).name, landmark.x, landmark.y])
        else:
            if empty:
                csv_data.append([frame_number, landmark_type + "HAND_" + mp_holistic.HandLandmark(idx).name, None, None])
            else:
                csv_data.append([frame_number, landmark_type + "HAND_" + mp_holistic.HandLandmark(idx).name, landmark.x, landmark.y])

video_path = "orange.mp4"
output_csv = "extracted_features.csv"

# Initialize MediaPipe Pose and Drawing utilities
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
csv_data = []
bounding_box = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = holistic.process(frame_rgb)

    norm_landmarks = {}

    if (result.pose_landmarks):
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data, "POSE", norm_landmarks)

        if result.left_hand_landmarks:
            write_landmarks_to_csv(result.left_hand_landmarks.landmark, frame_number, csv_data, "LEFT", norm_landmarks) 
        else:
            write_landmarks_to_csv(mp_holistic.HandLandmark, frame_number, csv_data, "LEFT", norm_landmarks, empty=True)

        if result.right_hand_landmarks:
            write_landmarks_to_csv(result.right_hand_landmarks.landmark, frame_number, csv_data, "RIGHT", norm_landmarks) 
        else:
            write_landmarks_to_csv(mp_holistic.HandLandmark, frame_number, csv_data, "RIGHT", norm_landmarks, empty=True)

    # frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # # Draw face landmarks
    # mp_drawing.draw_landmarks(frame_rgb, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    
    # # Right hand
    # mp_drawing.draw_landmarks(frame_rgb, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # # Left Hand
    # mp_drawing.draw_landmarks(frame_rgb, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # # Pose Detections
    # mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    

    # # Display the frame
    # cv2.imshow('MediaPipe', frame_rgb)
        nose_x = norm_landmarks["nose_x"]
        nose_y = norm_landmarks["nose_y"]
        left_eye_outer_x = norm_landmarks["left_eye_outer_x"]
        right_eye_outer_x = norm_landmarks["right_eye_outer_x"]

        head_unit = abs(right_eye_outer_x - left_eye_outer_x)
        height_unit = head_unit * (frame.shape[1] / frame.shape[0])
        lower_x = nose_x - (4 * head_unit)
        upper_x = nose_x + (4 * head_unit)
        lower_y = nose_y - (1.5 * height_unit) 
        upper_y = nose_y + (7 * height_unit)
        bounding_box[frame_number] = (lower_x, upper_x, lower_y, upper_y)

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

## POSE NORMALIZATION

def pose_normalization(old_data, bounds):
    normalized = []
    for feature in old_data:
        frame_num, name, x, y = feature
        if x is None and y is None:
            new_x = 0
            new_y = 0
        else:
            new_x = (x - bounds[frame_num][0]) / (bounds[frame_num][1] - bounds[frame_num][0])
            new_y = (y - bounds[frame_num][2]) / (bounds[frame_num][3] - bounds[frame_num][2])
        normalized.append([frame_num, name, new_x, new_y])
    return normalized

normalized_csv_data = pose_normalization(csv_data, bounding_box)

# Save the CSV data to a file
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y'])
    csv_writer.writerows(normalized_csv_data)
