import cv2
import mediapipe as mp
import csv

def write_landmarks_to_csv(landmarks, frame_number, csv_data, landmark_type, empty=False):
    for idx, landmark in enumerate(landmarks):
        if landmark_type == "POSE":
            if 0 < idx <= 24:
                csv_data.append([frame_number, mp_holistic.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
        else:
            if empty:
                csv_data.append([frame_number, landmark_type + "_" + mp_holistic.HandLandmark(idx).name, 0, 0, 0])
            else:
                csv_data.append([frame_number, landmark_type + "_" + mp_holistic.HandLandmark(idx).name, landmark.x, landmark.y, landmark.z])

video_path = "orange.mp4"
output_csv = "output.csv"

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
csv_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = holistic.process(frame_rgb)

    # Draw the pose and hand landmarks on the frame

    if (result.left_hand_landmarks or result.right_hand_landmarks):
        if result.left_hand_landmarks:
            write_landmarks_to_csv(result.left_hand_landmarks.landmark, frame_number, csv_data, "LEFT") 
        else:
            write_landmarks_to_csv(mp_holistic.HandLandmark, frame_number, csv_data, "LEFT", empty=True)

        if result.right_hand_landmarks:
            write_landmarks_to_csv(result.right_hand_landmarks.landmark, frame_number, csv_data, "RIGHT") 
        else:
            write_landmarks_to_csv(mp_holistic.HandLandmark, frame_number, csv_data, "RIGHT", empty=True)

        if result.pose_landmarks:
            write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data, "POSE")

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

    frame_number += 1

cap.release()
cv2.destroyAllWindows()


# Save the CSV data to a file
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
    csv_writer.writerows(csv_data)
