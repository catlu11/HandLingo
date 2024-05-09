import cv2
import mediapipe as mp
import csv
import numpy as np
from scipy.spatial.distance import euclidean

def extract_keyframes(video_path):
    video = cv2.VideoCapture(video_path)
    keyframes = set()
    prev_gray = None
    success, prev_frame = video.read()
    count = 0

    hist_diffs = []
    prev_curr = []
    while success:
        success, curr_frame = video.read()
        if not success:
            break
        
        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Compute histograms for grayscale frames
            hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0,256])
            hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0,256])

            # Typecast for compareHist assertion
            hist_prev = np.squeeze(np.float32(hist_prev))
            hist_curr = np.squeeze(np.float32(hist_curr))

            # Compute histogram difference between successive frames
            hist_diff = euclidean(hist_prev, hist_curr)
            hist_diffs.append(hist_diff)

        # Set current frame as previous for the next iteration
        prev_curr.append((prev_gray, curr_gray))
        prev_gray = curr_gray.copy()
        count += 1

    prev_curr = prev_curr[1:]
    assert len(prev_curr) == len(hist_diffs)

    # Calculate mean and standard deviation of histogram difference
    mean, std_dev = np.mean(hist_diffs), np.std(hist_diffs)

    # Compute threshold value "Th"
    Th = mean + 0.5 * std_dev

    for i, hist_pair in enumerate(prev_curr):
        prev, curr = hist_pair

        # Calculate Euclidean distance "Ed"
        Ed = np.linalg.norm(prev - curr)

        if Ed > Th:
            # Add current frame to keyframes set
            keyframes.add(i+1)

    video.release()
    return keyframes

def write_landmarks_to_csv(landmarks, frame_number, csv_data, landmark_type, norm_landmarks, empty=False):
    mp_holistic = mp.solutions.holistic

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

def pose_normalization(old_data, bounds):
    normalized = []
    lwx, lwy, rwx, rwy = None, None, None, None
    for idx, feature in enumerate(old_data):
        frame_num, name, x, y = feature
        if x is None and y is None:
            new_x = 0
            new_y = 0
        else:
            new_x = (x - bounds[frame_num][0]) / (bounds[frame_num][1] - bounds[frame_num][0])
            new_y = (y - bounds[frame_num][2]) / (bounds[frame_num][3] - bounds[frame_num][2])

        if (idx % 118) == 17:
            lwx, lwy = new_x, new_y
        if (idx % 118) == 38:
            rwx, rwy = new_x, new_y
        if (idx % 118) > 17 and (idx % 118) < 38:
            new_x -= lwx
            new_y -= lwy
        if (idx % 118) > 38:
            new_x -= rwx
            new_y -= rwy

        normalized.append([frame_num, name, new_x, new_y])
    return normalized

def get_landmarks(video_path, output_csv, keyframes):
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

        if frame_number not in keyframes:
            frame_number += 1
            continue

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        result = holistic.process(frame_rgb)

        norm_landmarks = {}

        # Draw the pose and hand landmarks on the frame

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

    normalized_csv_data = pose_normalization(csv_data, bounding_box)

    # Save the CSV data to a file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'landmark', 'x', 'y'])
        csv_writer.writerows(normalized_csv_data)
