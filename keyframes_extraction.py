import cv2
import numpy as np

def extract_keyframes(video_path):
    video = cv2.VideoCapture(video_path)
    keyframes = set()
    prev_gray = None
    success, prev_frame = video.read()
    count = 0

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

            # Normalize histograms
            hist_prev = cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_curr = cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Compute histogram difference between successive frames
            hist_diff = cv2.compareHist(prev_gray, curr_gray, cv2.HISTCMP_CORREL)
            # Calculate mean and standard deviation of histogram difference
            mean, std_dev = cv2.meanStdDev(hist_diff)

            # Compute threshold value "Th"
            Th = mean + 0.5 * std_dev

            # Calculate Euclidean distance "Ed"
            Ed = np.linalg.norm(prev_gray - curr_gray)

            if Ed > Th:
                # Add current frame to keyframes set
                keyframes.add(count)
        
        # Set current frame as previous for the next iteration
        prev_gray = curr_gray.copy()
        count += 1

    video.release()
    return keyframes

keyframes = extract_keyframes("videos/orange.mp4")
print(keyframes)

