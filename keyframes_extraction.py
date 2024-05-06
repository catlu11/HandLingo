import cv2
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

            # Normalize histograms
            #hist_prev = cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #hist_curr = cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Typecast for compareHist assertion
            hist_prev = np.float32(hist_prev)
            hist_curr = np.float32(hist_curr)

            # Compute histogram difference between successive frames
            #hist_diff = cv2.compareHist(prev_gray, curr_gray, cv2.HISTCMP_CORREL)
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

keyframes = extract_keyframes("videos/07076.mp4")
print(keyframes)

