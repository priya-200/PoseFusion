import cv2
import mediapipe as mp
import time
import streamlit as st
import numpy as np

# Mediapipe pose and segmentation setup
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose()
segmenter = mp_segmentation.SelfieSegmentation()

# Load reference pose image
reference_img_path = 'E:\projects\Pose Estimator\images\download.jpeg'
reference_img = cv2.imread(reference_img_path)
reference_img_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)


def get_normalized_landmark(img):
    """
    Get the normalized landmarks of the image.
    """
    results = pose.process(img)
    if not results.pose_landmarks:
        return None
    landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
    hip_midpoint = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    ]).mean(axis=0)
    normalized_landmarks = [(x - hip_midpoint[0], y - hip_midpoint[1], z - hip_midpoint[2])
                            for x, y, z in landmarks]
    return np.array(normalized_landmarks)


def similarity_score(pose1, pose2):
    p1 = []
    p2 = []

    # Ensure correct dtype
    pose_1 = np.array(pose1, dtype=float)
    pose_2 = np.array(pose2, dtype=float)

    # Normalize coordinates
    pose_1[:, 0] = pose_1[:, 0] / max(pose_1[:, 0])
    pose_1[:, 1] = pose_1[:, 1] / max(pose_1[:, 1])
    pose_2[:, 0] = pose_2[:, 0] / max(pose_2[:, 0])
    pose_2[:, 1] = pose_2[:, 1] / max(pose_2[:, 1])

    # Flatten poses into 1D arrays
    for joint in range(pose_1.shape[0]):
        x1 = pose_1[joint][0]
        y1 = pose_1[joint][1]
        x2 = pose_2[joint][0]
        y2 = pose_2[joint][1]

        p1.append(x1)
        p1.append(y1)
        p2.append(x2)
        p2.append(y2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Compute cosine similarity
    scoreA = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

    # Compute weighted distance (a simple L2 distance metric)
    scoreB = np.sum(np.abs(p1 - p2))

    # Return both scores for debugging
    return scoreA, scoreB


# Streamlit UI setup
st.title("Pose Match Game")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Align your pose with the reference image on the left.
2. You have 30 seconds to match the pose.
3. After 30 seconds, the similarity score will be displayed.
""")

# Layout for side-by-side images
col1, col2 = st.columns(2)

with col1:
    st.header("Reference Pose")
    st.image(reference_img_rgb, channels="RGB", use_container_width=True)

ref_lm = get_normalized_landmark(reference_img)

# Placeholder for the webcam feed
video_feed_placeholder = col2.empty()

# Add the start button below the columns
start_game = st.button("Start Game")

# Timer Display (using placeholder to dynamically update)
timer_placeholder = st.empty()
similarity_placeholder = st.empty()

# Timer configuration
time_limit = 30  # seconds

# Initialize game state variables
start_time = None
game_started = False

# Function to start the game and update the timer
if start_game:
    game_started = True
    start_time = time.time()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while game_started:
        # Read from webcam
        success, frame = cap.read()
        if not success:
            st.write("Error accessing webcam!")
            break

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Process the frame for pose detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(img_rgb)
        results_seg = segmenter.process(img_rgb)

        # Create a blank image (black background)
        h, w, c = frame.shape
        blank_image = np.zeros((h, w, 3), dtype=np.uint8)

        # If landmarks are found, draw the pose on the blank image
        if results_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(blank_image, results_pose.pose_landmarks,
                                                      mp_pose.POSE_CONNECTIONS)

        # If the person is segmented (foreground mask)
        if results_seg.segmentation_mask is not None:
            # Get the segmentation mask
            mask = results_seg.segmentation_mask
            # Create a binary mask where the person is
            person_mask = np.stack((mask, mask, mask), axis=-1)  # Convert to 3 channels

            # Inverse mask (background)
            inverse_mask = np.ones_like(person_mask) - person_mask

            # Use the mask to remove the person from the original image (black background where person is)
            result_img = cv2.bitwise_and(frame, inverse_mask.astype(np.uint8))

            # Combine the pose skeleton on top of the black background
            result_img = cv2.add(result_img, blank_image)

            # Display the image with pose skeleton only
            cv2.imshow("Skeleton Only", result_img)

        else:
            # If no segmentation mask, show the blank image (no segmentation)
            cv2.imshow("Skeleton Only", blank_image)

        # Show the webcam feed in the UI
        video_feed_placeholder.image(result_img, channels="BGR", use_container_width=True)

        # Timer Logic: Update the timer every second
        elapsed_time = int(time.time() - start_time)
        remaining_time = max(0, time_limit - elapsed_time)
        timer_placeholder.write(f"Time Left: {remaining_time}s")

        # Calculate similarity score
        user_lm = get_normalized_landmark(img_rgb)
        if user_lm is not None:
            scoreA, scoreB = similarity_score(ref_lm, user_lm)
            similarity_placeholder.write(f"Cosine Similarity Score: {scoreA:.2f}")
            similarity_placeholder.write(f"Weighted Distance Score: {scoreB:.2f}")
        else:
            similarity_placeholder.write("Waiting for pose detection...")

        # Check if time is up
        if remaining_time <= 0:
            timer_placeholder.write("Time's up!")
            game_started = False

    cap.release()
