import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Capture video input for pose detection
cap = cv2.VideoCapture(r"videoFiles\video1.mp4.mp4")


# Function to calculate instability based on pose landmarks
def calculate_instability(landmarks):
    if landmarks is None:
        return 0
    # Example instability metric: variance of y-coordinates of key points
    y_coords = [landmark.y for landmark in landmarks.landmark]
    instability = np.var(y_coords)
    return instability


# Function to classify level of drunkenness based on instability
def classify_drunkenness(instability):
    if instability < 0.0005:
        return "Sober"
    elif instability < 0.005:
        return "Slightly Drunk"
    elif instability < 0.01:
        return "Moderately Drunk"
    else:
        return "Highly Drunk"


# Read each frame/image from capture object
while True:
    ret, img = cap.read()
    if not ret:
        break

    # Resize image/frame for display
    img = cv2.resize(img, (600, 600))

    # Perform pose detection
    results = pose.process(img)

    # Draw the detected pose on the original video
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2))

    # Calculate instability
    instability = calculate_instability(results.pose_landmarks)

    # Classify level of drunkenness
    drunkenness_level = classify_drunkenness(instability)

    # Display the drunkenness level on the video
    cv2.putText(img, f"Drunkenness Level: {drunkenness_level}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)

    # Print pose landmarks and drunkenness level
    print(f"Pose Landmarks: {results.pose_landmarks}")
    print(f"Drunkenness Level: {drunkenness_level}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()