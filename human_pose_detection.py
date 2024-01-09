import cv2
import mediapipe as mp

# Step 2: Initialize the MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Step 4: Capture video stream
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while cap.isOpened():
    # Step 5: Process each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 6: Process the frame with the Holistic model
    results = holistic.process(rgb_frame)

    # Step 7: Accessing pose landmarks
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        # Do something with the pose landmarks if needed

    # Step 8: Render the results
    # mp_holistic.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Step 9: Display the output
    cv2.imshow('MediaPipe Holistic', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Step 9 (cont.): Release resources
cap.release()
cv2.destroyAllWindows()
