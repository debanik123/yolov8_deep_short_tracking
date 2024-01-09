import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hand Tracking
        results = self.hands.process(frame_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw a rectangle around the detected hand
                self.draw_hand_rectangle(frame, landmarks)

        return frame

    def draw_hand_rectangle(self, frame, landmarks):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Draw the rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Example usage:

# Create HandTracker instance
hand_tracker = HandTracker()

# Open a video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with hand tracking
    processed_frame = hand_tracker.process_frame(frame)

    # Display the processed frame
    cv2.imshow("Hand Tracking", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
