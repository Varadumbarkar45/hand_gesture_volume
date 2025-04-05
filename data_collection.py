import cv2
import mediapipe as mp
import csv
import os

# Define the dataset file name
DATA_FILE = "gesture_data.csv"

# Check if file exists, otherwise create it with headers
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [f"x{i+1}" for i in range(21)] + [f"y{i+1}" for i in range(21)] + [f"z{i+1}" for i in range(21)] + ["label"]
        writer.writerow(headers)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Ask for the gesture label
gesture_label = input("Enter the label for this gesture (e.g., 0 for fist, 1 for palm, etc.): ")

print("\n[INFO] Show your gesture and press 'q' to stop collecting data...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract all 21 landmark positions
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            # Save to CSV
            with open(DATA_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks + [gesture_label])

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n[INFO] Data collection complete! Check 'gesture_data.csv'.")
