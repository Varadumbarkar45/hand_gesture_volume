import cv2
import mediapipe as mp
import numpy as np
import joblib
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Load the trained model
MODEL_PATH = "gesture_model.pkl"
clf = joblib.load(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize system audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Start webcam
cap = cv2.VideoCapture(0)

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

            # Extract landmark positions
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            # Convert to numpy array and reshape
            landmarks = np.array(landmarks[:63]).reshape(1, -1)  # Use only first 63 features

            # Predict gesture
            gesture = clf.predict(landmarks)[0]

            # Adjust volume based on gesture
            if gesture == 0:  # Fist (Mute)
                volume.SetMasterVolumeLevelScalar(0.0, None)  # Mute
                cv2.putText(frame, "Muted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif gesture == 1:  # Open Palm (Full Volume)
                volume.SetMasterVolumeLevelScalar(1.0, None)  # Max volume
                cv2.putText(frame, "Max Volume", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif gesture == 2:  # Two Fingers (Medium Volume)
                volume.SetMasterVolumeLevelScalar(0.5, None)  # 50% volume
                cv2.putText(frame, "Medium Volume", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            elif gesture == 3:  # Thumbs Up (Increase Volume)
                current_vol = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(min(current_vol + 0.1, 1.0), None)  # Increase volume by 10%
                cv2.putText(frame, "Volume Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            elif gesture == 4:  # Thumbs Down (Decrease Volume)
                current_vol = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(max(current_vol - 0.1, 0.0), None)  # Decrease volume by 10%
                cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
