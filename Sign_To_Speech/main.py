import cv2
import numpy as np
import onnxruntime as ort
import json
import time

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- Config ----------
FIXED_LEN = 40
CONFIDENCE_THRESHOLD = 0.6
MOVEMENT_THRESHOLD = 0.015
IDLE_FRAMES_TO_STOP = 15
MAX_RECORD_SECONDS = 5
WAKE_HOLD_DURATION = 1.0

HAND_LANDMARKS = 21
POSE_LANDMARKS = 33

# ---------- Load model + labels ----------
with open("idx_to_sign.json") as f:
    idx_to_sign = json.load(f)
idx_to_sign = {int(k): v for k, v in idx_to_sign.items()}

session = ort.InferenceSession("sign_model.onnx")
input_name = session.get_inputs()[0].name

# ---------- MediaPipe Tasks setup ----------
hand_base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

pose_base_options = mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

def extract_frame_landmarks(frame_rgb, timestamp_ms):
    """Returns a (75, 2) array for one frame: 21 left + 21 right + 33 pose."""
    output = np.zeros((HAND_LANDMARKS * 2 + POSE_LANDMARKS, 2), dtype=np.float32)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    if hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
            if label == "Left":
                output[0:21] = coords
            else:
                output[21:42] = coords

    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    if pose_result.pose_landmarks:
        coords = np.array([[lm.x, lm.y] for lm in pose_result.pose_landmarks[0]], dtype=np.float32)
        if coords.shape[0] == POSE_LANDMARKS:
            output[42:75] = coords

    return output

def pad_or_truncate(seq, fixed_len=FIXED_LEN):
    n_frames = seq.shape[0]
    if n_frames == fixed_len:
        return seq
    elif n_frames > fixed_len:
        indices = np.linspace(0, n_frames - 1, fixed_len).astype(int)
        return seq[indices]
    else:
        pad_width = fixed_len - n_frames
        padding = np.zeros((pad_width, seq.shape[1], seq.shape[2]), dtype=np.float32)
        return np.concatenate([seq, padding], axis=0)

def compute_movement(prev, curr):
    if prev is None or curr is None:
        return 0.0
    return np.linalg.norm(curr - prev, axis=1).mean()

def check_wake_gesture(landmarks_75x2):
    left = landmarks_75x2[0:21]
    right = landmarks_75x2[21:42]
    return (left.sum() != 0) or (right.sum() != 0)

def predict_sign(clip_frames_75x2):
    padded = pad_or_truncate(np.array(clip_frames_75x2))
    flat = padded.reshape(1, FIXED_LEN, -1).astype(np.float32)
    logits = session.run(None, {input_name: flat})[0]
    probs = np.exp(logits) / np.exp(logits).sum()
    pred_idx = int(probs.argmax())
    confidence = float(probs.max())
    return idx_to_sign[pred_idx], confidence

# ---------- Main loop ----------
def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    sentence = []

    state = "IDLE"
    wake_start = None
    frame_buffer = []
    prev_landmarks = None
    idle_counter = 0
    record_start_time = None

    start_time = time.time()

    print("Starting... show your hand to wake, sign, then hold still to confirm.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.time() - start_time) * 1000)

        landmarks = extract_frame_landmarks(frame_rgb, timestamp_ms)

        display_text = f"State: {state}"

        if state == "IDLE":
            if check_wake_gesture(landmarks):
                if wake_start is None:
                    wake_start = time.time()
                elif time.time() - wake_start >= WAKE_HOLD_DURATION:
                    state = "RECORDING"
                    frame_buffer = []
                    idle_counter = 0
                    prev_landmarks = None
                    record_start_time = time.time()
                    print("Woke up — recording")
            else:
                wake_start = None

        elif state == "RECORDING":
            frame_buffer.append(landmarks)
            movement = compute_movement(prev_landmarks, landmarks)

            if landmarks.sum() == 0 or movement < MOVEMENT_THRESHOLD:
                idle_counter += 1
            else:
                idle_counter = 0

            timed_out = (time.time() - record_start_time) > MAX_RECORD_SECONDS

            if idle_counter >= IDLE_FRAMES_TO_STOP or timed_out:
                if len(frame_buffer) >= 5:
                    sign, conf = predict_sign(frame_buffer)
                    print(f"Predicted: {sign} (confidence {conf:.2f})")
                    if conf >= CONFIDENCE_THRESHOLD:
                        sentence.append(sign)
                        print("Sentence so far:", " ".join(sentence))
                    else:
                        print("Low confidence, ignored")
                state = "IDLE"
                wake_start = None

            prev_landmarks = landmarks

        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Sentence: " + " ".join(sentence), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("ASL Real-Time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            print("Sentence cleared")

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()

if __name__ == "__main__":
    main()
