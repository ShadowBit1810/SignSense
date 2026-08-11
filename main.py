"""
Live webcam ASL sign inference -- capture-window mode with speech output.

Requires:
  pip install mediapipe onnxruntime opencv-python numpy pyttsx3

Files needed in the same folder (or update paths below):
  - pose_landmarker_lite.task   (MediaPipe Tasks pose model, already downloaded)
  - hand_landmarker.task        (MediaPipe Tasks hand model, already downloaded)
  - best_model_<fingerprint>_motion.onnx   (your trained model)
  - label_map_<fingerprint>_motion.json    (index -> sign name)

CRITICAL: preprocessing here MUST exactly match training (normalize_landmarks +
add_motion_features). Any difference between train-time and inference-time
preprocessing silently produces wrong predictions -- this script mirrors the
notebook's Cell 5 functions exactly.

HOW THIS VERSION WORKS (capture-window, not continuous):
  1. Idle, watching for motion (or press SPACE to start manually).
  2. Once triggered, collects landmarks for CAPTURE_SECONDS.
  3. Runs ONE prediction on that window, speaks it aloud via TTS.
  4. Shows a brief cooldown, then returns to idle, ready for the next sign.
"""

import json
import time
import collections
import numpy as np
import cv2
import onnxruntime as ort
import mediapipe as mp
import pyttsx3
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------- CONFIG -- update these paths/filenames ----------------
POSE_TASK_PATH = "pose_landmarker_lite.task"
HAND_TASK_PATH = "hand_landmarker.task"
ONNX_MODEL_PATH = "best_model_2f40e2e8c9_motion.onnx"   # <- your exported model
LABEL_MAP_PATH = "label_map_2f40e2e8c9_motion.json"     # <- matching label map

FIXED_LEN = 40
HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
CONFIDENCE_THRESHOLD = 0.6

CAPTURE_SECONDS = 5.0    # how long to record one sign attempt
COOLDOWN_SECONDS = 1.5    # pause after speaking, before ready for the next sign
TRIGGER_MODE = "manual"   # "manual" = press SPACE to start capture, "motion" = auto-trigger on hand detected
# --------------------------------------------------------------------


def normalize_landmarks(seq):
    """seq: (frames, 75, 2). Must exactly match training's normalize_landmarks."""
    left_shoulder = seq[:, 42 + 11, :]
    right_shoulder = seq[:, 42 + 12, :]

    shoulders_detected = ~((left_shoulder == 0).all(axis=1) | (right_shoulder == 0).all(axis=1))

    center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder, axis=1, keepdims=True)
    shoulder_width = np.where(shoulder_width < 0.05, 0.3, shoulder_width)

    seq = seq - center[:, None, :]
    seq = seq / shoulder_width[:, None, :]
    seq[~shoulders_detected] = 0

    return seq


def add_motion_features(seq):
    """seq: (frames, 75, 2) normalized. Returns (frames, 75, 4) = [x,y,dx,dy].
    Must exactly match training's add_motion_features."""
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, velocity], axis=-1)


class LandmarkExtractor:
    """Wraps MediaPipe Tasks pose + hand landmarkers, outputs the same
    (75, 2) [left_hand(21), right_hand(21), pose(33)] layout as training."""

    def __init__(self):
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

        self._timestamp_ms = 0

    def extract(self, frame_bgr):
        """frame_bgr: raw OpenCV BGR frame. Returns (75, 2) float32 array,
        zeros where a landmark set wasn't detected."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._timestamp_ms += 33  # approx 30fps step; must be strictly increasing

        output = np.zeros((HAND_LANDMARKS * 2 + POSE_LANDMARKS, 2), dtype=np.float32)

        # --- Pose ---
        pose_result = self.pose_landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if pose_result.pose_landmarks:
            pts = pose_result.pose_landmarks[0]  # first detected person
            for i, lm in enumerate(pts[:POSE_LANDMARKS]):
                output[42 + i] = [lm.x, lm.y]

        # --- Hands (with handedness to place into correct left/right slot) ---
        hand_result = self.hand_landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                offset = 0 if label == "Left" else 21
                for i, lm in enumerate(hand_landmarks[:HAND_LANDMARKS]):
                    output[offset + i] = [lm.x, lm.y]

        return output

    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()


def predict_sign(sess, input_name, idx_to_sign, frames):
    """frames: list of (75,2) raw landmark arrays, any length -- resampled to FIXED_LEN."""
    seq = np.stack(frames, axis=0)                       # (n, 75, 2)
    idx = np.linspace(0, len(seq) - 1, FIXED_LEN).astype(int)
    seq = seq[idx]                                        # (40, 75, 2)

    seq = normalize_landmarks(seq)
    seq = add_motion_features(seq)                         # (40, 75, 4)
    seq = seq.reshape(1, FIXED_LEN, -1).astype(np.float32)  # (1, 40, 300)

    logits = sess.run(None, {input_name: seq})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    top_idx = int(np.argmax(probs))
    return idx_to_sign[top_idx], float(probs[top_idx])


def main():
    with open(LABEL_MAP_PATH) as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}

    sess = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    print(f"Loaded ONNX model with {len(sign_to_idx)} classes. Input: {sess.get_inputs()[0].shape}")

    tts = pyttsx3.init()
    tts.setProperty("rate", 165)

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)

    state = "idle"          # idle -> capturing -> cooldown -> idle
    captured_frames = []
    state_start_time = 0.0
    last_result_text = ""

    print("Ready. Press SPACE to record a sign, 'q' to quit."
          if TRIGGER_MODE == "manual" else
          "Ready. Show a hand to auto-start recording, 'q' to quit.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            landmarks = extractor.extract(frame)  # (75, 2), raw this frame
            hand_visible = not (landmarks[0:42] == 0).all()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            now = time.time()

            if state == "idle":
                cv2.putText(frame, "Ready (SPACE to sign)" if TRIGGER_MODE == "manual" else "Ready...",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                if last_result_text:
                    cv2.putText(frame, f"Last: {last_result_text}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                should_start = (key == ord(" ")) if TRIGGER_MODE == "manual" else hand_visible
                if should_start:
                    state = "capturing"
                    captured_frames = []
                    state_start_time = now

            elif state == "capturing":
                captured_frames.append(landmarks)
                elapsed = now - state_start_time
                cv2.putText(frame, f"Recording... {elapsed:.1f}s", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if elapsed >= CAPTURE_SECONDS:
                    sign, confidence = predict_sign(sess, input_name, idx_to_sign, captured_frames)
                    if confidence > CONFIDENCE_THRESHOLD:
                        last_result_text = f"{sign} ({confidence:.0%})"
                        tts.say(sign.replace("_", " "))
                        tts.runAndWait()
                    else:
                        last_result_text = f"unsure ({confidence:.0%})"
                    state = "cooldown"
                    state_start_time = now

            elif state == "cooldown":
                cv2.putText(frame, f"Result: {last_result_text}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if now - state_start_time >= COOLDOWN_SECONDS:
                    state = "idle"

            cv2.imshow("ASL Recognition (live)", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()