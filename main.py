"""
Continuous live ASL sign detection -> sentence output.

Unlike infer_live_mediapipe_tasks.py (which needs a manual start/stop per
sign), this version runs continuously: it predicts on a sliding window
every frame, but only "confirms" a sign once the same prediction has been
stable for several frames in a row. Confirmed signs accumulate into a
rolling word buffer, which is checked against SENTENCE_TEMPLATES after
every new confirmation. No stopping needed between signs.

Requires:
  pip install mediapipe onnxruntime opencv-python numpy pyttsx3

Needs config.json (same folder) + sentence_templates.py (same folder) +
the model/task files config.json points to.
"""

import json
import time
import os
import sys
import collections
import numpy as np
import cv2
import onnxruntime as ort
import mediapipe as mp
import pyttsx3
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from sentence_templates import match_sentence


def get_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    base_dir = get_base_dir()
    config_path = os.path.join(base_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    for key in ["pose_task_path", "hand_task_path", "onnx_model_path", "label_map_path"]:
        cfg[key] = os.path.join(base_dir, cfg[key])
        if not os.path.exists(cfg[key]):
            print(f"ERROR: required file missing -- {key} = {cfg[key]}")
            sys.exit(1)
    return cfg


CFG = load_config()
POSE_TASK_PATH = CFG["pose_task_path"]
HAND_TASK_PATH = CFG["hand_task_path"]
ONNX_MODEL_PATH = CFG["onnx_model_path"]
LABEL_MAP_PATH = CFG["label_map_path"]

FIXED_LEN = CFG["fixed_len"]
HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
CONFIDENCE_THRESHOLD = CFG["confidence_threshold"]
TTS_RATE = CFG["tts_rate"]
CAMERA_INDEX = CFG["camera_index"]

# How many consecutive frames the SAME prediction must hold before it
# counts as "confirmed" -- higher = fewer false positives, but slower
# to react. Tune this if signs get missed or double-counted.
STABILITY_FRAMES = CFG.get("stability_frames", 8)

# After confirming a sign, ignore repeats of that SAME sign for this many
# frames -- prevents one held sign from being confirmed over and over
# while the signer is still mid-transition to the next sign.
SAME_SIGN_COOLDOWN_FRAMES = CFG.get("same_sign_cooldown_frames", 25)

# How many recent confirmed signs to keep in the buffer for sentence matching
WORD_BUFFER_SIZE = CFG.get("word_buffer_size", 4)

# If no new sign is confirmed for this many seconds, the word buffer is
# automatically cleared -- prevents a stale half-finished word (e.g. you
# signed "look" but never followed up with "bird") from silently sticking
# around and accidentally completing a sentence minutes later.
WORD_BUFFER_TIMEOUT_SECONDS = CFG.get("word_buffer_timeout_seconds", 6.0)

# --- Confusable-pair margin check ---
# Some signs look visually similar to the model (e.g. "thankyou" vs "talk"
# both start near the mouth/chin) and can flip a coin between them even at
# high confidence. This adds a second check: for any pair of signs listed
# here, the top prediction must beat the runner-up by CONFUSABLE_MARGIN
# (in probability, 0-1) or the frame is treated as "unclear" and ignored
# rather than wrongly confirmed. Add pairs as: ["signA", "signB"].
# Set in config.json as "confusable_pairs" and "confusable_margin".
CONFUSABLE_PAIRS = set()
for pair in CFG.get("confusable_pairs", [["thankyou", "talk"]]):
    CONFUSABLE_PAIRS.add(frozenset(pair))
CONFUSABLE_MARGIN = CFG.get("confusable_margin", 0.15)

# --- Label overrides ---
# Remap a predicted sign to a different output word before it's shown,
# spoken, or added to the sentence buffer. Useful when the model
# consistently mixes up two signs and you'd rather just treat one
# prediction as the other, e.g. model predicts "talk" but you always
# mean "thankyou" -- set {"talk": "thankyou"} and every "talk" prediction
# becomes "thankyou" everywhere downstream (display, buffer, speech,
# sentence matching). The original sign the model is NOT trained
# differently -- this only relabels the output. Set in config.json as
# "label_overrides". Leave empty {} to disable.
LABEL_OVERRIDES = CFG.get("label_overrides", {"talk": "thankyou"})

# --- Speech mode ---
# "words_and_sentences" (default): speak every confirmed sign, AND speak
#     the full sentence whenever the word buffer matches a template.
# "sentence_only": stay silent on individual signs; only speak when the
#     word buffer matches a SENTENCE_TEMPLATES entry.
# "words_only": only ever speak individual confirmed signs; never try to
#     match or speak full sentences.
SPEECH_MODE = CFG.get("speech_mode", "words_and_sentences")
if SPEECH_MODE not in ("words_and_sentences", "sentence_only", "words_only"):
    print(f"WARNING: unknown speech_mode '{SPEECH_MODE}', falling back to 'words_and_sentences'")
    SPEECH_MODE = "words_and_sentences"


def normalize_landmarks(seq):
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
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, velocity], axis=-1)


class LandmarkExtractor:
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
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33

        output = np.zeros((HAND_LANDMARKS * 2 + POSE_LANDMARKS, 2), dtype=np.float32)

        pose_result = self.pose_landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if pose_result.pose_landmarks:
            pts = pose_result.pose_landmarks[0]
            for i, lm in enumerate(pts[:POSE_LANDMARKS]):
                output[42 + i] = [lm.x, lm.y]

        hand_result = self.hand_landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name
                offset = 0 if label == "Left" else 21
                for i, lm in enumerate(hand_landmarks[:HAND_LANDMARKS]):
                    output[offset + i] = [lm.x, lm.y]

        return output

    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()


def predict_window(sess, input_name, idx_to_sign, frame_window):
    seq = np.stack(frame_window, axis=0)               # (FIXED_LEN, 75, 2)
    seq = normalize_landmarks(seq)
    seq = add_motion_features(seq)                        # (FIXED_LEN, 75, 4)
    seq = seq.reshape(1, FIXED_LEN, -1).astype(np.float32)

    logits = sess.run(None, {input_name: seq})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    top2_idx = np.argsort(probs)[-2:][::-1]  # [best_idx, second_best_idx]
    top_idx, second_idx = int(top2_idx[0]), int(top2_idx[1])
    return (
        idx_to_sign[top_idx], float(probs[top_idx]),
        idx_to_sign[second_idx], float(probs[second_idx]),
    )


def main():
    with open(LABEL_MAP_PATH) as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}

    sess = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    print(f"Loaded ONNX model with {len(sign_to_idx)} classes.")
    print(f"Speech mode: {SPEECH_MODE}")

    tts = pyttsx3.init()
    tts.setProperty("rate", TTS_RATE)

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    frame_window = collections.deque(maxlen=FIXED_LEN)

    # Debouncing state
    last_prediction = None
    stable_count = 0
    cooldown_sign = None
    cooldown_remaining = 0

    word_buffer = collections.deque(maxlen=WORD_BUFFER_SIZE)
    last_sentence = ""
    sentence_display_until = 0
    last_confirmed_time = time.time()

    print("Running continuously -- sign naturally, no need to stop between signs. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            landmarks = extractor.extract(frame)
            frame_window.append(landmarks)

            display_text = ""

            if len(frame_window) == FIXED_LEN:
                sign, confidence, second_sign, second_confidence = predict_window(
                    sess, input_name, idx_to_sign, frame_window
                )

                # If this sign is a known confusable pair with the runner-up,
                # require a bigger confidence gap before trusting it.
                is_confusable_pair = frozenset((sign, second_sign)) in CONFUSABLE_PAIRS
                margin = confidence - second_confidence
                margin_ok = (not is_confusable_pair) or (margin >= CONFUSABLE_MARGIN)

                # Relabel the prediction (e.g. "talk" -> "thankyou") AFTER the
                # confusable-pair check above, so that check still compares
                # the model's real top-2 predictions. Everything from here
                # down (stability, buffer, display, speech) uses the
                # overridden label.
                sign = LABEL_OVERRIDES.get(sign, sign)

                if confidence > CONFIDENCE_THRESHOLD and margin_ok:
                    # --- Stability check: same prediction N frames running ---
                    if sign == last_prediction:
                        stable_count += 1
                    else:
                        stable_count = 1
                        last_prediction = sign

                    # --- Cooldown countdown (prevents re-confirming a held sign) ---
                    if cooldown_remaining > 0:
                        cooldown_remaining -= 1
                        if sign != cooldown_sign:
                            cooldown_remaining = 0  # signer moved on, cooldown irrelevant now

                    is_confirmable = (
                        stable_count >= STABILITY_FRAMES
                        and not (sign == cooldown_sign and cooldown_remaining > 0)
                    )

                    if is_confirmable:
                        # --- New sign confirmed ---
                        word_buffer.append(sign)
                        cooldown_sign = sign
                        cooldown_remaining = SAME_SIGN_COOLDOWN_FRAMES
                        stable_count = 0  # require fresh stability for the next sign
                        last_confirmed_time = time.time()

                        sentence = None
                        if SPEECH_MODE in ("words_and_sentences", "sentence_only"):
                            sentence = match_sentence(list(word_buffer))

                        if sentence:
                            last_sentence = sentence
                            sentence_display_until = time.time() + 3.0
                            tts.say(sentence)
                            tts.runAndWait()
                            word_buffer.clear()  # sentence completed, start fresh
                        elif SPEECH_MODE in ("words_and_sentences", "words_only"):
                            tts.say(sign.replace("_", " "))
                            tts.runAndWait()
                        # else: sentence_only mode with no match -- stay silent,
                        # keep accumulating in word_buffer for a future match

                    display_text = f"{sign} ({confidence:.0%}) [{stable_count}/{STABILITY_FRAMES}]"
                    if is_confusable_pair and not margin_ok:
                        display_text += f"  UNCLEAR vs {second_sign} (margin {margin:.0%})"

            # --- Auto-clear word buffer after inactivity ---
            if word_buffer and (time.time() - last_confirmed_time) > WORD_BUFFER_TIMEOUT_SECONDS:
                word_buffer.clear()

            # --- On-screen overlay ---
            cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Words: " + " ".join(word_buffer), (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if time.time() < sentence_display_until:
                cv2.putText(frame, last_sentence, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("ASL Continuous Sentence Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()