"""
cv_main.py -- OpenCV-window front-end for the ASL sign detector.

No tkinter, no extra dependencies beyond what main.py already needs.
Camera preview runs continuously, but the model only predicts while
prediction is switched ON -- avoids random hand movement between signs
being misread as a sign.

Controls (with the OpenCV window focused):
  SPACE : toggle prediction ON / OFF
  c     : clear the word buffer
  q     : quit

Reuses all model/config logic from main.py (same folder) so there is
only one place that defines how signs are detected.
"""

import collections
import time

import cv2
import numpy as np

import main as core  # reuses CFG, LandmarkExtractor, predict_window, etc.


# Colors (BGR, since this is OpenCV)
GREEN = (100, 230, 120)
RED = (90, 90, 250)
YELLOW = (70, 230, 230)
WHITE = (245, 245, 245)
GRAY = (150, 150, 150)
PANEL_BG = (25, 22, 20)  # near-black panel background, distinct from any video content

PANEL_HEIGHT = 170  # pixels, fixed height text panel below the camera feed


def build_panel(width, predicting, display_text, word_buffer, last_sentence,
                 sentence_display_until):
    """Builds a fixed-height, solid-background panel with all status text.
    Kept separate from the video frame so text contrast never depends on
    whatever is happening in the camera feed behind it."""
    panel = np.full((PANEL_HEIGHT, width, 3), PANEL_BG, dtype=np.uint8)

    # Thin separator line at the very top of the panel
    cv2.line(panel, (0, 0), (width, 0), (60, 60, 60), 2)

    # --- Row 1: status + current sign prediction ---
    status_text = "PREDICTING" if predicting else "PAUSED -- press SPACE to start"
    status_color = GREEN if predicting else RED
    cv2.putText(panel, status_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if display_text:
        cv2.putText(panel, display_text, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    # --- Row 2: word buffer ---
    words_text = "Words: " + (" ".join(word_buffer) if word_buffer else "(empty)")
    cv2.putText(panel, words_text, (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2)

    # --- Row 3 (dedicated, larger): last matched sentence ---
    # This is the main "sentence output" area -- always reserved, so it
    # never overlaps or gets lost against other text.
    cv2.rectangle(panel, (10, 108), (width - 10, PANEL_HEIGHT - 8), (15, 12, 10), -1)
    cv2.rectangle(panel, (10, 108), (width - 10, PANEL_HEIGHT - 8), (60, 60, 60), 1)

    if time.time() < sentence_display_until and last_sentence:
        cv2.putText(panel, last_sentence, (20, PANEL_HEIGHT - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, GREEN, 2)
    else:
        cv2.putText(panel, "(sentence appears here)", (20, PANEL_HEIGHT - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1)

    return panel


def main():
    import json
    with open(core.LABEL_MAP_PATH) as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}

    import onnxruntime as ort
    sess = ort.InferenceSession(core.ONNX_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    print(f"Loaded ONNX model with {len(sign_to_idx)} classes.")
    print(f"Speech mode: {core.SPEECH_MODE}")

    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty("rate", core.TTS_RATE)

    extractor = core.LandmarkExtractor()
    cap = cv2.VideoCapture(core.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {core.CAMERA_INDEX}.")
        return

    frame_window = collections.deque(maxlen=core.FIXED_LEN)

    last_prediction = None
    stable_count = 0
    cooldown_sign = None
    cooldown_remaining = 0

    word_buffer = collections.deque(maxlen=core.WORD_BUFFER_SIZE)
    last_sentence = ""
    sentence_display_until = 0
    last_confirmed_time = time.time()

    predicting = False  # starts OFF -- press SPACE to begin

    cv2.namedWindow("ASL Sign Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Sign Detector", 900, 700)

    print("Camera live. SPACE = toggle prediction on/off, c = clear buffer, q = quit.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            display_text = ""

            if predicting:
                landmarks = extractor.extract(frame)
                frame_window.append(landmarks)

                if len(frame_window) == core.FIXED_LEN:
                    sign, confidence, second_sign, second_confidence = core.predict_window(
                        sess, input_name, idx_to_sign, frame_window
                    )

                    is_confusable_pair = frozenset((sign, second_sign)) in core.CONFUSABLE_PAIRS
                    margin = confidence - second_confidence
                    margin_ok = (not is_confusable_pair) or (margin >= core.CONFUSABLE_MARGIN)

                    sign = core.LABEL_OVERRIDES.get(sign, sign)

                    if confidence > core.CONFIDENCE_THRESHOLD and margin_ok:
                        # --- Stability check ---
                        if sign == last_prediction:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_prediction = sign

                        # --- Cooldown countdown ---
                        if cooldown_remaining > 0:
                            cooldown_remaining -= 1
                            if sign != cooldown_sign:
                                cooldown_remaining = 0

                        is_confirmable = (
                            stable_count >= core.STABILITY_FRAMES
                            and not (sign == cooldown_sign and cooldown_remaining > 0)
                        )

                        if is_confirmable:
                            word_buffer.append(sign)
                            cooldown_sign = sign
                            cooldown_remaining = core.SAME_SIGN_COOLDOWN_FRAMES
                            stable_count = 0
                            last_confirmed_time = time.time()

                            sentence = None
                            if core.SPEECH_MODE in ("words_and_sentences", "sentence_only"):
                                sentence = core.match_sentence(list(word_buffer))

                            if sentence:
                                last_sentence = sentence
                                sentence_display_until = time.time() + 3.0
                                tts.say(sentence)
                                tts.runAndWait()
                                word_buffer.clear()
                            elif core.SPEECH_MODE in ("words_and_sentences", "words_only"):
                                tts.say(sign.replace("_", " "))
                                tts.runAndWait()

                        display_text = f"{sign} ({confidence:.0%}) [{stable_count}/{core.STABILITY_FRAMES}]"
                        if is_confusable_pair and not margin_ok:
                            display_text += f"  unclear vs {second_sign} ({margin:.0%})"

            # --- Auto-clear word buffer after inactivity ---
            if word_buffer and (time.time() - last_confirmed_time) > core.WORD_BUFFER_TIMEOUT_SECONDS:
                word_buffer.clear()

            # --- Build the dedicated text panel and stack it below the video ---
            panel = build_panel(
                width=frame.shape[1],
                predicting=predicting,
                display_text=display_text,
                word_buffer=word_buffer,
                last_sentence=last_sentence,
                sentence_display_until=sentence_display_until,
            )
            combined = np.vstack([frame, panel])

            cv2.putText(combined, "SPACE: toggle  |  c: clear  |  q: quit",
                        (18, combined.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)

            cv2.imshow("ASL Sign Detector", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                predicting = not predicting
                if predicting:
                    # fresh start -- don't let stale frames bias the window
                    frame_window.clear()
                    last_prediction = None
                    stable_count = 0
                    print("-> Prediction ON")
                else:
                    print("-> Prediction OFF (paused)")
            elif key == ord("c"):
                word_buffer.clear()
                last_sentence = ""
                sentence_display_until = 0
                print("-> Word buffer cleared")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()