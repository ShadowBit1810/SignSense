/*
====================================================================
ASL Recognition - browser runtime
====================================================================

Execution flow:

1. Page loads
2. app.js checks the browser + HTTP server
3. Camera permission can be tested independently
4. config.json is loaded
5. label_map JSON is loaded
6. sentence_data JSON is loaded
7. MediaPipe Tasks Vision is imported
8. Pose + Hand .task files are loaded
9. ONNX Runtime Web is imported
10. ONNX model is loaded
11. Start Recognition:
      Camera frame
        -> MediaPipe pose + hands
        -> 75 landmarks x/y
        -> shoulder normalization
        -> motion / velocity features
        -> 40-frame window
        -> ONNX
        -> softmax
        -> top-2 result
        -> confidence / confusable check
        -> stability
        -> same-sign cooldown
        -> word buffer
        -> sentence template
        -> Web Speech API

This mirrors the supplied Python workflow.

Important browser replacements:
- cv2.VideoCapture      -> navigator.mediaDevices.getUserMedia()
- cv2 frame             -> <video> element
- MediaPipe Python      -> MediaPipe Tasks Vision JS
- onnxruntime Python    -> onnxruntime-web
- pyttsx3               -> Web Speech API
====================================================================
*/

const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const cameraPlaceholder = $("cameraPlaceholder");
const loadingOverlay = $("loadingOverlay");
const loadingStep = $("loadingStep");

const status = $("status");
const statusText = $("statusText");
const signCount = $("signCount");

const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const cameraBtn = $("cameraBtn");
const clearBtn = $("clearBtn");

const signLine = $("signLine");
const wordsLine = $("wordsLine");
const sentenceBox = $("sentenceBox");
const loadLog = $("loadLog");

const errorBanner = $("errorBanner");
const errorTitle = $("errorTitle");
const errorDetail = $("errorDetail");
const errorRetryBtn = $("errorRetryBtn");
const errorDismissBtn = $("errorDismissBtn");


// ==================================================================
// Configuration
// ==================================================================

const DEFAULT_CONFIG = {
  pose_task_path: "mediapipe_tasks/pose_landmarker_lite.task",
  hand_task_path: "mediapipe_tasks/hand_landmarker.task",
  onnx_model_path: "models/best_model_2f40e2e8c9_motion.onnx",
  label_map_path: "models/label_map_2f40e2e8c9_motion.json",

  fixed_len: 40,
  confidence_threshold: 0.55,

  speech_mode: "words_and_sentences",
  tts_rate: 165,
  camera_index: 0,

  stability_frames: 3,
  same_sign_cooldown_frames: 15,
  word_buffer_size: 4,
  word_buffer_timeout_seconds: 6.0,

  confusable_pairs: [["thankyou", "talk"]],
  confusable_margin: 0.15,

  process_interval_ms: 40,
  prefer_webgpu_onnx: true,
  prefer_gpu_mediapipe: false,

  label_overrides: {
    talk: "thankyou"
  }
};

let CFG = { ...DEFAULT_CONFIG };


// ==================================================================
// Runtime state
// ==================================================================

let ort = null;
let FilesetResolver = null;
let HandLandmarker = null;
let PoseLandmarker = null;

let poseLandmarker = null;
let handLandmarker = null;
let onnxSession = null;

let modelInputName = null;
let modelOutputName = null;

let signToIndex = {};
let indexToSign = {};
let sentenceTemplates = [];

let cameraStream = null;

let cameraOn = false;
let recognizing = false;

let initializationStarted = false;
let dependenciesReady = false;
let mediaPipeReady = false;
let onnxReady = false;
let labelsReady = false;

let processing = false;
let animationFrame = 0;
let lastProcessTime = 0;
let mediaTimestamp = 0;

const frameWindow = [];

let lastPrediction = null;
let stableCount = 0;

let cooldownSign = null;
let cooldownRemaining = 0;

let wordBuffer = [];
let lastConfirmedTime = 0;

let lastSentence = "";
let sentenceDisplayUntil = 0;

let confirmedSigns = 0;

let currentDisplay = "";


// ==================================================================
// Logging / UI
// ==================================================================

function log(message, type = "info") {
  const time = new Date().toLocaleTimeString();
  const prefix =
    type === "error" ? "[ERROR]" :
    type === "warn" ? "[WARN]" :
    "[INFO]";

  loadLog.textContent += `\n${time} ${prefix} ${message}`;
  loadLog.scrollTop = loadLog.scrollHeight;

  if (type === "error") {
    console.error(message);
  } else if (type === "warn") {
    console.warn(message);
  } else {
    console.log(message);
  }
}

function setLoading(message) {
  if (loadingStep) {
    loadingStep.textContent = message;
  }
  loadingOverlay?.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay?.classList.add("hidden");
}

function setStatus(message, state = "off") {
  if (!status) return;

  statusText.textContent = message;
  status.className = "status-pill";

  if (state === "on") {
    status.classList.add("on");
  }

  if (state === "error") {
    status.classList.add("error");
  }
}

function showError(title, detail) {
  errorTitle.textContent = title;
  errorDetail.textContent = detail;
  errorBanner.classList.add("show");
  setStatus("ERROR — see log", "error");
}

function hideError() {
  errorBanner.classList.remove("show");
}

function formatSign(text) {
  return String(text ?? "").replaceAll("_", " ");
}

function updateUI(text = currentDisplay) {
  currentDisplay = text || "";

  if (currentDisplay) {
    signLine.textContent = currentDisplay;
    signLine.classList.remove("empty");
  } else if (!recognizing) {
    signLine.textContent = "Nothing detected yet.";
    signLine.classList.add("empty");
  }

  wordsLine.textContent =
    `Words: ${wordBuffer.length ? wordBuffer.join(" ") : "(empty)"}`;

  if (
    lastSentence &&
    performance.now() < sentenceDisplayUntil
  ) {
    sentenceBox.textContent = lastSentence;
    sentenceBox.classList.remove("empty");
  } else {
    sentenceBox.textContent = "(sentence appears here)";
    sentenceBox.classList.add("empty");
  }

  signCount.textContent =
    `${confirmedSigns} ${confirmedSigns === 1 ? "sign" : "signs"} detected`;
}

function setButtonState() {
  const canUseCamera =
    !!navigator.mediaDevices?.getUserMedia;

  cameraBtn.disabled = !canUseCamera;
  clearBtn.disabled = !dependenciesReady && !cameraOn;

  startBtn.disabled =
    !(dependenciesReady && cameraOn) || recognizing;

  stopBtn.disabled = !recognizing;
}


// ==================================================================
// HTTP / browser diagnostics
// ==================================================================

function checkEnvironment() {
  const protocolOK =
    location.protocol === "http:" ||
    location.protocol === "https:";

  if (!protocolOK) {
    showError(
      "Use the local HTTP server",
      "Open this page through http://localhost:8000 instead of file://."
    );

    log(
      `Bad protocol: ${location.protocol}. Use: python -m http.server 8000`,
      "error"
    );

    return false;
  }

  if (!window.isSecureContext) {
    log(
      "The page is not marked as a secure context. localhost normally is, but camera permissions may be blocked on unusual hosts.",
      "warn"
    );
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    log(
      "navigator.mediaDevices.getUserMedia is unavailable in this browser/context.",
      "error"
    );
    return false;
  }

  log(`Page URL: ${location.href}`);
  log(`Origin: ${location.origin}`);
  log(`User agent: ${navigator.userAgent}`);

  return true;
}


// ==================================================================
// Generic file loading
// ==================================================================

async function fetchJSON(path) {
  log(`Fetching ${path} ...`);

  const response = await fetch(path, {
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(
      `${path} returned HTTP ${response.status} ${response.statusText}`
    );
  }

  return await response.json();
}

async function checkAsset(path) {
  log(`Checking asset: ${path}`);

  const response = await fetch(path, {
    method: "HEAD",
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(
      `${path} is not reachable (HTTP ${response.status})`
    );
  }

  return true;
}


// ==================================================================
// Config / labels / sentences
// ==================================================================

function mergeConfig(raw) {
  return {
    ...DEFAULT_CONFIG,
    ...raw,
    label_overrides: {
      ...DEFAULT_CONFIG.label_overrides,
      ...(raw?.label_overrides || {})
    }
  };
}

async function loadConfiguration() {
  setLoading("Loading config.json…");

  try {
    const raw = await fetchJSON("./config.json");
    CFG = mergeConfig(raw);
    log("config.json loaded successfully.");
  } catch (error) {
    /*
     * Do not kill the whole page for missing config.
     * This lets you test camera access first.
     */
    CFG = { ...DEFAULT_CONFIG };
    log(
      `config.json could not be loaded. Using built-in defaults. ${error.message}`,
      "warn"
    );
  }

  log(`fixed_len = ${CFG.fixed_len}`);
  log(`confidence_threshold = ${CFG.confidence_threshold}`);
  log(`speech_mode = ${CFG.speech_mode}`);
  log(`tts_rate = ${CFG.tts_rate}`);
  log(`camera_index = ${CFG.camera_index}`);
  log(`process_interval_ms = ${CFG.process_interval_ms}`);

  if (
    !["words_and_sentences", "sentence_only", "words_only"]
      .includes(CFG.speech_mode)
  ) {
    log(
      `Unknown speech_mode '${CFG.speech_mode}', using words_and_sentences.`,
      "warn"
    );
    CFG.speech_mode = "words_and_sentences";
  }
}

async function loadLabels() {
  setLoading("Loading label map…");

  const raw = await fetchJSON(CFG.label_map_path);

  signToIndex = raw;
  indexToSign = {};

  for (const [label, index] of Object.entries(raw)) {
    indexToSign[Number(index)] = label;
  }

  labelsReady = true;

  log(
    `Label map loaded: ${Object.keys(signToIndex).length} classes.`
  );
}

async function loadSentences() {
  setLoading("Loading sentence templates…");

  /*
   * Browser version of the supplied sentence_templates.py.
   *
   * If sentence_data.json is absent, recognition still works for
   * individual signs; sentence matching simply stays disabled.
   */
  const candidates = [
    "./sentence_data.json",
    "./sentence_data(6).json"
  ];

  for (const path of candidates) {
    try {
      const raw = await fetchJSON(path);

      sentenceTemplates =
        Array.isArray(raw?.templates)
          ? raw.templates
          : [];

      log(
        `Loaded ${sentenceTemplates.length} sentence templates from ${path}.`
      );

      return;
    } catch {
      // Try the next filename.
    }
  }

  sentenceTemplates = [];

  log(
    "No sentence_data JSON found. Word recognition remains available; sentence matching is disabled.",
    "warn"
  );
}


// ==================================================================
// Dynamic browser dependency loading
// ==================================================================

async function loadBrowserDependencies() {
  setLoading("Loading browser ML libraries…");

  try {
    log("Importing @mediapipe/tasks-vision...");
    ({
      FilesetResolver,
      HandLandmarker,
      PoseLandmarker
    } = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm"
    ));

    log("MediaPipe Tasks Vision import succeeded.");
  } catch (error) {
    throw new Error(
      `MediaPipe JS import failed: ${error.message}`
    );
  }

  try {
    log("Importing onnxruntime-web...");
    ort = await import(
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/+esm"
    );

    log("ONNX Runtime Web import succeeded.");

    /*
     * Make the WASM files explicit instead of relying on automatic
     * path guessing. This is important when the application runs
     * from localhost.
     */
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

      /*
       * RESPONSIVENESS FIX (1/3):
       * proxy=true runs the WASM inference session inside a Web Worker
       * instead of the main thread. Without this, every session.run()
       * call blocks the UI thread (buttons, video paint, key handling)
       * for the duration of inference. With proxy on, ORT ships the
       * tensors to a worker and inference happens off-thread, so the
       * page stays responsive even while a prediction is running.
       */
      ort.env.wasm.proxy = true;

      /*
       * Use multiple WASM threads when the page is cross-origin
       * isolated (required for SharedArrayBuffer). Falls back to 1
       * automatically otherwise. More threads = faster inference =
       * shorter main-thread-adjacent stalls.
       */
      ort.env.wasm.numThreads =
        window.crossOriginIsolated
          ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))
          : 1;
    }

    if (ort.env?.webgpu) {
      ort.env.webgpu.powerPreference = "high-performance";
    }
  } catch (error) {
    throw new Error(
      `ONNX Runtime Web import failed: ${error.message}`
    );
  }

  dependenciesReady = true;
}


// ==================================================================
// MediaPipe initialization
// ==================================================================

async function createMediaPipe() {
  setLoading("Checking MediaPipe task files…");

  await checkAsset(CFG.pose_task_path);
  await checkAsset(CFG.hand_task_path);

  setLoading("Starting MediaPipe Vision…");

  /*
   * CPU is deliberately the default, matching your supplied config.
   * This avoids unreliable delegate behavior on some machines.
   */
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
  );

  setLoading("Creating pose detector…");

  poseLandmarker =
    await PoseLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath: CFG.pose_task_path
        },
        runningMode: "VIDEO",
        numPoses: 1
      }
    );

  log("PoseLandmarker ready.");

  setLoading("Creating hand detector…");

  handLandmarker =
    await HandLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath: CFG.hand_task_path
        },
        runningMode: "VIDEO",
        numHands: 2
      }
    );

  log("HandLandmarker ready.");

  mediaPipeReady = true;
}


// ==================================================================
// ONNX initialization
// ==================================================================

async function createONNX() {
  setLoading("Checking ONNX model…");

  await checkAsset(CFG.onnx_model_path);

  setLoading("Loading ONNX model…");

  const availableProviders = [];

  if (
    CFG.prefer_webgpu_onnx &&
    "gpu" in navigator
  ) {
    availableProviders.push("webgpu");
  }

  availableProviders.push("wasm");

  let lastError = null;

  for (const provider of availableProviders) {
    try {
      log(`Trying ONNX execution provider: ${provider}`);

      const options = {
        executionProviders: [provider],
        graphOptimizationLevel: "all"
      };

      onnxSession =
        await ort.InferenceSession.create(
          CFG.onnx_model_path,
          options
        );

      log(
        `ONNX model loaded successfully using ${provider}.`
      );

      lastError = null;
      break;
    } catch (error) {
      lastError = error;

      log(
        `ONNX provider ${provider} failed: ${error.message}`,
        "warn"
      );
    }
  }

  if (!onnxSession) {
    throw new Error(
      `ONNX model could not be initialized. ${lastError?.message || ""}`
    );
  }

  modelInputName =
    onnxSession.inputNames[0];

  modelOutputName =
    onnxSession.outputNames[0];

  log(`ONNX input name: ${modelInputName}`);
  log(`ONNX output name: ${modelOutputName}`);

  /*
   * Useful sanity check before using the model.
   */
  try {
    const metadata =
      onnxSession.inputMetadata?.[modelInputName];

    if (metadata) {
      log(
        `ONNX input metadata: ${JSON.stringify(metadata)}`
      );
    }
  } catch {
    // Metadata isn't required for execution.
  }

  onnxReady = true;
}


// ==================================================================
// Camera
// ==================================================================

async function enumerateCameras() {
  const devices =
    await navigator.mediaDevices.enumerateDevices();

  const cameras =
    devices.filter(
      (device) => device.kind === "videoinput"
    );

  log(`Browser reports ${cameras.length} camera(s).`);

  cameras.forEach((camera, index) => {
    log(
      `Camera ${index}: ${camera.label || "(label hidden until permission)"}`
    );
  });

  return cameras;
}

async function startCamera() {
  if (cameraOn && cameraStream) {
    return true;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Camera API is unavailable. Open through http://localhost:8000."
    );
  }

  hideError();

  setLoading("Requesting camera permission…");

  /*
   * Request the camera directly first.
   *
   * This is much more reliable than trying to map Python's numeric
   * camera_index before browser permission has been granted.
   */
  let requestedStream;

  try {
    requestedStream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        }
      });
  } catch (error) {
    throw new Error(
      `Browser denied/failed camera access: ${error.name}: ${error.message}`
    );
  }

  cameraStream = requestedStream;

  video.srcObject = cameraStream;

  await video.play();

  /*
   * Wait until video dimensions are actually available.
   */
  if (
    video.readyState < HTMLMediaElement.HAVE_METADATA
  ) {
    await new Promise((resolve) => {
      video.addEventListener(
        "loadedmetadata",
        resolve,
        { once: true }
      );
    });
  }

  cameraOn = true;

  cameraPlaceholder.style.display = "none";

  cameraBtn.textContent = "Camera On";

  setStatus(
    recognizing ? "PREDICTING" : "CAMERA READY",
    recognizing ? "on" : "off"
  );

  log(
    `Camera active: ${video.videoWidth}x${video.videoHeight}`
  );

  hideLoading();

  setButtonState();

  /*
   * Now that permission is granted, enumerate devices and report them.
   */
  await enumerateCameras().catch((error) => {
    log(
      `Camera enumeration warning: ${error.message}`,
      "warn"
    );
  });

  return true;
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
  }

  cameraStream = null;
  video.srcObject = null;
  cameraOn = false;
  recognizing = false;

  frameWindow.length = 0;

  cameraPlaceholder.style.display = "flex";
  cameraBtn.textContent = "Camera On";

  setStatus("PAUSED — camera off", "off");

  setButtonState();

  log("Camera stopped.");
}


// ==================================================================
// MediaPipe landmark extraction
// ==================================================================

/*
 * RESPONSIVENESS FIX (2/3):
 * Yields one macrotask back to the browser's event loop. MediaPipe's
 * detectForVideo() calls are synchronous and each one can take several
 * milliseconds. Running pose detection immediately followed by hand
 * detection immediately followed by ONNX prep, all in one unbroken
 * stretch, is what made clicks/keypresses feel laggy -- the browser
 * never got a turn to handle input or paint between them. Awaiting
 * this after each heavy step gives the event loop a chance to process
 * pending UI events before continuing.
 */
function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function extractLandmarks() {
  /*
   * Exact model layout from main.py:
   *
   * 21 left-hand landmarks
   * 21 right-hand landmarks
   * 33 pose landmarks
   * ----------------------
   * 75 landmarks
   * 2 values per landmark: x, y
   *
   * Shape:
   *   [75][2]
   */

  const landmarks = new Float32Array(75 * 2);

  const poseResult =
    poseLandmarker.detectForVideo(
      video,
      mediaTimestamp
    );

  await yieldToBrowser();

  const handResult =
    handLandmarker.detectForVideo(
      video,
      mediaTimestamp
    );

  await yieldToBrowser();

  // Pose
  if (poseResult?.landmarks?.length) {
    const pose = poseResult.landmarks[0];

    for (
      let i = 0;
      i < Math.min(33, pose.length);
      i++
    ) {
      const index = 42 + i;

      landmarks[index * 2] = pose[i].x;
      landmarks[index * 2 + 1] = pose[i].y;
    }
  }

  // Hands
  if (handResult?.landmarks?.length) {
    for (
      let h = 0;
      h < handResult.landmarks.length;
      h++
    ) {
      const hand =
        handResult.landmarks[h];

      const handedness =
        handResult.handedness?.[h]?.[0]?.categoryName
        || "";

      let offset = null;

      if (
        handedness.toLowerCase() ===
        "left"
      ) {
        offset = 0;
      }

      if (
        handedness.toLowerCase() ===
        "right"
      ) {
        offset = 21;
      }

      if (offset === null) {
        continue;
      }

      for (
        let i = 0;
        i < Math.min(21, hand.length);
        i++
      ) {
        const index = offset + i;

        landmarks[index * 2] = hand[i].x;
        landmarks[index * 2 + 1] = hand[i].y;
      }
    }
  }

  return landmarks;
}


// ==================================================================
// Preprocessing - exact Python equivalent
// ==================================================================

function normalizeSequence(sequence) {
  const frameCount = sequence.length;
  const output =
    new Float32Array(
      frameCount * 75 * 2
    );

  for (
    let frame = 0;
    frame < frameCount;
    frame++
  ) {
    const srcFrame =
      sequence[frame];

    const leftShoulder =
      (42 + 11) * 2;

    const rightShoulder =
      (42 + 12) * 2;

    const lx = srcFrame[leftShoulder];
    const ly = srcFrame[leftShoulder + 1];

    const rx = srcFrame[rightShoulder];
    const ry = srcFrame[rightShoulder + 1];

    const shouldersDetected =
      !(
        lx === 0 &&
        ly === 0
      ) &&
      !(
        rx === 0 &&
        ry === 0
      );

    if (!shouldersDetected) {
      /*
       * Exactly like:
       * seq[~shoulders_detected] = 0
       */
      continue;
    }

    const centerX =
      (lx + rx) / 2;

    const centerY =
      (ly + ry) / 2;

    let shoulderWidth =
      Math.hypot(
        rx - lx,
        ry - ly
      );

    if (shoulderWidth < 0.05) {
      shoulderWidth = 0.3;
    }

    for (
      let p = 0;
      p < 75;
      p++
    ) {
      const src =
        p * 2;

      const dst =
        frame * 75 * 2 +
        src;

      output[dst] =
        (srcFrame[src] - centerX) /
        shoulderWidth;

      output[dst + 1] =
        (srcFrame[src + 1] - centerY) /
        shoulderWidth;
    }
  }

  return output;
}

function addMotionFeatures(normalized) {
  const frameCount = CFG.fixed_len;

  const output =
    new Float32Array(
      frameCount * 75 * 4
    );

  for (
    let frame = 0;
    frame < frameCount;
    frame++
  ) {
    for (
      let landmark = 0;
      landmark < 75;
      landmark++
    ) {
      const xy =
        frame * 75 * 2 +
        landmark * 2;

      const out =
        frame * 75 * 4 +
        landmark * 4;

      const x =
        normalized[xy];

      const y =
        normalized[xy + 1];

      let vx = 0;
      let vy = 0;

      if (frame > 0) {
        const prev =
          (frame - 1) * 75 * 2 +
          landmark * 2;

        vx =
          x -
          normalized[prev];

        vy =
          y -
          normalized[prev + 1];
      }

      output[out] = x;
      output[out + 1] = y;
      output[out + 2] = vx;
      output[out + 3] = vy;
    }
  }

  return output;
}


// ==================================================================
// ONNX prediction
// ==================================================================

function softmax(logits) {
  let max =
    Number.NEGATIVE_INFINITY;

  for (const value of logits) {
    if (value > max) {
      max = value;
    }
  }

  const exps =
    new Float32Array(
      logits.length
    );

  let sum = 0;

  for (
    let i = 0;
    i < logits.length;
    i++
  ) {
    const value =
      Math.exp(
        logits[i] - max
      );

    exps[i] = value;
    sum += value;
  }

  if (sum <= 0) {
    return exps;
  }

  for (
    let i = 0;
    i < exps.length;
    i++
  ) {
    exps[i] /= sum;
  }

  return exps;
}

async function predictCurrentWindow() {
  if (
    !onnxSession ||
    frameWindow.length !==
      CFG.fixed_len
  ) {
    return null;
  }

  /*
   * 1. Normalize
   * 2. Add motion
   * 3. Reshape [1, 40, 300]
   */
  const normalized =
    normalizeSequence(
      frameWindow
    );

  const features =
    addMotionFeatures(
      normalized
    );

  const tensor =
    new ort.Tensor(
      "float32",
      features,
      [
        1,
        CFG.fixed_len,
        75 * 4
      ]
    );

  const result =
    await onnxSession.run({
      [modelInputName]:
        tensor
    });

  const output =
    result[modelOutputName];

  if (!output?.data) {
    throw new Error(
      "ONNX output tensor is empty."
    );
  }

  const probabilities =
    softmax(output.data);

  let first = -1;
  let second = -1;

  for (
    let i = 0;
    i < probabilities.length;
    i++
  ) {
    if (
      first < 0 ||
      probabilities[i] >
        probabilities[first]
    ) {
      second = first;
      first = i;
    } else if (
      second < 0 ||
      probabilities[i] >
        probabilities[second]
    ) {
      second = i;
    }
  }

  return {
    sign:
      indexToSign[first] ??
      `class_${first}`,

    confidence:
      Number(
        probabilities[first] || 0
      ),

    secondSign:
      indexToSign[second] ??
      `class_${second}`,

    secondConfidence:
      Number(
        probabilities[second] || 0
      )
  };
}


// ==================================================================
// Recognition filtering
// ==================================================================

function isConfusablePair(a, b) {
  return (
    CFG.confusable_pairs || []
  ).some((pair) => {
    if (
      !Array.isArray(pair) ||
      pair.length !== 2
    ) {
      return false;
    }

    return (
      (
        pair[0] === a &&
        pair[1] === b
      ) ||
      (
        pair[0] === b &&
        pair[1] === a
      )
    );
  });
}

function applyOverride(sign) {
  if (
    CFG.label_overrides &&
    Object.prototype.hasOwnProperty.call(
      CFG.label_overrides,
      sign
    )
  ) {
    return CFG.label_overrides[sign];
  }

  return sign;
}

function matchSentence(words) {
  for (
    const template of sentenceTemplates
  ) {
    if (
      !Array.isArray(
        template?.words
      )
    ) {
      continue;
    }

    const n = template.words.length;

    if (
      n === 0 ||
      n > words.length
    ) {
      continue;
    }

    const tail =
      words.slice(
        words.length - n
      );

    let same = true;

    for (
      let i = 0;
      i < n;
      i++
    ) {
      if (
        tail[i] !==
        template.words[i]
      ) {
        same = false;
        break;
      }
    }

    if (same) {
      return (
        template.sentence ||
        null
      );
    }
  }

  return null;
}


// ==================================================================
// Speech
// ==================================================================

function speak(text) {
  if (
    !("speechSynthesis" in window)
  ) {
    log(
      "Web Speech API is unavailable; sign output will still appear on screen.",
      "warn"
    );
    return;
  }

  if (!text) {
    return;
  }

  const utterance =
    new SpeechSynthesisUtterance(
      formatSign(text)
    );

  /*
   * Python:
   *   pyttsx3.setProperty("rate", 165)
   *
   * Browser API uses a normalized rate.
   */
  utterance.rate =
    Math.min(
      2,
      Math.max(
        0.5,
        Number(CFG.tts_rate || 165) /
          165
      )
    );

  utterance.pitch = 1;
  utterance.volume = 1;

  /*
   * Don't queue a long list of old words.
   */
  window.speechSynthesis.cancel();

  window.speechSynthesis.speak(
    utterance
  );
}


// ==================================================================
// Confirm sign -> word -> sentence -> speech
// ==================================================================

function confirmSign(sign) {
  wordBuffer.push(sign);

  while (
    wordBuffer.length >
    CFG.word_buffer_size
  ) {
    wordBuffer.shift();
  }

  cooldownSign = sign;
  cooldownRemaining =
    CFG.same_sign_cooldown_frames;

  stableCount = 0;
  confirmedSigns++;

  lastConfirmedTime =
    performance.now() /
    1000;

  let sentence = null;

  if (
    CFG.speech_mode ===
      "words_and_sentences" ||
    CFG.speech_mode ===
      "sentence_only"
  ) {
    sentence =
      matchSentence(
        wordBuffer
      );
  }

  if (sentence) {
    lastSentence =
      sentence;

    sentenceDisplayUntil =
      performance.now() +
      3000;

    log(
      `Sentence matched: "${sentence}"`
    );

    speak(sentence);

    /*
     * Exact supplied Python behavior:
     * sentence completed -> buffer cleared.
     */
    wordBuffer = [];
  } else if (
    CFG.speech_mode ===
      "words_and_sentences" ||
    CFG.speech_mode ===
      "words_only"
  ) {
    speak(sign);
  }

  updateUI();
}


// ==================================================================
// Prediction result handling
// ==================================================================

function handlePrediction(result) {
  if (!result) {
    return;
  }

  const {
    sign: rawSign,
    confidence,
    secondSign,
    secondConfidence
  } = result;

  const pair =
    isConfusablePair(
      rawSign,
      secondSign
    );

  const margin =
    confidence -
    secondConfidence;

  const marginOK =
    !pair ||
    margin >=
      CFG.confusable_margin;

  /*
   * Override AFTER confusable checking,
   * exactly like the supplied Python.
   */
  const sign =
    applyOverride(
      rawSign
    );

  if (
    confidence <=
      CFG.confidence_threshold
  ) {
    updateUI(
      `${formatSign(sign)} — low confidence ` +
      `(${Math.round(confidence * 100)}%)`
    );

    return;
  }

  if (!marginOK) {
    updateUI(
      `${formatSign(sign)} — unclear vs ` +
      `${formatSign(secondSign)}`
    );

    return;
  }

  if (
    sign ===
    lastPrediction
  ) {
    stableCount++;
  } else {
    lastPrediction = sign;
    stableCount = 1;
  }

  if (
    cooldownRemaining > 0
  ) {
    cooldownRemaining--;

    if (
      sign !==
      cooldownSign
    ) {
      cooldownRemaining = 0;
    }
  }

  const canConfirm =
    stableCount >=
      CFG.stability_frames &&
    !(
      sign ===
        cooldownSign &&
      cooldownRemaining > 0
    );

  if (canConfirm) {
    log(
      `Confirmed: ${sign} (${Math.round(confidence * 100)}%)`
    );

    confirmSign(sign);
  }

  let display =
    `${formatSign(sign)} ` +
    `(${Math.round(confidence * 100)}%) ` +
    `[${stableCount}/${CFG.stability_frames}]`;

  if (
    pair &&
    !marginOK
  ) {
    display +=
      ` — unclear vs ${formatSign(secondSign)}`;
  }

  updateUI(display);
}


// ==================================================================
// Video / frame loop
// ==================================================================

function nextTimestamp() {
  /*
   * MediaPipe VIDEO mode requires timestamps to increase.
   *
   * performance.now() is a safe monotonically increasing source.
   */
  const now =
    performance.now();

  if (
    now <= mediaTimestamp
  ) {
    mediaTimestamp++;
  } else {
    mediaTimestamp =
      Math.floor(now);
  }

  return mediaTimestamp;
}

async function processVideoFrame(now) {
  if (
    !recognizing ||
    !cameraOn ||
    !mediaPipeReady ||
    !onnxReady ||
    processing
  ) {
    return;
  }

  if (
    now -
      lastProcessTime <
    Number(
      CFG.process_interval_ms ||
      40
    )
  ) {
    return;
  }

  lastProcessTime =
    now;

  /*
   * Never process before the video actually has pixels.
   */
  if (
    video.readyState <
      HTMLMediaElement.HAVE_CURRENT_DATA ||
    video.videoWidth <= 0 ||
    video.videoHeight <= 0
  ) {
    return;
  }

  processing = true;

  try {
    nextTimestamp();

    const landmarks =
      await extractLandmarks();

    frameWindow.push(
      landmarks
    );

    /*
     * Sliding 40-frame window.
     */
    while (
      frameWindow.length >
      CFG.fixed_len
    ) {
      frameWindow.shift();
    }

    if (
      frameWindow.length ===
      CFG.fixed_len
    ) {
      const result =
        await predictCurrentWindow();

      handlePrediction(result);
    }

    /*
     * Auto-clear stale incomplete words.
     */
    if (
      wordBuffer.length &&
      lastConfirmedTime > 0
    ) {
      const seconds =
        performance.now() /
          1000 -
        lastConfirmedTime;

      if (
        seconds >
        CFG.word_buffer_timeout_seconds
      ) {
        wordBuffer = [];

        log(
          "Word buffer timed out and was cleared."
        );

        updateUI();
      }
    }

    if (
      lastSentence &&
      performance.now() >=
        sentenceDisplayUntil
    ) {
      updateUI();
    }
  } catch (error) {
    /*
     * Keep the camera alive even if one inference fails.
     */
    log(
      `Frame error: ${error.stack || error.message}`,
      "error"
    );
  } finally {
    processing = false;
  }
}

function renderLoop(now) {
  void processVideoFrame(now);

  animationFrame =
    requestAnimationFrame(
      renderLoop
    );
}


// ==================================================================
// User controls
// ==================================================================

async function cameraButtonAction() {
  if (cameraOn) {
    stopCamera();
    return;
  }

  try {
    await startCamera();
  } catch (error) {
    showError(
      "Camera access failed",
      error.message
    );

    log(
      `Camera error: ${error.stack || error.message}`,
      "error"
    );
  }
}

async function startRecognition() {
  if (!dependenciesReady) {
    showError(
      "Browser ML libraries are not ready",
      "See the technical log for the stage that failed."
    );
    return;
  }

  if (!mediaPipeReady || !onnxReady) {
    showError(
      "Recognition models are not ready",
      "MediaPipe or ONNX failed to initialize."
    );
    return;
  }

  try {
    if (!cameraOn) {
      await startCamera();
    }

    /*
     * Match Python's "fresh start" behavior.
     */
    frameWindow.length = 0;

    lastPrediction = null;
    stableCount = 0;

    cooldownSign = null;
    cooldownRemaining = 0;

    lastProcessTime = 0;

    recognizing = true;

    setStatus(
      "PREDICTING",
      "on"
    );

    setButtonState();

    log(
      "Recognition started. Waiting for first 40-frame window..."
    );

    updateUI();
  } catch (error) {
    showError(
      "Recognition could not start",
      error.message
    );

    log(
      `Recognition start error: ${error.stack || error.message}`,
      "error"
    );
  }
}

function pauseRecognition() {
  recognizing = false;

  frameWindow.length = 0;

  lastPrediction = null;
  stableCount = 0;

  setStatus(
    cameraOn
      ? "CAMERA READY"
      : "PAUSED — camera off",
    "off"
  );

  setButtonState();

  log(
    "Recognition paused."
  );
}

function clearBuffer() {
  wordBuffer = [];

  lastSentence = "";
  sentenceDisplayUntil = 0;

  frameWindow.length = 0;

  lastPrediction = null;
  stableCount = 0;

  cooldownSign = null;
  cooldownRemaining = 0;

  updateUI();

  log(
    "Word buffer and temporal prediction state cleared."
  );
}


// ==================================================================
// Initialization
// ==================================================================

async function initialize() {
  if (initializationStarted) {
    return;
  }

  initializationStarted = true;

  hideError();

  setLoading(
    "Checking browser and server…"
  );

  if (!checkEnvironment()) {
    initializationStarted = false;
    hideLoading();
    setButtonState();
    return;
  }

  /*
   * Enable camera testing immediately.
   * This is deliberate: camera access should NOT be hidden behind
   * successful ONNX / MediaPipe initialization.
   */
  cameraBtn.disabled = false;

  try {
    await loadConfiguration();

    /*
     * Labels and sentence data are independent from the camera.
     */
    try {
      await loadLabels();
    } catch (error) {
      log(
        `Label map failed: ${error.message}`,
        "error"
      );
    }

    try {
      await loadSentences();
    } catch (error) {
      log(
        `Sentence data failed: ${error.message}`,
        "warn"
      );
    }

    /*
     * ML libraries are loaded separately so a failed CDN import
     * produces an obvious error instead of a dead page.
     */
    try {
      await loadBrowserDependencies();
    } catch (error) {
      showError(
        "Browser ML library failed",
        error.message
      );

      log(
        error.stack || error.message,
        "error"
      );

      hideLoading();
      setButtonState();
      initializationStarted = false;
      return;
    }

    /*
     * MediaPipe and ONNX are initialized independently.
     */
    try {
      await createMediaPipe();
    } catch (error) {
      log(
        `MediaPipe initialization failed: ${error.message}`,
        "error"
      );
    }

    try {
      await createONNX();
    } catch (error) {
      log(
        `ONNX initialization failed: ${error.message}`,
        "error"
      );
    }

    if (
      mediaPipeReady &&
      onnxReady
    ) {
      dependenciesReady = true;

      setStatus(
        cameraOn
          ? "CAMERA READY — click Start"
          : "READY — click Start",
        "off"
      );

      log(
        "All recognition dependencies are ready."
      );
    } else {
      showError(
        "Recognition is not fully ready",
        "Camera testing can still work. Check the technical log for MediaPipe/ONNX asset errors."
      );

      setStatus(
        "MODEL SETUP ERROR",
        "error"
      );
    }

    hideLoading();

    setButtonState();
  } catch (error) {
    showError(
      "Initialization failed",
      error.message
    );

    log(
      error.stack || error.message,
      "error"
    );

    hideLoading();
    setButtonState();
  } finally {
    initializationStarted = false;
  }
}


// ==================================================================
// Events
// ==================================================================

startBtn.addEventListener(
  "click",
  startRecognition
);

stopBtn.addEventListener(
  "click",
  pauseRecognition
);

cameraBtn.addEventListener(
  "click",
  cameraButtonAction
);

clearBtn.addEventListener(
  "click",
  clearBuffer
);

errorRetryBtn.addEventListener(
  "click",
  () => {
    hideError();

    /*
     * Reset only initialization state.
     * Camera, if running, is left intact.
     */
    initializationStarted = false;

    void initialize();
  }
);

errorDismissBtn.addEventListener(
  "click",
  hideError
);

/*
 * Python's SPACE key equivalent.
 */
window.addEventListener(
  "keydown",
  (event) => {
    if (
      event.code !==
      "Space"
    ) {
      return;
    }

    if (
      event.target instanceof
        HTMLInputElement ||
      event.target instanceof
        HTMLTextAreaElement
    ) {
      return;
    }

    event.preventDefault();

    if (recognizing) {
      pauseRecognition();
    } else if (
      dependenciesReady
    ) {
      void startRecognition();
    }
  }
);

window.addEventListener(
  "beforeunload",
  () => {
    cancelAnimationFrame(
      animationFrame
    );

    if (cameraStream) {
      for (
        const track of
        cameraStream.getTracks()
      ) {
        track.stop();
      }
    }

    try {
      window.speechSynthesis?.cancel();
    } catch {
      // Ignore shutdown errors.
    }

    try {
      poseLandmarker?.close();
      handLandmarker?.close();
    } catch {
      // Ignore cleanup errors.
    }
  }
);


// ==================================================================
// Start
// ==================================================================

setStatus(
  "STARTING…",
  "off"
);

updateUI();

setButtonState();

animationFrame =
  requestAnimationFrame(
    renderLoop
  );

void initialize();