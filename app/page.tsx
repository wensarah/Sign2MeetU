"use client";

// Main app. Pipeline:
//   front camera ──► MediaPipe hands ──► SignSegmenter
//   hands enter the frame ──► recording starts automatically
//                         ──► one front-camera still ──► /api/scene (runs while you sign)
//   each sign ──► kNN classifier (top 3) + video clip ──► Perceptron decides,
//                 and names the body part / object for pointing signs (HERE, THAT, PAIN)
//   no hands for 2s (or "Done", or 15s limit) ──► signs + scene + recent sentences
//                 ──► /api/sentence ──► show + speak
import { useCallback, useEffect, useRef, useState } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { loadHandLandmarker, frameFeatures, makeFingerprint, drawHands, type Frame } from "@/lib/landmarks";
import { findPoint, findLocation, drawMark, type PointMark } from "@/lib/pointing";
import { predict, mergeDatasets, emptyDataset, type Dataset, type Prediction } from "@/lib/knn";
import {
  SignSegmenter,
  SENTENCE_GAP_MS,
  RECORDING_MAX_MS,
  RECORDING_MIN_MS,
  type SegmenterEvent,
} from "@/lib/segmenter";
import { UNKNOWN, VOCAB, VOCAB_KEYS, textFor } from "@/lib/vocab";

const MIN_CONFIDENCE = 0.5; // Perceptron confidence we accept on its own
const KNN_FALLBACK = 0.6; // classifier confidence needed when Perceptron can't decide
const MIN_EXAMPLES_PER_SIGN = 3; // before that, the classifier only gives Perceptron hints
const MIN_SIGNERS = 2; // ...and the same until at least this many people have recorded
const DATASET_STORAGE_KEY = "sign2meetu-dataset"; // where /train saves examples on this device
const SCENE_WIDTH = 1024; // back-camera scan: more pixels = better object detection
const BACKGROUND_WIDTH = 640; // front-camera background while signing: speed matters more
const MARK_WIDTH = 960; // pointing stills
// Signs whose target is looked up on a marked still: a ray from the fingertip, or a circle.
const MARK_MODE: Record<string, PointMark["mode"]> = { THAT: "point", PAIN: "location" };
// Max lead-in before a sign in the clip sent to Perceptron. Keep it short: the
// model samples a limited number of frames, so idle time steals frames from the sign.
const IDLE_CLIP_MS = 1000;
const MARK_EVERY_MS = 150; // at most one pointing-still copy per this many ms (keeps hand tracking fast)
const AUTO_SCENE_WAIT_MS = 1500; // how long a finished sentence waits for the background scene
const HISTORY_SENTENCES = 2; // recent sentences sent along for context

type Phase = "starting" | "ready" | "listening" | "scanning" | "error";
type Facing = "user" | "environment";
type Scene = { place: string; objects: string[]; text: string[] };
/** Best still of each kind captured during one sign (see lib/pointing.ts). */
type MarkCapture = { canvas: HTMLCanvasElement; mark: PointMark; score: number; at: number };
type SignCaptures = { point?: MarkCapture; location?: MarkCapture };
type PerceptronResult = { sign: string; text: string; confidence: number; target?: string };
type Gloss = {
  id: number;
  status: "pending" | "done" | "skipped";
  key?: string;
  text?: string;
  target?: string; // body part or object pointed at (THAT, PAIN)
  markUrl?: string; // the marked still Perceptron looked at, if any
  via?: "perceptron" | "classifier";
};
type Sentence = { id: number; text: string; glosses: string[]; scenePlace: string; marks: string[] };

// iOS Safari records mp4; Chrome/Android record webm. Pick what this browser supports.
function pickMimeType(): string {
  const candidates = [
    "video/mp4;codecs=avc1",
    "video/mp4",
    "video/webm;codecs=vp8",
    "video/webm;codecs=vp9",
    "video/webm",
  ];
  if (typeof MediaRecorder === "undefined") return "";
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
}

/** Perceptron decides; the on-device classifier fills in when Perceptron can't. */
/**
 * The classifier may answer on its own only once it has seen every sign a few
 * times, from more than one signer (one person's data doesn't generalize).
 */
function classifierCoverage(dataset: Dataset) {
  const counts: Record<string, number> = {};
  for (const e of dataset.examples) counts[e.label] = (counts[e.label] ?? 0) + 1;
  const covered = VOCAB.filter((s) => (counts[s.key] ?? 0) >= MIN_EXAMPLES_PER_SIGN).length;
  const signers = new Set(dataset.examples.map((e) => e.signer)).size;
  return { covered, total: VOCAB.length, signers, ready: covered === VOCAB.length && signers >= MIN_SIGNERS };
}

function decide(
  p: PerceptronResult | null,
  knn: Prediction | null,
  classifierReady: boolean
): { key: string; via: Gloss["via"]; target: string } | null {
  const perceptronSign = p && p.sign !== UNKNOWN && textFor(p.sign) ? p.sign : null;
  // Only Perceptron can see where the signer pointed; keep its target if the sign agrees.
  const targetFor = (key: string) => (p && p.sign === key ? p.target ?? "" : "");
  if (perceptronSign && (p!.confidence >= MIN_CONFIDENCE || perceptronSign === knn?.label)) {
    return { key: perceptronSign, via: "perceptron", target: targetFor(perceptronSign) };
  }
  if (classifierReady && knn && knn.label !== UNKNOWN && knn.confidence >= KNN_FALLBACK) {
    return { key: knn.label, via: "classifier", target: targetFor(knn.label) };
  }
  if (perceptronSign) return { key: perceptronSign, via: "perceptron", target: targetFor(perceptronSign) };
  return null;
}

function mergeScenes(manual: Scene | null, auto: Scene | null): Scene | null {
  if (!manual) return auto;
  if (!auto) return manual;
  // The deliberate scan is more specific, so it comes first.
  const objects = [...new Set([...manual.objects, ...auto.objects])].slice(0, 8);
  const text = [...new Set([...manual.text, ...auto.text])].slice(0, 5);
  return { place: manual.place || auto.place, objects, text };
}

function toScene(data: { place?: string; objects?: string[]; text?: string[] }): Scene | null {
  const place = data.place ?? "";
  const objects = data.objects ?? [];
  const text = data.text ?? [];
  if ((!place || place === "unknown") && !objects.length && !text.length) return null;
  return { place, objects, text };
}

/** Copy the current video frame into a new canvas (at most `width` pixels wide). */
function snapshot(video: HTMLVideoElement, width: number): HTMLCanvasElement {
  const vw = video.videoWidth || width;
  const vh = video.videoHeight || Math.round(width * 0.75);
  const scale = Math.min(1, width / vw);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(vw * scale);
  canvas.height = Math.round(vh * scale);
  canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

const toJpeg = (canvas: HTMLCanvasElement, quality: number): Promise<Blob | null> =>
  new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Small line icons (Lucide shapes), drawn inline so they inherit the text color.
function Icon({ children, className = "h-6 w-6" }: { children: React.ReactNode; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const CameraIcon = ({ className }: { className?: string }) => (
  <Icon className={className}>
    <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
    <circle cx="12" cy="13" r="3" />
  </Icon>
);

const ReplayIcon = ({ className }: { className?: string }) => (
  <Icon className={className}>
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
  </Icon>
);

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const segmenterRef = useRef(new SignSegmenter());
  const datasetRef = useRef<Dataset>(emptyDataset());
  const clipRef = useRef<{ recorder: MediaRecorder; done: Promise<Blob | null>; startedAt: number } | null>(null);
  const clipTickRef = useRef<(now: number, signing: boolean) => void>(() => {});
  const capturesRef = useRef<SignCaptures>({}); // pointing stills for the sign in progress
  const pointTickRef = useRef<(result: HandLandmarkerResult, video: HTMLVideoElement, now: number) => void>(() => {});
  const pendingRef = useRef<{ id: number; job: Promise<Gloss> }[]>([]); // signs in the current sentence
  const phaseRef = useRef<Phase>("starting");
  const sceneRef = useRef<Scene | null>(null); // from the back-camera scan
  const autoSceneRef = useRef<Promise<Scene | null> | null>(null); // background of the current recording
  const historyRef = useRef<string[]>([]); // recent English sentences, oldest first
  const idRef = useRef(0);
  const handlersRef = useRef<(ev: SegmenterEvent) => void>(() => {});
  const speechUnlocked = useRef(false);

  const [phase, setPhaseState] = useState<Phase>("starting");
  const [error, setError] = useState("");
  const [facing, setFacing] = useState<Facing>("user");
  const [handsVisible, setHandsVisible] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0); // whole seconds into the current recording
  const [progress, setProgress] = useState(0); // 0..1 toward "sentence done"
  const [glosses, setGlosses] = useState<Gloss[]>([]);
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const [translating, setTranslating] = useState(0); // number of sentences being translated
  const [notice, setNotice] = useState("");
  const [scene, setSceneState] = useState<Scene | null>(null);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [sceneError, setSceneError] = useState("");
  const [classifier, setClassifier] = useState({ examples: 0, covered: 0, total: VOCAB.length, signers: 0, ready: false });
  const [speak, setSpeak] = useState(true);
  const [showInfo, setShowInfo] = useState(false);

  // The camera loop runs outside React, so it reads these through refs.
  const setPhase = (p: Phase) => {
    phaseRef.current = p;
    setPhaseState(p);
  };
  const setScene = (s: Scene | null) => {
    sceneRef.current = s;
    setSceneState(s);
  };

  // --- Camera ------------------------------------------------------------
  const openCamera = useCallback(async (facingMode: Facing) => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode,
        // Front: small frames keep hand tracking fast. Back (scene scan): more detail for object detection.
        width: { ideal: facingMode === "environment" ? 1280 : 640 },
        height: { ideal: facingMode === "environment" ? 960 : 480 },
        frameRate: { ideal: 24, max: 30 },
      },
    });
    streamRef.current = stream;
    setFacing(facingMode);
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => {});
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let lastTime = -1;

    async function loadDataset() {
      let team = emptyDataset();
      try {
        const res = await fetch("/dataset.json", { cache: "no-store" });
        if (res.ok) team = (await res.json()) as Dataset;
      } catch {
        /* no team dataset yet: Perceptron works alone */
      }
      let local = emptyDataset();
      try {
        const saved = localStorage.getItem(DATASET_STORAGE_KEY);
        if (saved) local = JSON.parse(saved) as Dataset;
      } catch {
        /* private browsing etc. */
      }
      // Ignore examples of signs that are no longer in the vocabulary.
      const merged = mergeDatasets(team, local);
      datasetRef.current = { ...merged, examples: merged.examples.filter((e) => VOCAB_KEYS.includes(e.label)) };
      setClassifier({ examples: datasetRef.current.examples.length, ...classifierCoverage(datasetRef.current) });
    }

    async function start() {
      if (!window.isSecureContext) {
        setError("Camera needs HTTPS. Open the Vercel URL or use an ngrok tunnel.");
        setPhase("error");
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
        setError("This browser can't record video. Try Safari on iPhone or Chrome on Android.");
        setPhase("error");
        return;
      }
      try {
        await openCamera("user");
      } catch (e) {
        const name = e instanceof DOMException ? e.name : "";
        setError(
          name === "NotAllowedError"
            ? "Camera permission was denied. Allow camera access in your browser settings and reload."
            : "Couldn't open the camera. Is another app using it?"
        );
        setPhase("error");
        return;
      }
      if (cancelled) return;

      loadDataset();
      try {
        landmarkerRef.current = await loadHandLandmarker();
      } catch {
        setError("Couldn't load the hand tracker. Check your connection and reload.");
        setPhase("error");
        return;
      }
      if (cancelled) return;
      setPhase("listening"); // auto start: recording begins when hands enter the frame

      const loop = () => {
        const v = videoRef.current;
        const lm = landmarkerRef.current;
        if (v && lm && v.readyState >= 2 && v.currentTime !== lastTime && phaseRef.current !== "scanning") {
          lastTime = v.currentTime;
          const now = performance.now();
          const result = lm.detectForVideo(v, now);
          setHandsVisible(result.landmarks.length > 0);

          const c = canvasRef.current;
          if (c) {
            if (c.width !== v.videoWidth) {
              c.width = v.videoWidth;
              c.height = v.videoHeight;
            }
            drawHands(c.getContext("2d")!, result);
          }

          if (phaseRef.current === "listening") {
            const seg = segmenterRef.current;
            for (const ev of seg.push(frameFeatures(result), now)) handlersRef.current(ev);
            clipTickRef.current(now, seg.signing);
            if (seg.signing) pointTickRef.current(result, v, now);
            setRecording(seg.recording);
            setElapsed(Math.floor(seg.recordingMs(now) / 1000));
            setProgress(Math.round(seg.sentenceProgress(now) * 20) / 20);
          }
        }
        raf = requestAnimationFrame(loop);
      };
      loop();
    }

    start();
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [openCamera]);

  // --- Speech ------------------------------------------------------------
  // iOS only allows speech after a user gesture, so we "unlock" it on the first tap.
  const unlockSpeech = () => {
    if (speechUnlocked.current || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    window.speechSynthesis.speak(u);
    speechUnlocked.current = true;
  };

  const say = useCallback(
    (text: string) => {
      if (!speak || !text || !("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.95;
      window.speechSynthesis.speak(u);
    },
    [speak]
  );

  // --- One clip per sign, for Perceptron -----------------------------------
  // The recorder runs the whole time we're listening, so each sign's clip starts
  // BEFORE the hands come up (nothing gets cut off). While idle it restarts
  // every IDLE_CLIP_MS so that lead-in stays short.
  const startClip = () => {
    const stream = streamRef.current;
    if (!stream || clipRef.current) return;
    const mimeType = pickMimeType();
    try {
      const recorder = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
        videoBitsPerSecond: 600_000, // even a 7s clip stays well under 1 MB
      });
      const chunks: Blob[] = [];
      const done = new Promise<Blob | null>((resolve) => {
        recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
        recorder.onerror = () => resolve(null);
        recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || mimeType || "video/webm" }));
      });
      recorder.start();
      clipRef.current = { recorder, done, startedAt: performance.now() };
    } catch {
      /* recording unsupported: the classifier still works */
    }
  };

  /** Called every camera frame while listening: keep a fresh clip running. */
  clipTickRef.current = (now: number, signing: boolean) => {
    const clip = clipRef.current;
    if (!clip) return startClip();
    if (!signing && now - clip.startedAt > IDLE_CLIP_MS) {
      stopClip(); // nothing was signed in it: throw it away
      startClip();
    }
  };

  const stopClip = (): Promise<Blob | null> => {
    const clip = clipRef.current;
    clipRef.current = null;
    if (!clip) return Promise.resolve(null);
    if (clip.recorder.state !== "inactive") clip.recorder.stop();
    return clip.done;
  };

  // --- Pointing stills: keep the clearest pointing / jab frame of each sign ----
  pointTickRef.current = (result: HandLandmarkerResult, video: HTMLVideoElement, now: number) => {
    const caps = capturesRef.current;
    const keep = (kind: keyof SignCaptures, found: { score: number; mark: PointMark } | null) => {
      const prev = caps[kind];
      if (!found || (prev && (found.score <= prev.score || now - prev.at < MARK_EVERY_MS))) return;
      // Reuse this sign's canvas instead of allocating a new one every time.
      const canvas = prev?.canvas ?? snapshot(video, MARK_WIDTH);
      if (prev) canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
      caps[kind] = { canvas, mark: found.mark, score: found.score, at: now };
    };
    keep("point", findPoint(result));
    keep("location", findLocation(result));
  };

  /** Ask Perceptron what the marked still points at. Returns "" if it can't tell. */
  const lookUpTarget = async (capture: MarkCapture): Promise<{ target: string; url: string }> => {
    drawMark(capture.canvas, capture.mark);
    const photo = await toJpeg(capture.canvas, 0.85);
    if (!photo) return { target: "", url: "" };
    const url = URL.createObjectURL(photo); // shown in the result so people can see what Perceptron saw
    try {
      const form = new FormData();
      form.append("image", photo, "point.jpg");
      form.append("mode", capture.mark.mode);
      const res = await fetch("/api/point", { method: "POST", body: form });
      const data = await res.json();
      return { target: res.ok && data.kind !== "unknown" ? data.target ?? "" : "", url };
    } catch {
      return { target: "", url };
    }
  };

  // --- Recognize one sign --------------------------------------------------
  const recognize = async (frames: (Frame | null)[], clip: Promise<Blob | null>, caps: SignCaptures) => {
    const pick = await recognizeSign(frames, clip);
    const mode = pick && MARK_MODE[pick.key];
    const capture = mode === "point" ? caps.point : mode === "location" ? caps.location : undefined;
    if (!pick || !capture) return pick ? { ...pick, markUrl: "" } : null;
    // A marked still is more precise than the video clip; keep the clip's guess as backup.
    const { target, url } = await lookUpTarget(capture);
    return { ...pick, target: target || pick.target, markUrl: url };
  };

  const recognizeSign = async (frames: (Frame | null)[], clip: Promise<Blob | null>) => {
    const seq = makeFingerprint(frames);
    const examples = datasetRef.current.examples;
    const knn = seq && examples.length ? predict(seq, examples) : null;

    let perceptron: PerceptronResult | null = null;
    const blob = await clip;
    if (blob && blob.size > 0) {
      try {
        const form = new FormData();
        form.append("video", blob, `sign.${blob.type.includes("mp4") ? "mp4" : "webm"}`);
        if (knn) form.append("candidates", JSON.stringify(knn.top));
        const res = await fetch("/api/translate", { method: "POST", body: form });
        if (res.ok) perceptron = (await res.json()) as PerceptronResult;
      } catch {
        /* network trouble: fall back to the classifier */
      }
    }
    const pick = decide(perceptron, knn, classifierCoverage(datasetRef.current).ready);
    // Open the browser console to see which recognizer got each sign right or wrong.
    console.info(
      `[sign] → ${pick ? `${pick.key} (via ${pick.via})` : "skipped"} | Perceptron: ${
        perceptron ? `${perceptron.sign} ${Math.round(perceptron.confidence * 100)}%` : "no answer"
      } | classifier: ${knn ? knn.top.map((t) => `${t.label} ${Math.round(t.score * 100)}%`).join(", ") : "none"} | clip ${
        blob ? `${Math.round(blob.size / 1024)} KB` : "missing"
      }`
    );
    return pick;
  };

  // --- Background scene, captured from the front camera while signing --------
  const captureAutoScene = async (): Promise<Scene | null> => {
    const video = videoRef.current;
    if (!video) return null;
    try {
      const photo = await toJpeg(snapshot(video, BACKGROUND_WIDTH), 0.7);
      if (!photo) return null;
      const form = new FormData();
      form.append("image", photo, "background.jpg");
      const res = await fetch("/api/scene", { method: "POST", body: form });
      if (!res.ok) return null;
      return toScene(await res.json());
    } catch {
      return null; // translation still works without it
    }
  };

  // --- Throw away a recording that was too short -----------------------------
  const discardSentence = () => {
    autoSceneRef.current = null;
    const jobs = pendingRef.current;
    pendingRef.current = [];
    const ids = new Set(jobs.map((j) => j.id));
    setGlosses((g) => g.filter((x) => !ids.has(x.id)));
    if (jobs.length) {
      setNotice(`That was shorter than ${RECORDING_MIN_MS / 1000} second, so it was skipped. Sign a little longer.`);
    }
  };

  // --- Turn the finished sentence into English -------------------------------
  const finishSentence = async () => {
    const jobs = pendingRef.current;
    const autoScene = autoSceneRef.current;
    pendingRef.current = [];
    autoSceneRef.current = null;
    if (!jobs.length) return;

    setTranslating((n) => n + 1);
    try {
      const results = await Promise.all(jobs.map((j) => j.job));
      const ids = new Set(jobs.map((j) => j.id));
      setGlosses((g) => g.filter((x) => !ids.has(x.id)));

      const done = results.filter((r) => r.status === "done");
      if (!done.length) {
        setNotice("Didn't catch any signs. Try again a little slower, with good lighting.");
        return;
      }

      // The background scan started when the hands came up; it's usually done by now.
      const background = autoScene
        ? await Promise.race([autoScene, sleep(AUTO_SCENE_WAIT_MS).then(() => null)])
        : null;
      const sceneNow = mergeScenes(sceneRef.current, background);

      const shown = done.map((r) => (r.target ? `${r.text} (${r.target})` : r.text!));
      let text = done.map((r) => r.text).join(" ");
      let scenePlace = "";
      try {
        const res = await fetch("/api/sentence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            // Send what each sign MEANS (its display text), not its internal key:
            // the STUDY sign is shown as "tutoring center", so it's sent as TUTORING_CENTER.
            glosses: done.map((r) => ({ gloss: (r.text || r.key!).toUpperCase(), target: r.target ?? "" })),
            history: historyRef.current.slice(-HISTORY_SENTENCES),
            ...(sceneNow ? { scene: sceneNow } : {}),
          }),
        });
        const data = await res.json();
        if (res.ok && data.sentence) {
          text = data.sentence;
          if (data.usedScene && sceneNow) scenePlace = sceneNow.place;
        }
      } catch {
        /* keep the plain word list */
      }

      historyRef.current = [...historyRef.current, text].slice(-5);
      setNotice("");
      const marks = done.map((r) => r.markUrl).filter((u): u is string => Boolean(u));
      setSentences((s) => [{ id: ++idRef.current, text, glosses: shown, scenePlace, marks }, ...s]);
      say(text);
    } finally {
      setTranslating((n) => n - 1);
    }
  };

  // --- Segmenter events (called from the camera loop) -----------------------
  handlersRef.current = (ev: SegmenterEvent) => {
    if (ev.type === "recordingStart") {
      setNotice("");
      autoSceneRef.current = captureAutoScene();
    } else if (ev.type === "signStart") {
      startClip(); // normally already running (see clipTickRef)
      capturesRef.current = {};
    } else if (ev.type === "signCancel") {
      stopClip();
      startClip();
      capturesRef.current = {};
    } else if (ev.type === "signEnd") {
      const clip = stopClip();
      startClip(); // the next sign's clip starts right away
      const caps = capturesRef.current;
      capturesRef.current = {};
      const id = ++idRef.current;
      setGlosses((g) => [...g, { id, status: "pending" }]);
      const job = recognize(ev.frames, clip, caps)
        .catch(() => null)
        .then((pick) => {
          const gloss: Gloss = pick
            ? {
                id,
                status: "done",
                key: pick.key,
                text: textFor(pick.key) || pick.key,
                target: pick.target || undefined,
                markUrl: pick.markUrl || undefined,
                via: pick.via,
              }
            : { id, status: "skipped" };
          setGlosses((list) => list.map((x) => (x.id === id ? gloss : x)));
          return gloss;
        });
      pendingRef.current.push({ id, job });
    } else if (ev.type === "sentenceEnd") {
      setProgress(0);
      finishSentence();
    } else if (ev.type === "sentenceCancel") {
      setProgress(0);
      discardSentence();
    }
  };

  // --- Controls --------------------------------------------------------------
  const finishNow = () => {
    unlockSpeech();
    for (const ev of segmenterRef.current.flush()) handlersRef.current(ev);
  };

  const undo = () => {
    const last = pendingRef.current.pop();
    if (!last) return;
    segmenterRef.current.forgetLastSign();
    setGlosses((g) => g.filter((x) => x.id !== last.id));
  };

  // --- Scene scan (back camera, one photo) -------------------------------------
  // Phones can't run both cameras at once, so scanning waits until no recording is in progress.
  const canScan = (phase === "ready" || phase === "listening") && !recording && !sceneLoading;

  const scanScene = async () => {
    if (!canScan) return;
    const resumeTo = phase; // go back to listening or paused afterwards
    unlockSpeech();
    segmenterRef.current.reset();
    stopClip(); // the recorder is tied to the front camera stream we're about to close
    setSceneError("");
    setPhase("scanning");

    let photo: Blob | null = null;
    try {
      await openCamera("environment");
      await sleep(800); // let the camera focus and adjust exposure
      if (videoRef.current) photo = await toJpeg(snapshot(videoRef.current, SCENE_WIDTH), 0.8);
    } catch {
      setSceneError("Couldn't open the back camera.");
    }

    try {
      await openCamera("user");
    } catch {
      setError("Couldn't reopen the front camera. Reload the page.");
      setPhase("error");
      return;
    }
    setPhase(resumeTo);
    if (!photo) return;

    setSceneLoading(true);
    try {
      const form = new FormData();
      form.append("image", photo, "scene.jpg");
      const res = await fetch("/api/scene", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      const scanned = toScene(data);
      if (!scanned) setSceneError("Couldn't make out the surroundings. Try again with more light.");
      else setScene(scanned);
    } catch {
      setSceneError("Scene scan failed. Translation still works without it.");
    } finally {
      setSceneLoading(false);
    }
  };

  // --- UI ----------------------------------------------------------------
  const latest = sentences[0];
  const listening = phase === "listening";
  const mirrored = facing === "user";
  const maxSeconds = RECORDING_MAX_MS / 1000;
  const statusMessage =
    phase === "starting"
      ? "Starting camera."
      : phase === "scanning"
        ? "Scanning surroundings."
        : phase === "error"
          ? error
          : translating > 0
            ? "Putting it into English."
            : recording
              ? "Recording."
              : listening
                ? "Waiting for hands."
                : "Paused.";
  const pill =
    phase === "scanning"
      ? "📷 Scanning surroundings…"
      : recording
        ? `Recording 0:${String(elapsed).padStart(2, "0")} / 0:${maxSeconds}`
        : handsVisible
          ? "✋ Starting…"
          : "Raise your hands to start";

  // The whole app fits one screen: header and cards keep their size, the camera
  // takes whatever height is left (never scrolls).
  return (
    <main
      onPointerDown={unlockSpeech}
      className="mx-auto flex h-[100dvh] max-w-md flex-col overflow-hidden px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-[max(0.5rem,env(safe-area-inset-top))]"
    >
      <header className="flex shrink-0 items-start justify-between gap-2 py-1">
        <div className="min-w-0">
          <h1>
            {/* eslint-disable-next-line @next/next/no-img-element -- static hand-drawn SVG, no optimization needed */}
            <img src="/brand/title.svg" alt="Sign2MeetU" className="h-[4.5rem] w-auto" />
          </h1>
          <p className="text-sm font-medium leading-snug text-[var(--ink)]">Sign and translate!</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setSpeak((s) => !s)}
            className="min-h-11 whitespace-nowrap rounded-full bg-[var(--panel)] px-3 py-1.5 text-sm"
            aria-pressed={speak}
          >
            {speak ? "🔊 Voice on" : "🔇 Voice off"}
          </button>
          <button
            onClick={() => setShowInfo((visible) => !visible)}
            className="min-h-11 min-w-11 rounded-full bg-[var(--panel)] px-3 py-1.5 text-sm text-[var(--muted)]"
            aria-label={showInfo ? "Hide help information" : "Show help information"}
            aria-expanded={showInfo}
            aria-controls="help-info"
            title="How to use Sign2MeetU"
          >
            ⓘ
          </button>
        </div>
      </header>

      <div
        className={`grid shrink-0 transition-[grid-template-rows,opacity] duration-300 ease-in-out ${
          showInfo ? "mb-2 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        }`}
        inert={!showInfo}
        aria-hidden={!showInfo}
      >
        <div className="overflow-hidden">
          <aside
            id="help-info"
            className="max-h-[45dvh] overflow-y-auto rounded-2xl border-2 border-[var(--accent)] bg-[var(--panel)] p-4 text-sm leading-relaxed text-[var(--ink)]"
            aria-label="How to use Sign2MeetU"
          >
            <p>
              Raise your hands into the frame and recording starts on its own. Sign one word at a time and lower your
              hands briefly between signs. When you&apos;re finished, keep your hands down for{" "}
              {SENTENCE_GAP_MS / 1000} seconds (or tap Done) and the app says the whole sentence.
            </p>
            <p className="mt-2">
              Recordings stop automatically after {maxSeconds} seconds. Anything shorter than{" "}
              {RECORDING_MIN_MS / 1000} second is skipped.
            </p>
            <p className="mt-2">
              Scan surroundings takes one photo with the back camera so words like &ldquo;that&rdquo; can be named.
              The photo is used once and not saved.
            </p>
            <p className="mt-2 text-[var(--muted)]">
              Classifier: {classifier.examples} training examples from {classifier.signers} signer
              {classifier.signers === 1 ? "" : "s"}, {classifier.covered}/{classifier.total} signs with{" "}
              {MIN_EXAMPLES_PER_SIGN}+ examples.{" "}
              {classifier.ready
                ? "It can answer when Perceptron is unsure."
                : `Until every sign is covered by ${MIN_SIGNERS}+ signers, it only gives Perceptron hints.`}
            </p>
          </aside>
        </div>
      </div>

      {/* Camera + cards, centered as one group: spare height splits evenly above and below.
          The camera keeps the hand-drawn frame's exact proportions (public/brand/camera-frame.png,
          745×940). It wants full screen width, but shrinks when the screen is too short. */}
      <div className="flex min-h-0 flex-1 flex-col justify-center" style={{ containerType: "inline-size" }}>
      <div
        className="flex min-h-0 shrink items-center justify-center"
        style={{ height: "calc(100cqw * 940 / 745)", containerType: "size" }}
      >
        <div
          className="relative"
          style={{ width: "min(100cqw, calc(100cqh * 745 / 940))", aspectRatio: "745 / 940" }}
        >
          <section
            className="absolute bottom-[5.2%] left-[5.1%] right-[5.9%] top-[4.7%] overflow-hidden rounded-2xl bg-[var(--panel)] shadow-none"
            aria-label="Camera preview"
          >
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              className={`h-full w-full object-cover ${mirrored ? "-scale-x-100" : ""}`}
              aria-hidden="true"
            />
            <canvas
              ref={canvasRef}
              className={`pointer-events-none absolute inset-0 h-full w-full object-cover ${mirrored ? "-scale-x-100" : ""}`}
              aria-hidden="true"
            />

            {phase !== "starting" && phase !== "error" && (
              <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
                {recording && <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[var(--rec)]" />}
                {pill}
              </div>
            )}

            {recording && (
              <>
                <div className="pointer-events-none absolute inset-0 rounded-2xl ring-4 ring-inset ring-[var(--rec)]" />
                <button
                  onClick={finishNow}
                  className="absolute right-3 top-3 z-10 min-h-11 rounded-full bg-[var(--ink)] px-4 text-sm font-semibold text-white"
                >
                  Done
                </button>
              </>
            )}

            {recording && progress > 0 && (
              <div className="absolute inset-x-0 bottom-4 px-4">
                <p className="mb-1 text-center text-sm font-medium text-white drop-shadow">Stopping…</p>
                <div className="h-1.5 rounded bg-black/40">
                  <div className="h-full rounded bg-[var(--rec)]" style={{ width: `${progress * 100}%` }} />
                </div>
              </div>
            )}

            {phase === "scanning" && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-base text-white">
                Hold still. Taking one photo…
              </div>
            )}

            {phase === "starting" && (
              <div className="absolute inset-0 flex items-center justify-center bg-[var(--panel)]/95 text-[var(--muted)]">
                Starting camera…
              </div>
            )}

            {phase === "error" && (
              <div className="absolute inset-0 flex items-center justify-center bg-[var(--panel)]/95 p-6 text-center text-[var(--muted)]">
                {error}
              </div>
            )}
          </section>
          {/* eslint-disable-next-line @next/next/no-img-element -- decorative overlay, must match the box exactly */}
          <img
            src="/brand/camera-frame.png"
            alt=""
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 z-20 h-full w-full select-none"
          />
        </div>
      </div>

      {/* Surroundings card */}
      <section
        className="mt-1 flex shrink-0 items-center gap-3 rounded-3xl bg-[var(--panel)] p-3"
        aria-label="Surroundings"
      >
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--bg)]/60 text-[var(--ink)]">
          <CameraIcon className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold leading-tight text-[var(--ink)]">
            {scene ? `📍 ${scene.place}` : "No surroundings scanned"}
          </p>
          <p className="mt-0.5 truncate text-xs leading-snug text-[var(--muted)]">
            {sceneError ||
              (scene
                ? [scene.objects.slice(0, 3).join(", "), scene.text.length ? `“${scene.text[0]}”` : ""]
                    .filter(Boolean)
                    .join(" · ") || "Nothing specific spotted"
                : recording
                  ? "Finish signing to scan."
                  : "Uses the back camera. Photo is used once and not saved.")}
          </p>
        </div>
        {scene && (
          <button
            onClick={() => setScene(null)}
            className="min-h-11 min-w-11 shrink-0 rounded-full text-[var(--muted)]"
            aria-label="Clear surroundings"
          >
            ✕
          </button>
        )}
        <div className="h-10 w-px shrink-0 bg-[var(--muted)]/25" aria-hidden="true" />
        <button
          onClick={scanScene}
          disabled={!canScan}
          className="flex min-h-12 shrink-0 items-center gap-2 rounded-2xl bg-[var(--accent)] px-3 text-base text-white disabled:opacity-50"
        >
          <CameraIcon className="h-5 w-5" />
          {sceneLoading ? "Reading…" : scene ? "Rescan" : "Scan"}
        </button>
      </section>

      {/* Translation card: signs in progress while signing, then the finished sentence */}
      <section
        id="result"
        className="mt-2 flex min-h-[5.5rem] shrink-0 items-center gap-3 rounded-3xl bg-[var(--accent)] p-4 text-white"
        aria-label="Translation"
        aria-live="polite"
        aria-atomic="true"
      >
        <div className="min-w-0 flex-1">
          {glosses.length > 0 ? (
            <ul className="flex flex-wrap gap-1.5" aria-label="Signs so far">
              {glosses.map((g) => (
                <li
                  key={g.id}
                  className={`rounded-full px-3 py-0.5 text-lg ${g.status === "done" ? "bg-white/20" : "bg-white/10"}`}
                  title={g.via ? `Recognized by ${g.via}` : undefined}
                >
                  {g.status === "pending" ? "…" : g.status === "skipped" ? "?" : g.text}
                  {g.target && <span className="text-white/70"> · {g.target}</span>}
                </li>
              ))}
            </ul>
          ) : translating > 0 ? (
            <p className="text-2xl leading-tight">Putting it into English…</p>
          ) : notice ? (
            <p className="text-base leading-snug">{notice}</p>
          ) : latest ? (
            <>
              <p className="line-clamp-2 text-2xl font-bold leading-tight">{latest.text}</p>
              <div className="mt-0.5 flex items-center gap-2">
                <p className="min-w-0 truncate text-sm text-white/70">
                  Signed: {latest.glosses.join(" · ")}
                  {latest.scenePlace && ` · 📍 ${latest.scenePlace}`}
                </p>
                {latest.marks.map((url) => (
                  // eslint-disable-next-line @next/next/no-img-element -- local blob URL, not an optimizable asset
                  <img
                    key={url}
                    src={url}
                    alt="Camera still with the pointing direction marked in green"
                    className="h-8 shrink-0 -scale-x-100 rounded-md object-cover"
                  />
                ))}
              </div>
            </>
          ) : (
            <p className="text-base leading-snug">
              {recording
                ? "Sign one word at a time. Lower your hands between signs."
                : `Raise your hands to start. Keep them down for ${SENTENCE_GAP_MS / 1000} seconds when you're done.`}
            </p>
          )}
        </div>
        {glosses.length > 0 ? (
          <button
            onClick={undo}
            className="flex min-h-12 shrink-0 items-center gap-2 rounded-2xl bg-white/15 px-3 text-base text-white"
          >
            <ReplayIcon className="h-5 w-5" />
            Undo
          </button>
        ) : (
          latest &&
          translating === 0 && (
            <button
              onClick={() => say(latest.text)}
              className="flex min-h-12 shrink-0 items-center gap-2 rounded-2xl bg-white/15 px-3 text-base text-white"
            >
              <ReplayIcon className="h-5 w-5" />
              Replay
            </button>
          )
        )}
      </section>
      </div>

      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {statusMessage}
      </p>
    </main>
  );
}
