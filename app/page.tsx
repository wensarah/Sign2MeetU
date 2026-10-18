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
  const toggleListening = () => {
    unlockSpeech();
    if (phase === "listening") {
      for (const ev of segmenterRef.current.flush()) handlersRef.current(ev); // keep what was signed
      segmenterRef.current.reset();
      stopClip();
      setRecording(false);
      setProgress(0);
      setPhase("ready"); // paused
    } else if (phase === "ready") {
      segmenterRef.current.reset();
      setNotice("");
      setPhase("listening");
    }
  };

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

  const clearAll = () => {
    pendingRef.current = [];
    autoSceneRef.current = null;
    historyRef.current = [];
    segmenterRef.current.reset();
    for (const s of sentences) s.marks.forEach((u) => URL.revokeObjectURL(u));
    setGlosses([]);
    setSentences([]);
    setNotice("");
    setProgress(0);
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
      : !listening
        ? "Paused"
        : recording
          ? `Recording 0:${String(elapsed).padStart(2, "0")} / 0:${maxSeconds}`
          : handsVisible
            ? "✋ Starting…"
            : "Raise your hands to start";

  return (
    <main
      onPointerDown={unlockSpeech}
      className="mx-auto flex min-h-[100dvh] max-w-md flex-col px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-[max(0.75rem,env(safe-area-inset-top))]"
    >
      <header className="flex items-start justify-between gap-2 py-2">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight">
            Sign2<span className="text-[var(--accent)]">MeetU</span>
          </h1>
          <p className="text-sm font-medium leading-snug text-[var(--ink)]">
            Sign and translate!
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setSpeak((s) => !s)}
            className="min-h-11 rounded-full bg-[var(--panel)] px-3 py-1.5 text-sm"
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
        className={`grid transition-[grid-template-rows,opacity] duration-300 ease-in-out ${
          showInfo ? "mb-2 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        }`}
        inert={!showInfo}
        aria-hidden={!showInfo}
      >
        <div className="overflow-hidden">
          <aside
            id="help-info"
            className="rounded-2xl border-2 border-[var(--accent)] bg-[var(--panel)] p-4 text-sm leading-relaxed text-[var(--ink)]"
            aria-label="How to use Sign2MeetU"
          >
            <p>
              Raise your hands into the frame and recording starts on its own. Sign one word at a time and lower your
              hands briefly between signs. When you&apos;re finished, keep your hands down for{" "}
              {SENTENCE_GAP_MS / 1000} seconds (or tap Done) and the app says the whole sentence.
            </p>
            <p className="mt-2">
              Recordings stop automatically after {maxSeconds} seconds. Anything shorter than{" "}
              {RECORDING_MIN_MS / 1000} second is skipped. The round button pauses and resumes listening.
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

      {/* Camera */}
      <div className="relative mt-2 aspect-[3/4] w-full">
        <svg className="absolute h-0 w-0" aria-hidden="true" focusable="false">
          <defs>
            <clipPath id="camera-bubble-clip" clipPathUnits="objectBoundingBox">
              <path d="M .08 0 H .92 Q 1 0 1 .08 V .86 Q 1 .94 .92 .94 H .88 L 1 1 L .76 .94 H .08 Q 0 .94 0 .86 V .08 Q 0 0 .08 0 Z" />
            </clipPath>
          </defs>
        </svg>
        <section
          className="camera-bubble-frame absolute inset-0 bg-[var(--panel)]"
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

          {recording && <div className="absolute inset-0 rounded-3xl ring-4 ring-inset ring-[var(--rec)]" />}

          {recording && progress > 0 && (
            <div className="absolute inset-x-0 bottom-[7%] px-4">
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
        <svg
          className="pointer-events-none absolute inset-0 z-20 h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
          focusable="false"
        >
          <path
            d="M 8 0 H 92 Q 100 0 100 8 V 86 Q 100 94 92 94 H 88 L 100 100 L 76 94 H 8 Q 0 94 0 86 V 8 Q 0 0 8 0 Z"
            fill="none"
            stroke="var(--accent)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>

      {/* Scene */}
      <section className="mt-3 rounded-2xl bg-[var(--panel)] p-3 text-sm" aria-label="Surroundings">
        <div className="flex items-center gap-2">
          {scene ? (
            <p className="min-w-0 flex-1 truncate font-medium">
              📍 {scene.place}
              {scene.objects.length > 0 && <span className="text-[var(--muted)]"> · {scene.objects.slice(0, 4).join(", ")}</span>}
              {scene.text.length > 0 && <span className="text-[var(--muted)]"> · “{scene.text.slice(0, 2).join("”, “")}”</span>}
            </p>
          ) : (
            <p className="min-w-0 flex-1 text-[var(--muted)]">No surroundings scanned</p>
          )}
          {scene && (
            <button
              onClick={() => setScene(null)}
              className="min-h-11 min-w-11 rounded-full bg-[var(--bg)] px-3 text-sm"
              aria-label="Clear surroundings"
            >
              ✕
            </button>
          )}
          <button
            onClick={scanScene}
            disabled={!canScan}
            className="min-h-11 shrink-0 rounded-full bg-[var(--ink)] px-3 text-sm text-white disabled:opacity-40"
          >
            {sceneLoading ? "Reading…" : scene ? "Rescan" : "Scan surroundings"}
          </button>
        </div>
        <p className="mt-1 text-xs text-[var(--muted)]">
          {sceneError ||
            (recording ? "Finish signing to scan." : "Uses the back camera. Photo is used once and not saved.")}
        </p>
      </section>

      {/* Result */}
      <section
        id="result"
        className="mt-3 mb-2 min-h-[5.5rem] rounded-2xl bg-[var(--ink)] p-4 text-white"
        aria-label="Translation"
        aria-live="polite"
        aria-atomic="true"
      >
        {translating > 0 ? (
          <p className="text-white">Putting it into English…</p>
        ) : notice ? (
          <p className="text-white">{notice}</p>
        ) : latest ? (
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-3xl font-bold leading-tight">{latest.text}</p>
              <p className="mt-1 text-xs text-white/80">
                Signed: {latest.glosses.join(" · ")}
                {latest.scenePlace && ` · 📍 ${latest.scenePlace}`}
              </p>
              {latest.marks.length > 0 && (
                <div className="mt-2 flex gap-2" aria-label="What Perceptron looked at for pointing signs">
                  {latest.marks.map((url) => (
                    // eslint-disable-next-line @next/next/no-img-element -- local blob URL, not an optimizable asset
                    <img
                      key={url}
                      src={url}
                      alt="Camera still with the pointing direction marked in green"
                      className="h-16 -scale-x-100 rounded-lg object-cover"
                    />
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={() => say(latest.text)}
              className="min-h-11 shrink-0 rounded-full bg-white/15 px-3 py-1.5 text-sm text-white"
            >
              Replay
            </button>
          </div>
        ) : (
          <p className="text-white">
            {!listening
              ? "Paused. Tap the round button to resume."
              : recording
                ? "Sign one word at a time. Lower your hands between signs."
                : `Raise your hands to start signing. Keep them down for ${SENTENCE_GAP_MS / 1000} seconds when you're done.`}
          </p>
        )}
      </section>

      {/* Transcript: signs in progress, then earlier sentences */}
      <section
        className="mt-3 mb-5 flex min-h-0 flex-1 items-start gap-2 rounded-2xl bg-[#905fc1] p-4 text-white"
        aria-label="Transcript"
      >
        <div className="min-w-0 flex-1 overflow-y-auto">
          {glosses.length > 0 ? (
            <ul className="flex flex-wrap gap-1.5" aria-label="Signs so far">
              {glosses.map((g) => (
                <li
                  key={g.id}
                  className={`rounded-full px-3 py-1 text-base ${
                    g.status === "done" ? "bg-[var(--ink)]" : "bg-white/20"
                  }`}
                  title={g.via ? `Recognized by ${g.via}` : undefined}
                >
                  {g.status === "pending" ? "…" : g.status === "skipped" ? "?" : g.text}
                  {g.target && <span className="text-white/75"> · {g.target}</span>}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-lg leading-snug text-white">
              {sentences.length > 1 ? "Earlier:" : "Signs appear here as you make them."}
            </p>
          )}
          {sentences.length > 1 && (
            <ol className="mt-2 space-y-1 text-base leading-snug text-white/90">
              {sentences.slice(1).map((s) => (
                <li key={s.id}>{s.text}</li>
              ))}
            </ol>
          )}
        </div>
        {(glosses.length > 0 || sentences.length > 0) && (
          <div className="flex shrink-0 flex-col gap-2">
            {glosses.length > 0 && (
              <button onClick={undo} className="min-h-11 rounded-full bg-[var(--ink)] px-3 py-1.5 text-sm text-white">
                Undo
              </button>
            )}
            <button onClick={clearAll} className="min-h-11 rounded-full bg-[var(--ink)] px-3 py-1.5 text-sm text-white">
              Clear
            </button>
          </div>
        )}
      </section>

      {/* Controls */}
      <div className="relative flex items-center justify-center pt-3">
        <button
          onClick={toggleListening}
          disabled={phase !== "ready" && phase !== "listening"}
          aria-label={listening ? "Pause listening" : "Resume listening"}
          aria-pressed={!listening}
          className="flex h-20 w-20 items-center justify-center rounded-full border-4 border-[var(--ink)] transition active:scale-95 disabled:opacity-40"
        >
          <span
            className={`block bg-[var(--rec)] transition-all ${
              listening ? "h-7 w-7 rounded-md" : "h-14 w-14 rounded-full"
            }`}
          />
        </button>
        {recording && (
          <button
            onClick={finishNow}
            className="absolute right-0 min-h-11 rounded-full bg-[var(--ink)] px-4 py-2 text-sm font-semibold text-white"
          >
            Done
          </button>
        )}
      </div>
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {statusMessage}
      </p>
    </main>
  );
}
