"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const CLIP_MS = 2500;
const MIN_CONFIDENCE = 0.5; // below this we show the guess but don't speak it

type Phase = "starting" | "ready" | "recording" | "thinking" | "error";

type Result = {
  sign: string;
  text: string;
  confidence: number;
};

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

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const speechUnlocked = useRef(false);

  const [phase, setPhase] = useState<Phase>("starting");
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [speak, setSpeak] = useState(true);
  const [showInfo, setShowInfo] = useState(false);

  // --- Camera ------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

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
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 24, max: 30 },
          },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setPhase("ready");
      } catch (e) {
        const name = e instanceof DOMException ? e.name : "";
        setError(
          name === "NotAllowedError"
            ? "Camera permission was denied. Allow camera access in your browser settings and reload."
            : "Couldn't open the camera. Is another app using it?"
        );
        setPhase("error");
      }
    }

    start();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

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

  // --- Record + translate ------------------------------------------------
  const recordClip = (): Promise<Blob> =>
    new Promise((resolve, reject) => {
      const stream = streamRef.current;
      if (!stream) return reject(new Error("No camera stream"));

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
        videoBitsPerSecond: 600_000, // keeps a 2.5s clip well under 1 MB
      });
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onerror = () => reject(new Error("Recording failed"));
      recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || mimeType || "video/webm" }));
      recorder.start();
      setTimeout(() => recorder.state !== "inactive" && recorder.stop(), CLIP_MS);
    });

  const handleRecord = async () => {
    if (phase !== "ready") return;
    unlockSpeech();
    setResult(null);
    setPhase("recording");

    try {
      const blob = await recordClip();
      setPhase("thinking");

      const ext = blob.type.includes("mp4") ? "mp4" : "webm";
      const form = new FormData();
      form.append("video", blob, `clip.${ext}`);

      const res = await fetch("/api/translate", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

      const r = data as Result;
      setResult(r);
      if (r.sign !== "UNKNOWN" && r.confidence >= MIN_CONFIDENCE) {
        setTranscript((t) => [...t, r.text]);
        say(r.text);
      }
    } catch (e) {
      setResult({ sign: "ERROR", text: e instanceof Error ? e.message : "Something went wrong", confidence: 0 });
    } finally {
      setPhase("ready");
    }
  };

  // --- UI ----------------------------------------------------------------
  const sentence = transcript.join(" ");
  const statusMessage =
    phase === "starting"
      ? "Starting camera."
      : phase === "recording"
        ? "Recording. Sign now."
        : phase === "thinking"
          ? "Reading your sign."
          : phase === "error"
            ? error
            : "";

  return (
    <main className="mx-auto flex min-h-[100dvh] max-w-md flex-col px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-[max(0.75rem,env(safe-area-inset-top))]">
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
              Tap the round button to record one sign. Keep your hands in the camera frame. Recognized words appear in
              the result box and build a sentence below.
            </p>
            <p className="mt-2">
              The confidence rate estimates how sure the app is about a translation. Turn Voice on to hear recognized
              words spoken aloud.
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
            className="h-full w-full -scale-x-100 object-cover"
            aria-hidden="true"
          />

          {phase === "recording" && (
            <>
              <div className="absolute inset-0 rounded-3xl ring-4 ring-inset ring-[var(--rec)]" />
              <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
                <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[var(--rec)]" /> Sign now
              </div>
              <div className="absolute inset-x-0 bottom-0 h-1.5 bg-black/40">
                <div
                  className="h-full origin-left bg-[var(--rec)]"
                  style={{ animation: `fill ${CLIP_MS}ms linear forwards` }}
                />
              </div>
            </>
          )}

          {phase === "thinking" && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-base text-white">
              Reading your sign…
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

      {/* Result */}
      <section
        id="result"
        className="mt-4 mb-2 min-h-[5.5rem] rounded-2xl bg-[var(--ink)] p-4 text-white"
        aria-label="Sign result"
        aria-live="polite"
        aria-atomic="true"
      >
        {!result && phase !== "thinking" && (
          <p className="text-white">Tap the button, then sign one word. Keep both hands in frame.</p>
        )}
        {result && result.sign === "ERROR" && <p className="text-white">{result.text}</p>}
        {result && result.sign === "UNKNOWN" && (
          <p className="text-white">Didn't catch that. Try again a little slower, with good lighting.</p>
        )}
        {result && result.sign !== "UNKNOWN" && result.sign !== "ERROR" && (
          <div>
            <div className="flex items-end justify-between gap-3">
              <p className="text-4xl font-bold leading-tight">{result.text}</p>
              <div className="shrink-0 text-right">
                <p className="text-sm text-white">
                  {Math.round(result.confidence * 100)}%{result.confidence < MIN_CONFIDENCE && " · unsure"}
                </p>
                <p className="text-xs text-white">Confidence rate</p>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Transcript */}
      <section
        className="mt-3 mb-5 flex min-h-0 flex-1 items-start gap-2 rounded-2xl bg-[#905fc1] p-4 text-white"
        aria-label="Transcript"
      >
        <p className="flex-1 overflow-y-auto text-lg leading-snug text-white">
          {sentence || "Your sentence builds up here."}
        </p>
        {transcript.length > 0 && (
          <div className="flex shrink-0 gap-2">
            <button onClick={() => say(sentence)} className="min-h-11 rounded-full bg-[var(--ink)] px-3 py-1.5 text-sm text-white">
              Replay
            </button>
            <button onClick={() => setTranscript([])} className="min-h-11 rounded-full bg-[var(--ink)] px-3 py-1.5 text-sm text-white">
              Clear
            </button>
          </div>
        )}
      </section>

      {/* Record button */}
      <div className="flex justify-center pt-3">
        <button
          onClick={handleRecord}
          disabled={phase !== "ready"}
          aria-label={phase === "recording" ? "Recording sign" : "Record a sign"}
          className="flex h-20 w-20 items-center justify-center rounded-full border-4 border-[var(--ink)] transition active:scale-95 disabled:opacity-40"
        >
          <span
            className={`block bg-[var(--rec)] transition-all ${
              phase === "recording" ? "h-7 w-7 rounded-md" : "h-14 w-14 rounded-full"
            }`}
          />
        </button>
      </div>
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {statusMessage}
      </p>
    </main>
  );
}
