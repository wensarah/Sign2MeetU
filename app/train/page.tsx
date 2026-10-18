"use client";

// /train — record labeled sign examples, test the kNN classifier, and export the dataset.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { VOCAB } from "@/lib/vocab";
import {
  loadHandLandmarker,
  frameFeatures,
  makeFingerprint,
  drawHands,
  FEATURE_VERSION,
  type Frame,
  type Fingerprint,
} from "@/lib/landmarks";
import {
  predict,
  evaluate,
  emptyDataset,
  mergeDatasets,
  newId,
  type Dataset,
  type Prediction,
  type EvalReport,
} from "@/lib/knn";
 
const CLIP_MS = 2500;
const COUNTDOWN = 3;
const STORAGE_KEY = "sign2meetu-dataset";
const NAME_KEY = "sign2meetu-signer";
const TARGET_PER_SIGN = 5;
 
type Phase = "loading" | "ready" | "countdown" | "recording" | "error";
type Mode = "record" | "test";
 
export default function TrainPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const latestResult = useRef<HandLandmarkerResult | null>(null);
  const capture = useRef<(Frame | null)[] | null>(null); // non-null while recording
 
  const [phase, setPhase] = useState<Phase>("loading");
  const [error, setError] = useState("");
  const [mode, setMode] = useState<Mode>("record");
  const [count, setCount] = useState(0);
  const [handsVisible, setHandsVisible] = useState(false);
 
  const [dataset, setDataset] = useState<Dataset>(emptyDataset);
  const [storageWarning, setStorageWarning] = useState(false);
  const [signer, setSigner] = useState("");
  const [selected, setSelected] = useState(VOCAB[0]?.key ?? "");
  const [message, setMessage] = useState("");
 
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [report, setReport] = useState<EvalReport | null>(null);
 
  // ---------- Load saved data ----------
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setDataset(mergeDatasets(JSON.parse(saved)));
      setSigner(localStorage.getItem(NAME_KEY) ?? "");
    } catch {
      /* private browsing etc. — start empty */
    }
  }, []);
 
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
      setStorageWarning(false);
    } catch {
      setStorageWarning(true); // storage full: user must download
    }
  }, [dataset]);
 
  useEffect(() => {
    try {
      localStorage.setItem(NAME_KEY, signer);
    } catch {}
  }, [signer]);
 
  // ---------- Camera + MediaPipe ----------
  useEffect(() => {
    let stream: MediaStream | null = null;
    let raf = 0;
    let lastTime = -1;
    let cancelled = false;
 
    async function start() {
      try {
        if (!window.isSecureContext) throw new Error("Camera needs HTTPS (use the Vercel URL or ngrok).");
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        });
        if (cancelled) return;
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play().catch(() => {});
 
        landmarkerRef.current = await loadHandLandmarker();
        if (cancelled) return;
        setPhase("ready");
 
        const loop = () => {
          const v = videoRef.current;
          const lm = landmarkerRef.current;
          if (v && lm && v.readyState >= 2 && v.currentTime !== lastTime) {
            lastTime = v.currentTime;
            const result = lm.detectForVideo(v, performance.now());
            latestResult.current = result;
            setHandsVisible(result.landmarks.length > 0);
 
            if (capture.current) capture.current.push(frameFeatures(result));
 
            const c = canvasRef.current;
            if (c) {
              if (c.width !== v.videoWidth) {
                c.width = v.videoWidth;
                c.height = v.videoHeight;
              }
              drawHands(c.getContext("2d")!, result);
            }
          }
          raf = requestAnimationFrame(loop);
        };
        loop();
      } catch (e) {
        setError(
          e instanceof DOMException && e.name === "NotAllowedError"
            ? "Camera permission denied. Allow camera access and reload."
            : e instanceof Error
              ? e.message
              : "Could not start camera or hand tracker."
        );
        setPhase("error");
      }
    }
 
    start();
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);
 
  // ---------- Record one fingerprint ----------
  const recordFingerprint = useCallback(async (): Promise<Fingerprint | null> => {
    setPhase("countdown");
    for (let i = COUNTDOWN; i > 0; i--) {
      setCount(i);
      await sleep(600);
    }
    setPhase("recording");
    capture.current = [];
    await sleep(CLIP_MS);
    const frames = capture.current;
    capture.current = null;
    setPhase("ready");
    return makeFingerprint(frames);
  }, []);
 
  const handleRecordExample = async () => {
    if (!signer.trim()) {
      setMessage("Enter your name first.");
      return;
    }
    setMessage("");
    const seq = await recordFingerprint();
    if (!seq) {
      setMessage("No hands detected. Keep both hands in frame and try again.");
      return;
    }
    setDataset((d) => ({
      ...d,
      examples: [...d.examples, { id: newId(), label: selected, signer: signer.trim(), seq }],
    }));
    const n = dataset.examples.filter((e) => e.label === selected && e.signer === signer.trim()).length + 1;
    setMessage(`Saved ${selected} #${n}`);
  };
 
  const handleTry = async () => {
    setPrediction(null);
    setMessage("");
    const seq = await recordFingerprint();
    if (!seq) {
      setMessage("No hands detected. Try again.");
      return;
    }
    try {
      setPrediction(predict(seq, dataset.examples));
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    }
  };
 
  const handleEvaluate = () => {
    setReport(null);
    setMessage("");
    if (dataset.examples.length < 2) {
      setMessage("Record some examples first.");
      return;
    }
    try {
      const t0 = performance.now();
      const r = evaluate(dataset.examples);
      setReport(r);
      setMessage(`Tested ${r.total} examples in ${Math.round(performance.now() - t0)} ms.`);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    }
  };
 
  // ---------- Dataset import / export ----------
  const download = () => {
    const name = (signer.trim() || "dataset").toLowerCase().replace(/\W+/g, "-");
    const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `dataset-${name}-${new Date().toISOString().slice(11, 16).replace(":", "")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
 
  const importFile = async (file: File) => {
    try {
      const incoming = JSON.parse(await file.text()) as Dataset;
      if (incoming.version !== FEATURE_VERSION) {
        setMessage(`That file uses feature version ${incoming.version}, this app expects ${FEATURE_VERSION}.`);
        return;
      }
      setDataset((d) => mergeDatasets(d, incoming));
      setMessage(`Imported ${incoming.examples.length} examples.`);
    } catch {
      setMessage("Couldn't read that file.");
    }
  };
 
  const loadTeamDataset = async () => {
    try {
      const res = await fetch("/dataset.json", { cache: "no-store" });
      if (!res.ok) throw new Error();
      const team = (await res.json()) as Dataset;
      setDataset((d) => mergeDatasets(d, team));
      setMessage(`Loaded ${team.examples.length} examples from the team dataset.`);
    } catch {
      setMessage("No public/dataset.json deployed yet.");
    }
  };
 
  const undo = () => {
    const mine = dataset.examples.filter((e) => e.signer === signer.trim());
    const last = mine[mine.length - 1];
    if (!last) return;
    setDataset((d) => ({ ...d, examples: d.examples.filter((e) => e.id !== last.id) }));
    setMessage(`Removed last ${last.label}.`);
  };
 
  const clearAll = () => {
    if (confirm("Delete ALL examples on this device? Download first if you need them.")) {
      setDataset(emptyDataset());
      setReport(null);
      setMessage("Cleared.");
    }
  };
 
  // ---------- Derived ----------
  const counts = useMemo(() => {
    const mine: Record<string, number> = {};
    const all: Record<string, number> = {};
    for (const e of dataset.examples) {
      all[e.label] = (all[e.label] ?? 0) + 1;
      if (e.signer === signer.trim()) mine[e.label] = (mine[e.label] ?? 0) + 1;
    }
    return { mine, all };
  }, [dataset, signer]);
 
  const signers = useMemo(() => [...new Set(dataset.examples.map((e) => e.signer))], [dataset]);
  const busy = phase !== "ready";
  const myDone = VOCAB.filter((s) => (counts.mine[s.key] ?? 0) >= TARGET_PER_SIGN).length;
 
  // ---------- UI ----------
  return (
    <main className="mx-auto max-w-md px-4 pb-10 pt-4">
      <header className="mb-3 flex items-baseline justify-between">
        <h1 className="text-lg font-bold">
          Sign2<span className="text-[var(--accent)]">MeetU</span> <span className="text-[var(--muted)]">· Train</span>
        </h1>
        <a href="/" className="text-sm text-[var(--muted)] underline">
          Back to app
        </a>
      </header>
 
      {/* Camera */}
      <section className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl bg-[var(--panel)]">
        <video ref={videoRef} playsInline muted autoPlay className="h-full w-full -scale-x-100 object-cover" />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover" />
 
        <div className="absolute left-2 top-2 rounded-full bg-black/60 px-2.5 py-1 text-xs">
          {phase === "loading" ? "Loading hand tracker…" : handsVisible ? "✋ Hands detected" : "No hands"}
        </div>
        {phase === "countdown" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-7xl font-bold">{count}</div>
        )}
        {phase === "recording" && (
          <>
            <div className="absolute inset-0 rounded-2xl ring-4 ring-inset ring-[var(--rec)]" />
            <div className="absolute right-2 top-2 rounded-full bg-[var(--rec)] px-2.5 py-1 text-xs font-semibold">
              Sign now
            </div>
          </>
        )}
        {phase === "error" && (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-[var(--muted)]">
            {error}
          </div>
        )}
      </section>
 
      {/* Mode tabs */}
      <div className="mt-3 grid grid-cols-2 gap-1 rounded-full bg-[var(--panel)] p-1 text-sm">
        {(["record", "test"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-full py-1.5 ${mode === m ? "bg-[var(--ink)] font-semibold text-[var(--bg)]" : ""}`}
          >
            {m === "record" ? "Record examples" : "Test classifier"}
          </button>
        ))}
      </div>
 
      {message && <p className="mt-3 text-sm text-[var(--accent)]">{message}</p>}
      {storageWarning && (
        <p className="mt-2 text-sm text-[var(--rec)]">
          Browser storage is full. Download your dataset now so nothing is lost.
        </p>
      )}
 
      {mode === "record" ? (
        <>
          <label className="mt-3 block text-sm text-[var(--muted)]">
            Your name
            <input
              value={signer}
              onChange={(e) => setSigner(e.target.value)}
              placeholder="e.g. alex"
              className="mt-1 w-full rounded-xl bg-[var(--panel)] px-3 py-2 text-[var(--ink)] outline-none"
            />
          </label>
 
          <p className="mt-3 text-sm text-[var(--muted)]">
            Pick a sign · {myDone}/{VOCAB.length} done ({TARGET_PER_SIGN} each)
          </p>
          <div className="mt-1 grid grid-cols-3 gap-1.5">
            {VOCAB.map((s) => {
              const n = counts.mine[s.key] ?? 0;
              return (
                <button
                  key={s.key}
                  onClick={() => setSelected(s.key)}
                  className={`rounded-xl px-2 py-2 text-left text-sm ${
                    selected === s.key ? "bg-[var(--accent)] text-[var(--bg)]" : "bg-[var(--panel)]"
                  }`}
                >
                  <span className="block truncate font-medium">{s.text}</span>
                  <span className="text-xs opacity-70">
                    {n >= TARGET_PER_SIGN ? "✓ " : ""}
                    {n}/{TARGET_PER_SIGN}
                  </span>
                </button>
              );
            })}
          </div>
 
          <div className="mt-4 flex gap-2">
            <button
              onClick={handleRecordExample}
              disabled={busy}
              className="flex-1 rounded-full bg-[var(--rec)] py-3 font-semibold disabled:opacity-40"
            >
              Record “{VOCAB.find((s) => s.key === selected)?.text}”
            </button>
            <button onClick={undo} disabled={busy} className="rounded-full bg-[var(--panel)] px-4 text-sm disabled:opacity-40">
              Undo
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="mt-3 flex gap-2">
            <button
              onClick={handleTry}
              disabled={busy}
              className="flex-1 rounded-full bg-[var(--rec)] py-3 font-semibold disabled:opacity-40"
            >
              Try a sign
            </button>
            <button
              onClick={handleEvaluate}
              disabled={busy}
              className="flex-1 rounded-full bg-[var(--panel)] py-3 text-sm disabled:opacity-40"
            >
              Run accuracy test
            </button>
          </div>
 
          {prediction && (
            <div className="mt-3 rounded-2xl bg-[var(--panel)] p-4">
              <p className="text-3xl font-bold">
                {VOCAB.find((s) => s.key === prediction.label)?.text ?? prediction.label}
              </p>
              <p className="text-sm text-[var(--muted)]">{Math.round(prediction.confidence * 100)}% of the vote</p>
              <ol className="mt-2 space-y-0.5 text-sm">
                {prediction.top.map((t) => (
                  <li key={t.label} className="flex justify-between">
                    <span>{t.label}</span>
                    <span className="text-[var(--muted)]">{Math.round(t.score * 100)}%</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
 
          {report && (
            <div className="mt-3 space-y-3 rounded-2xl bg-[var(--panel)] p-4 text-sm">
              <div>
                <p className="text-3xl font-bold">{(report.overall * 100).toFixed(1)}%</p>
                <p className="text-[var(--muted)]">
                  {signers.length > 1
                    ? `Leave-one-signer-out, ${report.total} examples, ${signers.length} signers`
                    : `Leave-one-out, ${report.total} examples (add a 2nd signer for an honest number)`}
                </p>
              </div>
              {report.bySigner.length > 1 && (
                <Table title="Tested on" rows={report.bySigner.map((r) => [r.signer, r.accuracy, r.n])} />
              )}
              <Table title="By sign (worst first)" rows={report.bySign.map((r) => [r.label, r.accuracy, r.n])} />
              {report.confusions.length > 0 && (
                <div>
                  <p className="mb-1 font-semibold">Most common mistakes</p>
                  {report.confusions.map((c) => (
                    <p key={c.expected + c.got} className="text-[var(--muted)]">
                      {c.count}× {c.expected} → {c.got}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
 
      {/* Dataset */}
      <section className="mt-6 rounded-2xl bg-[var(--panel)] p-4 text-sm">
        <p className="font-semibold">Dataset on this device</p>
        <p className="text-[var(--muted)]">
          {dataset.examples.length} examples · {signers.length} signer{signers.length === 1 ? "" : "s"}
          {signers.length > 0 && ` (${signers.join(", ")})`}
        </p>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <button onClick={download} className="rounded-full bg-[var(--ink)] py-2 font-semibold text-[var(--bg)]">
            Download
          </button>
          <label className="cursor-pointer rounded-full bg-[var(--bg)] py-2 text-center">
            Import file
            <input
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])}
            />
          </label>
          <button onClick={loadTeamDataset} className="rounded-full bg-[var(--bg)] py-2">
            Load team dataset
          </button>
          <button onClick={clearAll} className="rounded-full bg-[var(--bg)] py-2 text-[var(--rec)]">
            Clear all
          </button>
        </div>
      </section>
    </main>
  );
}
 
function Table({ title, rows }: { title: string; rows: [string, number, number][] }) {
  return (
    <div>
      <p className="mb-1 font-semibold">{title}</p>
      {rows.map(([name, acc, n]) => (
        <div key={name} className="flex items-center gap-2">
          <span className="w-24 truncate">{name}</span>
          <div className="h-1.5 flex-1 rounded bg-[var(--bg)]">
            <div
              className={`h-full rounded ${acc >= 0.7 ? "bg-[var(--accent)]" : "bg-[var(--rec)]"}`}
              style={{ width: `${acc * 100}%` }}
            />
          </div>
          <span className="w-16 text-right text-[var(--muted)]">
            {Math.round(acc * 100)}% · {n}
          </span>
        </div>
      ))}
    </div>
  );
}
 
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));