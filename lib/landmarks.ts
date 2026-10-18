// Turns camera frames into "fingerprints" of a sign.
//
// Pipeline:  video frame ──► MediaPipe Hand Landmarker ──► 21 points per hand
//            ──► frameFeatures() (normalize)  ──► one row of numbers per frame
//            ──► makeFingerprint() (trim + resample) ──► 16 rows = one example
//
// Both the /train page and the main app must use THESE functions, so training
// data and live data are processed identically.
 
import type { HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision";
 
// Served by our own app; scripts/copy-mediapipe.mjs puts the files there on `npm install`.
const WASM_URL = "/mediapipe";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
 
/** Number of frames every fingerprint is resampled to. */
export const SEQ_LEN = 16;
/** Numbers per hand per frame: 21 points × (x, y) + wrist location (x, y) + present flag. */
const PER_HAND = 21 * 2 + 2 + 1;
/** Numbers per frame (two hands). */
export const FRAME_DIM = PER_HAND * 2;
/** Bump this if you change how features are computed; old datasets become incompatible. */
export const FEATURE_VERSION = "v1-16x90";
 
/** How much hand location (where on the body) counts versus hand shape. */
const LOCATION_WEIGHT = 3;
 
export type Frame = number[]; // length FRAME_DIM
export type Fingerprint = Frame[]; // length SEQ_LEN
 
// ---------------------------------------------------------------------------
// 1. Load MediaPipe (once, lazily, browser only)
// ---------------------------------------------------------------------------
let landmarkerPromise: Promise<HandLandmarker> | null = null;
 
export function loadHandLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      // Dynamic import: MediaPipe only works in the browser, not during server rendering.
      const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      const make = (delegate: "GPU" | "CPU") =>
        HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      try {
        return await make("GPU");
      } catch {
        return await make("CPU"); // some phones don't support the GPU path
      }
    })();
    landmarkerPromise.catch(() => (landmarkerPromise = null)); // allow retry on failure
  }
  return landmarkerPromise;
}
 
// ---------------------------------------------------------------------------
// 2. One frame of MediaPipe output -> one row of normalized numbers
// ---------------------------------------------------------------------------
/**
 * For each hand we store:
 *  - SHAPE: all 21 points relative to the wrist, divided by the hand's size.
 *    This makes the numbers the same whether the hand is near or far, left or
 *    right in the frame.
 *  - LOCATION: where the wrist is in the frame (centered on 0), so signs at
 *    the chin and the chest look different.
 *  - PRESENT: 1 if this hand was detected, 0 if not.
 *
 * Hands go into fixed slots by MediaPipe's handedness label ("Left"/"Right"),
 * so the same hand always lands in the same slot.
 */
export function frameFeatures(result: HandLandmarkerResult): Frame | null {
  if (!result.landmarks?.length) return null;
 
  const frame: Frame = new Array(FRAME_DIM).fill(0);
  const used = [false, false];
 
  result.landmarks.forEach((points, i) => {
    const label = result.handedness?.[i]?.[0]?.categoryName;
    let slot = label === "Left" ? 0 : 1;
    if (used[slot]) slot = 1 - slot; // two hands with the same label: put the second in the free slot
    if (used[slot]) return;
    used[slot] = true;
 
    const wrist = points[0];
    const middleKnuckle = points[9];
    const size = Math.hypot(middleKnuckle.x - wrist.x, middleKnuckle.y - wrist.y) || 1e-6;
 
    const base = slot * PER_HAND;
    points.forEach((p, j) => {
      frame[base + j * 2] = (p.x - wrist.x) / size;
      frame[base + j * 2 + 1] = (p.y - wrist.y) / size;
    });
    frame[base + 42] = (wrist.x - 0.5) * LOCATION_WEIGHT;
    frame[base + 43] = (wrist.y - 0.5) * LOCATION_WEIGHT;
    frame[base + 44] = 1;
  });
 
  return frame;
}
 
// ---------------------------------------------------------------------------
// 3. A recording (many frames) -> a fixed-size fingerprint
// ---------------------------------------------------------------------------
/**
 * Recordings have different numbers of frames (phones run at different
 * speeds), so we resample every recording to exactly SEQ_LEN frames by
 * picking evenly spaced moments and blending neighbouring frames.
 *
 * `frames` may contain null for frames where no hand was visible. Leading and
 * trailing empty frames are trimmed off; gaps in the middle are filled from
 * the nearest frame that had a hand.
 *
 * Returns null if too few frames had a hand.
 */
export function makeFingerprint(frames: (Frame | null)[], minHandFrames = 6): Fingerprint | null {
  let start = frames.findIndex((f) => f !== null);
  let end = frames.length - 1 - [...frames].reverse().findIndex((f) => f !== null);
  if (start === -1) return null;
 
  const trimmed = frames.slice(start, end + 1);
  if (trimmed.filter(Boolean).length < minHandFrames) return null;
 
  // Fill gaps with the previous frame that had a hand.
  const filled: Frame[] = [];
  let last: Frame = trimmed[0]!;
  for (const f of trimmed) {
    if (f) last = f;
    filled.push(last);
  }
 
  // Resample to SEQ_LEN by linear interpolation.
  const out: Fingerprint = [];
  for (let k = 0; k < SEQ_LEN; k++) {
    const t = (k * (filled.length - 1)) / (SEQ_LEN - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, filled.length - 1);
    const w = t - lo;
    out.push(filled[lo].map((v, d) => round(v * (1 - w) + filled[hi][d] * w)));
  }
  return out;
}
 
const round = (v: number) => Math.round(v * 1000) / 1000; // keeps dataset files small
 
// ---------------------------------------------------------------------------
// 4. Drawing helper for the camera preview
// ---------------------------------------------------------------------------
export function drawHands(ctx: CanvasRenderingContext2D, result: HandLandmarkerResult | null) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  if (!result?.landmarks) return;
  for (const points of result.landmarks) {
    ctx.fillStyle = "#f5b83d";
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}