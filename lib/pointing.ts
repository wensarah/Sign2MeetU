// Finds the clearest "pointing" moment in a sign and marks it on a still frame,
// so Perceptron can answer a precise question ("what does this green line hit?")
// instead of guessing from a whole video clip.
//
//   THAT  ──► best frame of an extended index finger ──► green ray from the fingertip
//   PAIN  ──► frame where the two index fingers are closest ──► green circle there
//
// MediaPipe landmarks are normalized 0..1 in the raw (un-mirrored) camera image.

import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

type Pt = { x: number; y: number };

export type PointMark =
  | { mode: "point"; tip: Pt; dir: Pt | null; handSize: number } // dir null = pointing toward the camera
  | { mode: "location"; center: Pt; radius: number };

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * How much a hand looks like it's pointing: index finger extended, the other
 * fingers curled. Around +0.8 for a clear point, near 0 for an open hand or fist.
 */
function pointingScore(p: Pt[]): number {
  const size = dist(p[0], p[9]) || 1e-6;
  const index = dist(p[8], p[0]) / size;
  const others = (dist(p[12], p[0]) + dist(p[16], p[0]) + dist(p[20], p[0])) / (3 * size);
  return index - others;
}

/** Best pointing hand in this frame, if any hand is clearly pointing. */
export function findPoint(result: HandLandmarkerResult): { score: number; mark: PointMark } | null {
  let best: { score: number; mark: PointMark } | null = null;
  for (const p of result.landmarks ?? []) {
    const score = pointingScore(p);
    if (score < 0.4 || (best && score <= best.score)) continue;
    const size = dist(p[0], p[9]);
    const dx = p[8].x - p[5].x;
    const dy = p[8].y - p[5].y;
    const len = Math.hypot(dx, dy);
    // A finger pointing at the camera looks very short in 2D: no usable direction.
    const dir = len > 0.35 * size ? { x: dx / len, y: dy / len } : null;
    best = { score, mark: { mode: "point", tip: { x: p[8].x, y: p[8].y }, dir, handSize: size } };
  }
  return best;
}

/** Where a two-handed jab (PAIN) is happening: midpoint of the index tips. Closer tips score higher. */
export function findLocation(result: HandLandmarkerResult): { score: number; mark: PointMark } | null {
  const hands = result.landmarks ?? [];
  if (hands.length >= 2) {
    const [a, b] = hands;
    const gap = dist(a[8], b[8]);
    const size = (dist(a[0], a[9]) + dist(b[0], b[9])) / 2;
    return {
      score: 1 / (gap + 1e-3),
      mark: {
        mode: "location",
        center: { x: (a[8].x + b[8].x) / 2, y: (a[8].y + b[8].y) / 2 },
        radius: Math.max(size * 1.1, gap / 2 + size * 0.4),
      },
    };
  }
  if (hands.length === 1) {
    const p = hands[0];
    const size = dist(p[0], p[9]);
    return { score: 0, mark: { mode: "location", center: { x: p[8].x, y: p[8].y }, radius: size * 1.1 } };
  }
  return null;
}

/** Draw the mark onto a copy of the frame (canvas already holds the raw frame). */
export function drawMark(canvas: HTMLCanvasElement, mark: PointMark) {
  const ctx = canvas.getContext("2d")!;
  const { width: w, height: h } = canvas;
  const line = Math.max(3, Math.round(w / 160));
  ctx.lineCap = "round";

  const stroke = (draw: () => void) => {
    // Dark outline under bright green so it shows on any background.
    ctx.strokeStyle = "rgba(0,0,0,0.7)";
    ctx.lineWidth = line * 2.2;
    draw();
    ctx.strokeStyle = "#8ace00";
    ctx.lineWidth = line;
    draw();
  };

  if (mark.mode === "point") {
    const tip = { x: mark.tip.x * w, y: mark.tip.y * h };
    if (mark.dir) {
      // Extend the ray to the edge of the frame.
      const far = Math.hypot(w, h);
      const end = { x: tip.x + mark.dir.x * far, y: tip.y + mark.dir.y * far };
      stroke(() => {
        ctx.beginPath();
        ctx.moveTo(tip.x, tip.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      });
    }
    stroke(() => {
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, line * 3, 0, Math.PI * 2);
      ctx.stroke();
    });
  } else {
    stroke(() => {
      ctx.beginPath();
      ctx.arc(mark.center.x * w, mark.center.y * h, mark.radius * w, 0, Math.PI * 2);
      ctx.stroke();
    });
  }
}
