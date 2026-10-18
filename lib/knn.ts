// k-nearest-neighbors sign classifier.
//
// "Training" = keeping a list of labeled examples (the dataset).
// "Predicting" = finding the K examples most similar to a new fingerprint and
//                letting them vote.
//
// YOUR TASKS: implement dtwDistance() and predict(). Everything else is done.
// The /train page shows "kNN not implemented yet" until you do.
 
import { FEATURE_VERSION, type Fingerprint, type Frame } from "./landmarks";
 
export type Example = {
  id: string;
  label: string; // sign key from lib/vocab.ts, e.g. "HELLO"
  signer: string; // who recorded it, used for honest testing
  seq: Fingerprint;
};
 
export type Dataset = {
  version: string;
  examples: Example[];
};
 
export type Prediction = {
  label: string; // the winning sign
  confidence: number; // 0..1, share of the vote the winner got
  top: { label: string; score: number }[]; // up to 3 best signs, best first
};
 
export const K = 5;
 
// ---------------------------------------------------------------------------
// DONE: distance between two single frames (Euclidean distance)
// ---------------------------------------------------------------------------
export function frameDistance(a: Frame, b: Frame): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}
 
// ---------------------------------------------------------------------------
// TODO 1: Dynamic Time Warping distance between two fingerprints
// ---------------------------------------------------------------------------
/**
 * How different are two recordings, allowing one to be faster or slower?
 *
 * Plain approach (no DTW): compare frame 0 with frame 0, frame 1 with frame 1…
 * That fails when one person signs faster: their frame 5 might match the
 * other person's frame 8.
 *
 * DTW finds the best way to line up the frames, then adds up the distances
 * along that best alignment.
 *
 * Steps:
 *  1. Let n = a.length and m = b.length.
 *  2. Make a table `cost` with (n + 1) rows and (m + 1) columns, filled with
 *     Infinity. Set cost[0][0] = 0.
 *  3. For i from 1 to n, and j from 1 to m:
 *       d = frameDistance(a[i - 1], b[j - 1])
 *       cost[i][j] = d + the smallest of:
 *           cost[i - 1][j]      (a moves ahead, b waits)
 *           cost[i][j - 1]      (b moves ahead, a waits)
 *           cost[i - 1][j - 1]  (both move ahead)
 *  4. Return cost[n][m] / (n + m)   (dividing keeps the scale comparable)
 *
 * Optional speed-up (try after it works): only fill cells where
 * Math.abs(i - j) <= 4. This is called a "Sakoe-Chiba band".
 */
export function dtwDistance(a: Fingerprint, b: Fingerprint): number {
  // TODO: implement. Delete the line below when you do.
  throw new Error("kNN not implemented yet: write dtwDistance() in lib/knn.ts");
}
 
// ---------------------------------------------------------------------------
// TODO 2: predict the sign for a new fingerprint
// ---------------------------------------------------------------------------
/**
 * Steps:
 *  1. For every example in `examples`, compute dtwDistance(seq, example.seq).
 *     Keep pairs of { label, distance }.
 *  2. Sort the pairs by distance, smallest first.
 *  3. Take the first `k` pairs. These are the nearest neighbors.
 *  4. Vote: for each neighbor, add 1 / (distance + 1e-6) to votes[label].
 *     (Closer neighbors get a bigger vote. Using an object like
 *     `const votes: Record<string, number> = {}` works well.)
 *  5. Turn votes into a list, sort by vote (biggest first).
 *  6. confidence = winner's votes / total votes.
 *  7. Return {
 *       label: the winner's label,
 *       confidence,
 *       top: the first 3 entries as { label, score }, where score is that
 *            label's share of the total vote,
 *     }
 *
 * If `examples` is empty, return { label: "UNKNOWN", confidence: 0, top: [] }.
 */
export function predict(seq: Fingerprint, examples: Example[], k: number = K): Prediction {
  // TODO: implement. Delete the line below when you do.
  throw new Error("kNN not implemented yet: write predict() in lib/knn.ts");
}
 
// ---------------------------------------------------------------------------
// DONE: honest accuracy test (leave-one-signer-out)
// ---------------------------------------------------------------------------
export type EvalReport = {
  overall: number; // accuracy 0..1
  total: number;
  bySigner: { signer: string; accuracy: number; n: number }[];
  bySign: { label: string; accuracy: number; n: number }[];
  confusions: { expected: string; got: string; count: number }[];
};
 
/**
 * For each signer: train on everyone else, test on that signer.
 * This answers "does it work on a person it has never seen?"
 * With only one signer in the dataset, falls back to leave-one-example-out.
 */
export function evaluate(examples: Example[], k: number = K): EvalReport {
  const signers = [...new Set(examples.map((e) => e.signer))];
  const results: { expected: string; got: string; signer: string }[] = [];
 
  for (const ex of examples) {
    const train =
      signers.length > 1
        ? examples.filter((e) => e.signer !== ex.signer)
        : examples.filter((e) => e.id !== ex.id);
    const p = predict(ex.seq, train, k);
    results.push({ expected: ex.label, got: p.label, signer: ex.signer });
  }
 
  const acc = (rs: typeof results) => (rs.length ? rs.filter((r) => r.expected === r.got).length / rs.length : 0);
  const group = (key: "signer" | "expected") => {
    const m = new Map<string, typeof results>();
    for (const r of results) m.set(r[key], [...(m.get(r[key]) ?? []), r]);
    return m;
  };
 
  const confusionCounts = new Map<string, number>();
  for (const r of results) {
    if (r.expected !== r.got) {
      const key = `${r.expected}→${r.got}`;
      confusionCounts.set(key, (confusionCounts.get(key) ?? 0) + 1);
    }
  }
 
  return {
    overall: acc(results),
    total: results.length,
    bySigner: [...group("signer")].map(([signer, rs]) => ({ signer, accuracy: acc(rs), n: rs.length })),
    bySign: [...group("expected")]
      .map(([label, rs]) => ({ label, accuracy: acc(rs), n: rs.length }))
      .sort((a, b) => a.accuracy - b.accuracy),
    confusions: [...confusionCounts]
      .map(([key, count]) => {
        const [expected, got] = key.split("→");
        return { expected, got, count };
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, 8),
  };
}
 
// ---------------------------------------------------------------------------
// DONE: dataset helpers
// ---------------------------------------------------------------------------
export function emptyDataset(): Dataset {
  return { version: FEATURE_VERSION, examples: [] };
}
 
/** Combine datasets, skipping duplicates and anything from an incompatible version. */
export function mergeDatasets(...sets: Dataset[]): Dataset {
  const seen = new Set<string>();
  const examples: Example[] = [];
  for (const s of sets) {
    if (s.version !== FEATURE_VERSION) continue;
    for (const e of s.examples) {
      if (!seen.has(e.id)) {
        seen.add(e.id);
        examples.push(e);
      }
    }
  }
  return { version: FEATURE_VERSION, examples };
}
 
export function newId(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}