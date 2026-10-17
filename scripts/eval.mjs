#!/usr/bin/env node
// Accuracy test for the sign recognizer.
//
// Put clips in folders named after the sign key (from lib/vocab.ts):
//   test-clips/HELLO/alex-1.webm
//   test-clips/HELLO/sam-1.mp4
//   test-clips/THANK_YOU/alex-1.webm
//   test-clips/UNKNOWN/hands-down.webm   <- clips of NOT signing are useful too
//
// Then, with `npm run dev` running in another terminal:
//   npm run eval
//   npm run eval -- --dir test-clips --url http://localhost:3000
//
// Prints accuracy per sign, the most common mistakes, and writes results.csv.

import { readdir, readFile, writeFile } from "node:fs/promises";
import { join, extname } from "node:path";

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const DIR = arg("dir", "test-clips");
const URL = arg("url", "http://localhost:3000") + "/api/translate";
const MIME = { ".webm": "video/webm", ".mp4": "video/mp4", ".mov": "video/quicktime" };

const signs = (await readdir(DIR, { withFileTypes: true })).filter((d) => d.isDirectory()).map((d) => d.name);
if (signs.length === 0) {
  console.error(`No sign folders found in ${DIR}/. See the comment at the top of this file.`);
  process.exit(1);
}

const rows = [];
for (const expected of signs.sort()) {
  const files = (await readdir(join(DIR, expected))).filter((f) => MIME[extname(f).toLowerCase()]);
  for (const f of files) {
    const path = join(DIR, expected, f);
    const buf = await readFile(path);
    const form = new FormData();
    form.append("video", new Blob([buf], { type: MIME[extname(f).toLowerCase()] }), f);

    let got = "ERROR", confidence = 0, provider = "", ms = 0;
    try {
      const res = await fetch(URL, { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) ({ sign: got, confidence, provider, ms } = data);
      else got = `ERROR(${res.status})`;
    } catch (e) {
      got = "ERROR(network)";
    }
    const ok = got === expected;
    rows.push({ expected, got, ok, confidence, provider, ms, file: path });
    console.log(`${ok ? "✓" : "✗"} ${expected.padEnd(12)} → ${String(got).padEnd(12)} ${(confidence * 100).toFixed(0).padStart(3)}%  ${ms}ms  ${f}`);
  }
}

// Per-sign accuracy
console.log("\n=== Accuracy by sign ===");
const bySign = {};
for (const r of rows) (bySign[r.expected] ??= []).push(r);
const summary = Object.entries(bySign)
  .map(([sign, rs]) => ({ sign, n: rs.length, acc: rs.filter((r) => r.ok).length / rs.length }))
  .sort((a, b) => a.acc - b.acc);
for (const s of summary) {
  const bar = "█".repeat(Math.round(s.acc * 20)).padEnd(20, "·");
  console.log(`${s.sign.padEnd(12)} ${bar} ${(s.acc * 100).toFixed(0).padStart(3)}%  (n=${s.n})`);
}

const total = rows.filter((r) => r.ok).length / rows.length;
const avgMs = rows.reduce((a, r) => a + (r.ms || 0), 0) / rows.length;
console.log(`\nOverall: ${(total * 100).toFixed(1)}% on ${rows.length} clips, ${signs.length} signs. Avg latency ${avgMs.toFixed(0)} ms.`);

// Most common confusions
const confusions = {};
for (const r of rows.filter((r) => !r.ok)) {
  const k = `${r.expected} → ${r.got}`;
  confusions[k] = (confusions[k] || 0) + 1;
}
const top = Object.entries(confusions).sort((a, b) => b[1] - a[1]).slice(0, 8);
if (top.length) {
  console.log("\n=== Most common mistakes ===");
  for (const [k, n] of top) console.log(`${String(n).padStart(3)}×  ${k}`);
}

console.log("\nSigns under 50% are candidates to drop or re-describe in lib/vocab.ts.");

const csv = ["expected,got,correct,confidence,provider,ms,file", ...rows.map((r) =>
  [r.expected, r.got, r.ok, r.confidence, r.provider, r.ms, JSON.stringify(r.file)].join(",")
)].join("\n");
await writeFile("results.csv", csv);
console.log("Wrote results.csv");
