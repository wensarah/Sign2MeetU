#!/usr/bin/env node
// Merge everyone's downloaded dataset files into public/dataset.json.
//
//   node scripts/merge-datasets.mjs dataset-alex.json dataset-sam.json dataset-jo.json
//
// If public/dataset.json already exists, its examples are kept and the new
// files are added to it. Duplicates (same id) are skipped, so it's safe to
// merge the same file twice.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";

const OUT = "public/dataset.json";
const files = process.argv.slice(2);
if (files.length === 0) {
  console.error("Usage: node scripts/merge-datasets.mjs <file1.json> [file2.json ...]");
  process.exit(1);
}

const inputs = existsSync(OUT) ? [OUT, ...files] : files;
let version = null;
const seen = new Set();
const examples = [];

for (const f of inputs) {
  const data = JSON.parse(await readFile(f, "utf8"));
  version ??= data.version;
  if (data.version !== version) {
    console.warn(`⚠ Skipping ${f}: feature version ${data.version} doesn't match ${version}`);
    continue;
  }
  let added = 0;
  for (const e of data.examples ?? []) {
    if (seen.has(e.id)) continue;
    seen.add(e.id);
    examples.push(e);
    added++;
  }
  console.log(`${f}: +${added} examples`);
}

await mkdir("public", { recursive: true });
await writeFile(OUT, JSON.stringify({ version, examples }));

const bySign = {};
const bySigner = {};
for (const e of examples) {
  bySign[e.label] = (bySign[e.label] ?? 0) + 1;
  bySigner[e.signer] = (bySigner[e.signer] ?? 0) + 1;
}
console.log(`\nWrote ${OUT}: ${examples.length} examples`);
console.log("Signers:", bySigner);
console.log("Per sign:", bySign);
const thin = Object.entries(bySign).filter(([, n]) => n < 8);
if (thin.length) console.log("⚠ Signs with fewer than 8 examples:", thin.map(([s]) => s).join(", "));
