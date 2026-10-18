// Copies MediaPipe's WebAssembly files into public/mediapipe so the app serves
// them itself (no CDN, and always the same version as the installed package).
// Runs automatically after `npm install` (see "postinstall" in package.json),
// including on Vercel. public/mediapipe is git-ignored.

import { cp, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";

const src = "node_modules/@mediapipe/tasks-vision/wasm";
const dest = "public/mediapipe";

if (!existsSync(src)) {
  console.warn("[copy-mediapipe] @mediapipe/tasks-vision not installed yet, skipping.");
} else {
  await mkdir(dest, { recursive: true });
  await cp(src, dest, { recursive: true });
  console.log("[copy-mediapipe] copied WASM files to public/mediapipe");
}
