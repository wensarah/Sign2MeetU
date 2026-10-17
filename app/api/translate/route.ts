import { NextResponse } from "next/server";
import { SYSTEM_PROMPT, USER_PROMPT, RESPONSE_SCHEMA } from "@/lib/prompt";
import { VOCAB_KEYS, UNKNOWN, textFor } from "@/lib/vocab";

export const runtime = "nodejs";
export const maxDuration = 30;

const MAX_BYTES = 4 * 1024 * 1024; // stay under Vercel's ~4.5 MB body limit
const PERCEPTRON_BASE = "https://api.perceptron.inc/v1";

type Result = { sign: string; confidence: number };
type VideoPart = Record<string, unknown>;

export async function POST(req: Request) {
  const started = Date.now();

  let file: File | null = null;
  try {
    const form = await req.formData();
    file = form.get("video") as File | null;
  } catch {
    return NextResponse.json({ error: "Expected multipart form data with a 'video' field." }, { status: 400 });
  }
  if (!file || file.size === 0) {
    return NextResponse.json({ error: "No video received." }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json({ error: `Clip is too large (${(file.size / 1e6).toFixed(1)} MB). Record shorter or lower quality.` }, { status: 413 });
  }

  // Strip codec info: "video/webm;codecs=vp8" -> "video/webm"
  const mime = (file.type || "video/webm").split(";")[0];
  const bytes = Buffer.from(await file.arrayBuffer());

  if (!process.env.PERCEPTRON_API_KEY) {
    return NextResponse.json({ error: "PERCEPTRON_API_KEY is not set. Add it to .env.local (and to Vercel's environment variables)." }, { status: 500 });
  }

  const errors: string[] = [];

  // 1) Perceptron Mk1.5, the primary (and for the Perceptron track, the only) model.
  try {
    const { result, via } = await callPerceptron(bytes, mime);
    return respond(result, `perceptron (${via})`, started);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[translate] perceptron failed:", msg);
    errors.push(`perceptron: ${msg}`);
  }

  // 2) Optional emergency fallback. Leave GEMINI_API_KEY blank to keep this 100% Perceptron.
  if (process.env.GEMINI_API_KEY) {
    try {
      return respond(await callGemini(bytes.toString("base64"), mime), "gemini (fallback)", started);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[translate] gemini failed:", msg);
      errors.push(`gemini: ${msg}`);
    }
  }

  return NextResponse.json({ error: "Recognition failed.", details: errors }, { status: 502 });
}

function respond(raw: Result, provider: string, started: number) {
  const sign = VOCAB_KEYS.includes(raw.sign) ? raw.sign : UNKNOWN;
  const confidence = Math.max(0, Math.min(1, Number(raw.confidence) || 0));
  return NextResponse.json({ sign, text: textFor(sign), confidence, provider, ms: Date.now() - started });
}

function parseJson(text: string): Result {
  // Models sometimes wrap JSON in ```json fences despite instructions.
  const cleaned = text.replace(/^```(?:json)?\s*|\s*```$/g, "").trim();
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`No JSON in response: ${text.slice(0, 200)}`);
  return JSON.parse(match[0]);
}

// --- Perceptron Mk1.5 (direct API: https://docs.perceptron.inc) -------------
//
// Two ways to send the clip, controlled by PERCEPTRON_VIDEO_MODE:
//   inline - base64 data URL in a `video_url` part (fastest: one request)
//   upload - upload via the Files API, then reference it with `video_file_id`
//   auto   - try inline, and if Perceptron rejects it, retry with upload (default)
async function callPerceptron(bytes: Buffer, mime: string): Promise<{ result: Result; via: string }> {
  const mode = (process.env.PERCEPTRON_VIDEO_MODE || "auto").toLowerCase();

  if (mode !== "upload") {
    try {
      const part = { type: "video_url", video_url: { url: `data:${mime};base64,${bytes.toString("base64")}` } };
      return { result: await perceptronChat(part), via: "inline" };
    } catch (e) {
      if (mode === "inline") throw e;
      console.warn("[translate] inline video failed, retrying via file upload:", e instanceof Error ? e.message : e);
    }
  }

  const fileId = await uploadToPerceptron(bytes, mime);
  const part = { type: "video_file_id", video_file_id: { file_id: fileId } };
  return { result: await perceptronChat(part), via: "upload" };
}

async function perceptronChat(videoPart: VideoPart): Promise<Result> {
  const res = await fetch(`${PERCEPTRON_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.PERCEPTRON_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.PERCEPTRON_MODEL || "perceptron-mk1.5",
      temperature: 0,
      max_completion_tokens: 200,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: [videoPart, { type: "text", text: USER_PROMPT }] },
      ],
      response_format: {
        type: "json_schema",
        json_schema: { name: "sign_result", strict: true, schema: RESPONSE_SCHEMA },
      },
    }),
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const data = await res.json();
  const choice = data?.choices?.[0];
  const content = choice?.message?.content;
  if (typeof content !== "string") throw new Error(`Unexpected response: ${JSON.stringify(data).slice(0, 400)}`);
  if (choice.finish_reason && choice.finish_reason !== "stop") {
    console.warn("[translate] finish_reason:", choice.finish_reason);
  }
  return parseJson(content);
}

async function uploadToPerceptron(bytes: Buffer, mime: string): Promise<string> {
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("quicktime") ? "mov" : "webm";
  const form = new FormData();
  form.append("purpose", "vision");
  form.append("file", new Blob([new Uint8Array(bytes)], { type: mime }), `clip.${ext}`);

  const res = await fetch(`${PERCEPTRON_BASE}/files`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.PERCEPTRON_API_KEY}` },
    body: form,
  });
  if (!res.ok) throw new Error(`File upload HTTP ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const data = await res.json();
  if (!data?.id) throw new Error(`File upload returned no id: ${JSON.stringify(data).slice(0, 200)}`);
  return data.id as string;
}

// --- Optional Gemini fallback (REST, no SDK needed) -------------------------
async function callGemini(base64: string, mime: string): Promise<Result> {
  const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

  // Gemini's responseSchema doesn't accept additionalProperties.
  const { additionalProperties: _ignored, ...geminiSchema } = RESPONSE_SCHEMA;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": process.env.GEMINI_API_KEY! },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{ role: "user", parts: [{ inline_data: { mime_type: mime, data: base64 } }, { text: USER_PROMPT }] }],
      generationConfig: { temperature: 0, responseMimeType: "application/json", responseSchema: geminiSchema },
    }),
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text ?? "").join("");
  if (!text) throw new Error(`Unexpected response: ${JSON.stringify(data).slice(0, 300)}`);
  return parseJson(text);
}
