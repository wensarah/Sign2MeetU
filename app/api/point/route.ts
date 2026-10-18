import { NextResponse } from "next/server";

// Pointing target: one still from the front camera with a green mark drawn on it
// (a ray from the fingertip for THAT, a circle for PAIN). Perceptron names what
// the mark points at: a part of the signer's body or an object/place.
// Privacy: the photo is only forwarded to Perceptron for this one request.

export const runtime = "nodejs";
export const maxDuration = 30;

const PERCEPTRON_BASE = "https://api.perceptron.inc/v1";
const MAX_BYTES = 2 * 1024 * 1024;

const COMMON = `The image is a still from a front-facing phone camera looking at a person who is using sign language. The camera faces the person, so the person's RIGHT side appears on the LEFT of the image.
Name body parts from the person's own point of view (e.g. "left knee", "stomach", "right shoulder"). Name a side only when it is clear.
Use short lowercase names. Answer ONLY with JSON matching the schema.`;

const PROMPTS = {
  point: `${COMMON}

A bright green line with a dark outline starts at the person's index fingertip (small green circle) and continues in the direction their finger points. If there is only a circle and no line, the finger points toward the camera.
What does the finger point at?
- If it points at a part of the person's own body, kind = "body" and target = that body part.
- Otherwise, if the line reaches a visible object or place, kind = "object" and target = the FIRST object it reaches, with a short descriptor if helpful (e.g. "red water bottle", "door", "window").
- If it points toward the camera, off the edge of the image, or at nothing identifiable, kind = "unknown" and target = "".`,
  location: `${COMMON}

A bright green circle with a dark outline marks where the person is making a sign with their hands.
Which part of the person's own body is the circle on or right next to?
- If it is on or next to a body part, kind = "body" and target = that body part (e.g. "stomach", "head", "chest", "left knee").
- If the circle is in empty space well in front of the body, kind = "unknown" and target = "".`,
} as const;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    kind: { type: "string", enum: ["body", "object", "unknown"] },
    target: { type: "string" },
  },
  required: ["kind", "target"],
  additionalProperties: false,
} as const;

export async function POST(req: Request) {
  const started = Date.now();

  let file: File | null = null;
  let mode: keyof typeof PROMPTS = "point";
  try {
    const form = await req.formData();
    file = form.get("image") as File | null;
    if (form.get("mode") === "location") mode = "location";
  } catch {
    return NextResponse.json({ error: "Expected multipart form data with 'image' and 'mode' fields." }, { status: 400 });
  }
  if (!file || file.size === 0) return NextResponse.json({ error: "No image received." }, { status: 400 });
  if (file.size > MAX_BYTES) return NextResponse.json({ error: "Image is too large." }, { status: 413 });
  const mime = (file.type || "image/jpeg").split(";")[0];
  if (!mime.startsWith("image/")) return NextResponse.json({ error: "The 'image' field must be an image." }, { status: 400 });

  if (!process.env.PERCEPTRON_API_KEY) {
    return NextResponse.json({ error: "PERCEPTRON_API_KEY is not set. Add it to .env.local (and to Vercel's environment variables)." }, { status: 500 });
  }

  try {
    const bytes = Buffer.from(await file.arrayBuffer());
    const result = await callPerceptron(`data:${mime};base64,${bytes.toString("base64")}`, PROMPTS[mode]);
    return NextResponse.json({ ...result, mode, ms: Date.now() - started });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[point] perceptron failed:", msg);
    return NextResponse.json({ error: "Pointing lookup failed.", details: [msg] }, { status: 502 });
  }
}

async function callPerceptron(dataUrl: string, systemPrompt: string): Promise<{ kind: string; target: string }> {
  const res = await fetch(`${PERCEPTRON_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.PERCEPTRON_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.PERCEPTRON_MODEL || "perceptron-mk1.5",
      temperature: 0,
      max_completion_tokens: 150,
      messages: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: dataUrl } },
            { type: "text", text: "What does the green mark point at?" },
          ],
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: { name: "point_result", strict: true, schema: RESPONSE_SCHEMA },
      },
    }),
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== "string") throw new Error(`Unexpected response: ${JSON.stringify(data).slice(0, 400)}`);

  // Models sometimes wrap JSON in ```json fences despite instructions.
  const cleaned = content.replace(/^```(?:json)?\s*|\s*```$/g, "").trim();
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`No JSON in response: ${content.slice(0, 200)}`);
  const parsed = JSON.parse(match[0]);

  const kind = ["body", "object", "unknown"].includes(parsed.kind) ? parsed.kind : "unknown";
  const target = kind === "unknown" || typeof parsed.target !== "string" ? "" : parsed.target.trim().toLowerCase().slice(0, 40);
  return { kind: target ? kind : "unknown", target };
}
