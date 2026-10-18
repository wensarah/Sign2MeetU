import { NextResponse } from "next/server";

// Scene scan: one photo from the back camera -> what kind of place this is and
// the most prominent objects. Used by /api/sentence to resolve THAT / THERE.
// Privacy: the photo is only forwarded to Perceptron for this one request.
// It is never written to disk or logged.

export const runtime = "nodejs";
export const maxDuration = 30;

const PERCEPTRON_BASE = "https://api.perceptron.inc/v1";
const MAX_BYTES = 2 * 1024 * 1024; // the client sends ~640px JPEGs, far smaller than this
const MAX_OBJECTS = 8;

const SYSTEM_PROMPT = `You describe a phone camera photo so another system can understand where a person is.
The photo may be a selfie with a person in the foreground. Ignore that person and describe the surroundings behind or around them.

Return:
- place: a short, common name for the kind of place (for example "cafe", "campus library", "lecture hall", "dorm laundry room", "hallway"). Use a few words, lowercase.
- objects: up to ${MAX_OBJECTS} of the most prominent physical objects someone might point at or talk about, most prominent first, as short lowercase nouns ("printer", "coffee cup", "vending machine"). No people, no body parts, no text transcriptions.

Always describe what you actually see in the photo, even for outdoor scenes or slightly blurry photos.
Answer ONLY with JSON matching the schema.`;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    place: { type: "string" },
    objects: { type: "array", items: { type: "string" } },
  },
  required: ["place", "objects"],
  additionalProperties: false,
} as const;

export async function POST(req: Request) {
  const started = Date.now();

  let file: File | null = null;
  try {
    const form = await req.formData();
    file = form.get("image") as File | null;
  } catch {
    return NextResponse.json({ error: "Expected multipart form data with an 'image' field." }, { status: 400 });
  }
  if (!file || file.size === 0) {
    return NextResponse.json({ error: "No image received." }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json({ error: "Image is too large. Send a ~640px JPEG." }, { status: 413 });
  }
  const mime = (file.type || "image/jpeg").split(";")[0];
  if (!mime.startsWith("image/")) {
    return NextResponse.json({ error: "The 'image' field must be an image." }, { status: 400 });
  }

  if (!process.env.PERCEPTRON_API_KEY) {
    return NextResponse.json({ error: "PERCEPTRON_API_KEY is not set. Add it to .env.local (and to Vercel's environment variables)." }, { status: 500 });
  }

  try {
    const bytes = Buffer.from(await file.arrayBuffer());
    const scene = await callPerceptron(`data:${mime};base64,${bytes.toString("base64")}`);
    return NextResponse.json({ ...scene, ms: Date.now() - started });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[scene] perceptron failed:", msg);
    return NextResponse.json({ error: "Scene scan failed.", details: [msg] }, { status: 502 });
  }
}

async function callPerceptron(dataUrl: string): Promise<{ place: string; objects: string[] }> {
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
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: dataUrl } },
            { type: "text", text: "Where is this, and what are the main objects?" },
          ],
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: { name: "scene_result", strict: true, schema: RESPONSE_SCHEMA },
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

  const place = typeof parsed.place === "string" ? parsed.place.trim().slice(0, 60) : "unknown";
  const objects = Array.isArray(parsed.objects)
    ? parsed.objects
        .filter((o: unknown): o is string => typeof o === "string")
        .map((o: string) => o.trim().toLowerCase().slice(0, 40))
        .filter(Boolean)
        .slice(0, MAX_OBJECTS)
    : [];
  return { place: place || "unknown", objects };
}
