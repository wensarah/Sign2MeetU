import { NextResponse } from "next/server";

// Scene scan: one photo (back camera, or the front camera's background while
// signing) -> what kind of place this is, the most prominent objects, and any
// readable signs. Used by /api/sentence to resolve THAT / THERE and name places.
// Privacy: the photo is only forwarded to Perceptron for this one request.
// It is never written to disk or logged.

export const runtime = "nodejs";
export const maxDuration = 30;

const PERCEPTRON_BASE = "https://api.perceptron.inc/v1";
const MAX_BYTES = 3 * 1024 * 1024; // the client sends 640-1024px JPEGs, far smaller than this
const MAX_OBJECTS = 8;
const MAX_TEXT = 5;

const SYSTEM_PROMPT = `You help a sign language interpreter app understand the signer's surroundings.
You will see one photo from a phone camera. It usually shows a person signing, with the place around them.
- If the scene includes pointing_at and the signs include THAT, THERE, or ME, use pointing_at to fill in what the signer means ("WANT THAT" + pointing_at "coffee cup" → "Can I have that coffee?").
- If the signs include PAIN or HURT and pointing_at is a body part, say where it hurts ("My knee hurts.").
- Use the place only to choose natural, polite phrasing. Never add requests, objects or meaning that were not signed.

Return:
- place: a short, common name for the kind of place (for example "cafe", "campus library", "lecture hall", "dorm laundry room", "hallway"). A few lowercase words.
- objects: up to ${MAX_OBJECTS} of the most prominent physical objects someone might point at, ask for, or talk about, most prominent first. Prefer objects close to the person or within reach. Use 1-3 lowercase words, adding a color or type when it tells similar objects apart ("red water bottle", "coffee cup", "laptop", "phone charger", "first aid kit"). Prefer things a person could want, use, or ask about over walls, floors and ceilings. Do not list people or body parts here.
- text: up to ${MAX_TEXT} short pieces of readable text on signs, doors, labels or screens that name a place or thing ("Restroom", "Tutoring Center", "Exit", "Room 204"). Copy them exactly. Empty list if none are readable.
- pointing_at: if the person is clearly pointing with an index finger or hand, name what they are pointing at.
  - An object: use the same name you used in objects (for example "coffee cup").
  - Their own body: name the body part (for example "knee", "head", "stomach").
  - Themselves (finger on their own chest): "self".
  - Not pointing, or the target is unclear: an empty string "".

Rules:
- Only include objects and text you can actually see. If the photo is blurry or dark, return shorter lists rather than guessing.
- Never guess at a pointing target. If you cannot tell what is being pointed at, return "".
- Do not describe the person's identity, face, clothing or appearance.
Answer ONLY with JSON matching the schema.`;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    place: { type: "string" },
    objects: { type: "array", items: { type: "string" } },
    text: { type: "array", items: { type: "string" } },
    pointing_at: { type: "string" },
  },
  required: ["place", "objects", "text"],
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

async function callPerceptron(dataUrl: string): Promise<{ place: string; objects: string[]; text: string[] }> {
  const res = await fetch(`${PERCEPTRON_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.PERCEPTRON_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.PERCEPTRON_MODEL || "perceptron-mk1.5",
      temperature: 0,
      max_completion_tokens: 300,
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
  const text = Array.isArray(parsed.text)
    ? parsed.text
        .filter((t: unknown): t is string => typeof t === "string")
        .map((t: string) => t.replace(/\s+/g, " ").trim().slice(0, 40))
        .filter(Boolean)
        .slice(0, MAX_TEXT)
    : [];
  return { place: place || "unknown", objects, text };
}
