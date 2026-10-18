import { NextResponse } from "next/server";

// Gloss -> English: turns a sequence of recognized signs (ASL word order) into
// one natural English sentence. Text only, Perceptron only.
// Optional `scene` (from the scene scan) is used only to resolve references
// like THAT / THERE and to pick natural phrasing, never to add meaning.

export const runtime = "nodejs";
export const maxDuration = 30;

const PERCEPTRON_BASE = "https://api.perceptron.inc/v1";
const MAX_GLOSSES = 30;
const MAX_OBJECTS = 8;
const POINTING = ["THAT", "THIS", "THERE", "HERE", "IT", "POINT"];

type Scene = { place: string; objects: string[] };

const SYSTEM_PROMPT = `You translate American Sign Language (ASL) glosses into natural English.

You receive a list of glosses: the signs a person made, in ASL word order, written as uppercase words (for example "WHERE BATHROOM" or "ME WANT THAT"). ASL grammar differs from English: it often drops "is", "the" and "a", puts question words at the end, and uses pointing signs like THAT and THERE.

Write natural English that says what the signer meant, as a hearing person would say it out loud. Use one sentence, or two short sentences when the signs combine a request with a statement (ASL often does this). Add the subject (I / you) and helping words a speaker would use, e.g. HELP -> "Can you help me?", WHERE PAY -> "Where do I pay?".

Rules:
- Keep the signer's meaning exactly. Never add requests, facts, names, or feelings that were not signed.
- You may add small grammar words (is, the, a, I) and light politeness like "Excuse me" when it fits the setting, but nothing that changes the meaning.
- If a scene is provided and the glosses contain a pointing word (THAT, THIS, THERE, IT), name the object or place from the scene the signer is most likely pointing at. Scene objects are listed most prominent first; when several fit, pick the first one that makes sense with the other signs (something you can want, go to, buy, etc.). Do not leave "that" unresolved when a scene object fits.
- If a scene is provided, you may also choose wording that suits the setting (for example "restroom" in a restaurant). The scene must never change what was asked or said.
- If no scene is provided, keep pointing words as they are ("that", "there").
- If the glosses contain UNKNOWN, skip that sign.
- Use a question mark for questions.
- Answer ONLY with JSON matching the schema.

Examples:
- WANT THAT + scene "cafe; coffee cup, menu" -> "I want that coffee."
- WHERE BATHROOM + scene "restaurant; tables, menu" -> "Excuse me, where's the restroom?"
- HELP DOOR LOCKED + no scene -> "Can you help me? The door is locked."
- THANK_YOU + any scene -> "Thank you."
- WANT THAT + no scene -> "I want that."`;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: { sentence: { type: "string" } },
  required: ["sentence"],
  additionalProperties: false,
} as const;

export async function POST(req: Request) {
  const started = Date.now();

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Expected JSON: { glosses: string[], scene?: { place, objects } }" }, { status: 400 });
  }

  const glosses = parseGlosses((body as { glosses?: unknown })?.glosses);
  if (!glosses) {
    return NextResponse.json({ error: "`glosses` must be a non-empty array of strings." }, { status: 400 });
  }
  const scene = parseScene((body as { scene?: unknown })?.scene);

  if (!process.env.PERCEPTRON_API_KEY) {
    return NextResponse.json({ error: "PERCEPTRON_API_KEY is not set. Add it to .env.local (and to Vercel's environment variables)." }, { status: 500 });
  }

  try {
    const sentence = await callPerceptron(glosses, scene);
    return NextResponse.json({ sentence, usedScene: Boolean(scene), ms: Date.now() - started });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[sentence] perceptron failed:", msg);
    return NextResponse.json({ error: "Sentence translation failed.", details: [msg] }, { status: 502 });
  }
}

function parseGlosses(raw: unknown): string[] | null {
  if (!Array.isArray(raw)) return null;
  const glosses = raw
    .filter((g): g is string => typeof g === "string")
    .map((g) => g.trim().toUpperCase().replace(/\s+/g, "_"))
    .filter(Boolean)
    .slice(0, MAX_GLOSSES);
  return glosses.length ? glosses : null;
}

function parseScene(raw: unknown): Scene | null {
  if (!raw || typeof raw !== "object") return null;
  const { place, objects } = raw as { place?: unknown; objects?: unknown };
  const cleanPlace = typeof place === "string" ? place.trim().slice(0, 60) : "";
  const cleanObjects = Array.isArray(objects)
    ? objects.filter((o): o is string => typeof o === "string").map((o) => o.trim().slice(0, 40)).filter(Boolean).slice(0, MAX_OBJECTS)
    : [];
  if (!cleanPlace && cleanObjects.length === 0) return null;
  return { place: cleanPlace, objects: cleanObjects };
}

function userPrompt(glosses: string[], scene: Scene | null): string {
  const lines = [`Glosses: ${glosses.join(" ")}`];
  if (scene) {
    lines.push(`Scene: ${scene.place || "unknown place"}${scene.objects.length ? `; visible objects: ${scene.objects.join(", ")}` : ""}`);
    const pointing = glosses.filter((g) => POINTING.includes(g));
    if (pointing.length && scene.objects.length) {
      lines.push(`The signer pointed (${pointing.join(", ")}). Name which scene object they mean in the sentence.`);
    }
  } else {
    lines.push("Scene: none provided");
  }
  return lines.join("\n");
}

async function callPerceptron(glosses: string[], scene: Scene | null): Promise<string> {
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
        { role: "user", content: [{ type: "text", text: userPrompt(glosses, scene) }] },
      ],
      response_format: {
        type: "json_schema",
        json_schema: { name: "sentence_result", strict: true, schema: RESPONSE_SCHEMA },
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
  const sentence = String(JSON.parse(match[0]).sentence ?? "").trim();
  if (!sentence) throw new Error("Empty sentence in response.");
  return sentence;
}
