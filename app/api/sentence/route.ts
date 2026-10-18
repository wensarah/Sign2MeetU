import { NextResponse } from "next/server";

// Gloss -> English: turns a sequence of recognized signs (ASL word order) into
// natural English. Text only, Perceptron only.
//
// Each gloss may carry a `target`: what the signer pointed at or signed next to,
// as seen by Perceptron in that sign's video ("left ankle", "coffee cup").
// Optional `scene` (place + objects) and `history` (recent sentences) are used
// only to resolve references like THAT / HERE / MORE and to pick natural
// phrasing, never to add meaning.

export const runtime = "nodejs";
export const maxDuration = 30;

const PERCEPTRON_BASE = "https://api.perceptron.inc/v1";
const MAX_GLOSSES = 30;
const MAX_OBJECTS = 8;
const MAX_HISTORY = 3;
const POINTING = ["THAT", "THIS", "THERE", "IT", "POINT"];

type Gloss = { gloss: string; target: string };
type Scene = { place: string; objects: string[]; text: string[] };

const SYSTEM_PROMPT = `You translate American Sign Language (ASL) glosses into natural English, spoken aloud for a hearing person.

You receive the signs a person made, in ASL word order, as uppercase words (for example "WHERE BATHROOM" or "ME WANT THAT"). ASL grammar differs from English: it often drops "is", "the" and "a", puts question words at the end, and uses pointing as grammar.

Some signs come with what the signer pointed at or signed next to, written in brackets: THAT[left ankle] means the signer pointed at their own left ankle; THAT[coffee cup] means the signer pointed at a coffee cup; PAIN[stomach] means the PAIN sign was made at the stomach. These come from video and are the most reliable information about location. Always use them.

THAT is a pointing sign (at an object, a place, or the signer's own body). HERE means "here" (this place). POINT means the signer pointed at something (not the verb "to point"). ME means I / me. Multi-word signs are joined with underscores, e.g. TUTORING_CENTER.

Write natural English that says what the signer meant. Use one sentence, or two short sentences when the signs combine a request with a statement. Add the subject (I / you) and helping words a speaker would use.

Rules:
- Keep the signer's meaning exactly. Never add requests, facts, names, symptoms, or feelings that were not signed.
- You may add small grammar words (is, the, a, I, my, some) and light politeness like "Excuse me" when it fits, but nothing that changes the meaning.
- A bracketed target always wins over the scene. Use the scene only for pointing words WITHOUT a bracketed target, and for natural wording (for example "restroom" in a restaurant).
- If a scene is provided and a pointing word (THAT, THIS, THERE, IT) has no bracketed target, name the scene object the signer most likely means. Scene objects are listed most prominent first; pick the first one that makes sense with the other signs.
- If there is no target and no scene, keep pointing words as they are ("that", "there").
- Readable signs/text in the scene may be used to name a place or thing the signer referred to with the same meaning (e.g. BATHROOM with a sign reading "Restroom" -> "restroom"). Never add room numbers, directions or answers from the text that the signer didn't sign.
- Earlier sentences (if given) are only for resolving references like "more" or "that one". Do not repeat them.
- If the glosses contain UNKNOWN, skip that sign.
- Use a question mark for questions.
- Answer ONLY with JSON matching the schema.

Examples:
- ME PAIN THAT[left ankle] -> "I have pain in my left ankle."
- PAIN[head] -> "My head hurts."
- ME PAIN HERE -> "I'm in pain here."
- ME WANT WATER -> "I want some water."
- WANT THAT[coffee cup] -> "I want that coffee cup."
- WANT THAT + scene "cafe; coffee cup, menu" -> "I want that coffee."
- WHERE BATHROOM + scene "restaurant; tables, menu" -> "Excuse me, where's the restroom?"
- HELP ME -> "Can you help me?"
- MORE WATER -> "Can I have more water?"`;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: { sentence: { type: "string" } },
  required: ["sentence"],
  additionalProperties: false,
} as const;

export async function POST(req: Request) {
  const started = Date.now();

  let body: { glosses?: unknown; scene?: unknown; history?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Expected JSON: { glosses: (string | { gloss, target })[], scene?: { place, objects }, history?: string[] }" },
      { status: 400 }
    );
  }

  const glosses = parseGlosses(body?.glosses);
  if (!glosses) {
    return NextResponse.json({ error: "`glosses` must be a non-empty array of strings or { gloss, target } objects." }, { status: 400 });
  }
  const scene = parseScene(body?.scene);
  const history = parseHistory(body?.history);

  if (!process.env.PERCEPTRON_API_KEY) {
    return NextResponse.json({ error: "PERCEPTRON_API_KEY is not set. Add it to .env.local (and to Vercel's environment variables)." }, { status: 500 });
  }

  try {
    const sentence = await callPerceptron(userPrompt(glosses, scene, history));
    return NextResponse.json({ sentence, usedScene: Boolean(scene), ms: Date.now() - started });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[sentence] perceptron failed:", msg);
    return NextResponse.json({ error: "Sentence translation failed.", details: [msg] }, { status: 502 });
  }
}

const cleanText = (v: unknown, max: number) =>
  typeof v === "string" ? v.replace(/[\[\]\n\r]/g, " ").replace(/\s+/g, " ").trim().slice(0, max) : "";

function parseGlosses(raw: unknown): Gloss[] | null {
  if (!Array.isArray(raw)) return null;
  const glosses = raw
    .map((g): Gloss | null => {
      const word = typeof g === "string" ? g : (g as { gloss?: unknown })?.gloss;
      if (typeof word !== "string") return null;
      const gloss = word.trim().toUpperCase().replace(/\s+/g, "_");
      const target = typeof g === "string" ? "" : cleanText((g as { target?: unknown }).target, 40).toLowerCase();
      return gloss ? { gloss, target } : null;
    })
    .filter((g): g is Gloss => g !== null)
    .slice(0, MAX_GLOSSES);
  return glosses.length ? glosses : null;
}

function parseScene(raw: unknown): Scene | null {
  if (!raw || typeof raw !== "object") return null;
  const { place, objects, text } = raw as { place?: unknown; objects?: unknown; text?: unknown };
  const cleanPlace = cleanText(place, 60);
  const cleanObjects = Array.isArray(objects)
    ? objects.map((o) => cleanText(o, 40)).filter(Boolean).slice(0, MAX_OBJECTS)
    : [];
  const cleanText_ = Array.isArray(text) ? text.map((t) => cleanText(t, 40)).filter(Boolean).slice(0, 5) : [];
  if ((!cleanPlace || cleanPlace.toLowerCase() === "unknown") && cleanObjects.length === 0 && cleanText_.length === 0) {
    return null;
  }
  return { place: cleanPlace, objects: cleanObjects, text: cleanText_ };
}

function parseHistory(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((s) => cleanText(s, 200)).filter(Boolean).slice(-MAX_HISTORY);
}

function userPrompt(glosses: Gloss[], scene: Scene | null, history: string[]): string {
  const lines: string[] = [];
  if (history.length) lines.push(`Earlier sentences (oldest first): ${history.map((h) => `"${h}"`).join(" ")}`);
  lines.push(`Glosses: ${glosses.map((g) => (g.target ? `${g.gloss}[${g.target}]` : g.gloss)).join(" ")}`);
  if (scene) {
    lines.push(`Scene: ${scene.place || "unknown place"}${scene.objects.length ? `; visible objects: ${scene.objects.join(", ")}` : ""}`);
    if (scene.text.length) lines.push(`Readable signs/text in the scene: ${scene.text.map((t) => `"${t}"`).join(", ")}`);
    const unresolved = glosses.filter((g) => POINTING.includes(g.gloss) && !g.target).map((g) => g.gloss);
    if (unresolved.length && scene.objects.length) {
      lines.push(`The signer pointed (${unresolved.join(", ")}) without a known target. Name which scene object they mean in the sentence.`);
    }
  } else {
    lines.push("Scene: none provided");
  }
  return lines.join("\n");
}

async function callPerceptron(prompt: string): Promise<string> {
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
        { role: "user", content: [{ type: "text", text: prompt }] },
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
