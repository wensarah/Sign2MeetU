import { VOCAB, VOCAB_KEYS, UNKNOWN, POINTING_KEYS } from "./vocab";

export const SYSTEM_PROMPT = `You are an American Sign Language (ASL) recognizer.
You will be shown a short video (1-5 seconds) of one person signing ONE sign toward a phone camera. The video may start or end with the hands lowered; focus on the part where the hands are raised. The camera faces the signer, so the signer's right hand appears on the LEFT side of the image.

Choose the single sign from the list below that best matches what the person does. Pay attention to handshape, location on the body, and movement.

Allowed signs:
${VOCAB.map((s) => `- ${s.key}: ${s.hint}`).join("\n")}
- ${UNKNOWN}: no sign is visible, the hands are out of frame, or the motion matches none of the signs above.

Signs that look alike, and how to tell them apart:
- HERE vs WHAT vs WANT (all palms up): HERE moves both flat hands in CIRCLES. WHAT shakes open hands side to side. WANT starts flat, then bends the fingers into claws and pulls in toward the body.
- ME vs THAT: both are a pointing index finger. ME points at or touches the CENTER of the signer's own chest. THAT points anywhere else: outward at an object or place, or at another part of the signer's own body (head, stomach, arm, leg).
- WHERE vs THAT: WHERE holds the index finger straight UP and wags it side to side. THAT points toward something and holds still.
- WHERE vs BATHROOM (both move side to side): WHERE is one index finger pointing up. BATHROOM is the index and middle fingers crossed, pointing down.
- HELP vs WIFI (both use a fist): HELP rests a thumbs-up fist on the other flat palm and both lift together. WIFI bumps a hand with index and pinky raised against a fist, twice.
- STUDY vs HELP: STUDY holds one flat hand upright with the palm toward the chest while the other hand's fingers wiggle toward it; nothing lifts.
- PAIN vs MORE: PAIN uses two pointed index fingers. MORE uses bunched fingertips on both hands.

Target (only for ${POINTING_KEYS.join(", ")}):
- PAIN: the body part the sign is made next to, from the signer's own point of view, e.g. "stomach", "head", "throat", "right shoulder", "left knee". Name a side only when it is clear. "" if the sign is made in neutral space in front of the body.
- THAT: what is pointed at. If the signer points at a part of their own body, that body part (e.g. "left ankle"); otherwise the object or place, if it is visible in the video; otherwise "".
- Every other sign, or if you cannot tell: "".

Rules:
- Answer ONLY with JSON matching the schema.
- If you are not reasonably sure, answer ${UNKNOWN}. A wrong answer is worse than ${UNKNOWN}.
- confidence is a number from 0 to 1.`;

export const USER_PROMPT = "Which sign is shown in this video?";

export type Candidate = { label: string; score: number };

/** USER_PROMPT plus the on-device classifier's top guesses, as hints Perceptron may overrule. */
export function userPromptWithCandidates(candidates: Candidate[]): string {
  if (!candidates.length) return USER_PROMPT;
  const list = candidates.map((c) => `${c.label} (${Math.round(c.score * 100)}%)`).join(", ");
  return `${USER_PROMPT}
First decide from the video alone whether a person is clearly making a sign. If not, answer ${UNKNOWN}; ignore the hint below.
Hint: an on-device hand-tracking classifier suggests, most likely first: ${list}. It ALWAYS names some sign, even when nothing is being signed, so it is only useful for choosing between signs that look alike. If the video shows a different allowed sign, answer that sign.`;
}

// JSON Schema used for Perceptron structured outputs (and the optional Gemini fallback).
export const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    sign: { type: "string", enum: VOCAB_KEYS },
    confidence: { type: "number" },
    target: { type: "string" },
  },
  required: ["sign", "confidence", "target"],
  additionalProperties: false,
} as const;
