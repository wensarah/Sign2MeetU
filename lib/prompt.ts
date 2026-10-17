import { VOCAB, VOCAB_KEYS, UNKNOWN } from "./vocab";

export const SYSTEM_PROMPT = `You are an American Sign Language (ASL) recognizer.
You will be shown a short video (2-3 seconds) of one person signing ONE sign toward a front-facing phone camera. The video is mirrored like a selfie, so the signer's dominant hand may appear on either side.

Choose the single sign from the list below that best matches what the person does. Pay attention to handshape, location on the body, and movement.

Allowed signs:
${VOCAB.map((s) => `- ${s.key}: ${s.hint}`).join("\n")}
- ${UNKNOWN}: no sign is visible, the hands are out of frame, or the motion matches none of the signs above.

Rules:
- Answer ONLY with JSON matching the schema.
- If you are not reasonably sure, answer ${UNKNOWN}. A wrong answer is worse than ${UNKNOWN}.
- confidence is a number from 0 to 1.`;

export const USER_PROMPT = "Which sign is shown in this video?";

// JSON Schema used for Perceptron structured outputs (and the optional Gemini fallback).
export const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    sign: { type: "string", enum: VOCAB_KEYS },
    confidence: { type: "number" },
  },
  required: ["sign", "confidence"],
  additionalProperties: false,
} as const;
