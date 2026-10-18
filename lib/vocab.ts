// The fixed demo vocabulary. The model can ONLY answer with one of these
// keys (or UNKNOWN), which is what keeps accuracy usable.
//
// Each hint goes into Perceptron's prompt, so it must describe how OUR team
// actually signs it. Check new signs against Lifeprint or HandSpeak.
//
// `points: true` marks signs whose meaning depends on where the signer points
// or signs (a body part or an object). Perceptron reports that target too.

export type Sign = {
  key: string; // what the model returns
  text: string; // what we show and speak
  hint: string; // how it's signed (ASL), fed to the model
  points?: boolean; // Perceptron also reports the body part / object targeted
};

export const VOCAB: Sign[] = [
  { key: "ME", text: "Me", hint: "Index finger points at or touches the center of the signer's own chest, once." },
  {
    key: "HERE",
    text: "Here",
    hint: "Flat hands, and while keeping the palms towards the ceiling both hands move in a circle.",
  },
  {
    key: "THAT",
    text: "That",
    hint: "Index finger points outward, away from the body, toward an object or place.",
    points: true,
  },
  {
    key: "PAIN",
    text: "Pain",
    hint: "Both index fingers point toward each other and jab toward each other, usually once. Often made right next to the body part that hurts.",
    points: true,
  },
  { key: "WANT", text: "Want", hint: "Both hands palms up, begin flat, with fingers pointed outwards, then fingers bent like claw and pull in toward the body." },
  { key: "HELP", text: "Help", hint: "Thumbs-up fist resting on the flat palm of the other hand; both hands lift together." },
  { key: "MORE", text: "More", hint: "Both hands with fingertips bunched together (flattened O); the fingertips tap together, usually twice." },
  { key: "WHERE", text: "Where", hint: "Index finger held straight up, rest of hand is closed in a fist with the thumb placed over the other fingers, wags index finger side to side." },
  { key: "WHAT", text: "What", hint: "Both hands open with palms up, shaking slightly side to side, often with a questioning face." },
  { key: "WATER", text: "Water", hint: "One hand. W handshape (index, middle and ring fingers up) index finger taps the chin twice." },
  { key: "BATHROOM", text: "Bathroom", hint: "Middle and index finger crossed, and pointed downward, crossed fingers move side to side." },
  {
    key: "STUDY",
    text: "tutoring center",
    hint: "one hand is flat and vertical, with the palm facing the chest, and the other hand wriggles its fingers towards the opposite hand's palm.",
  },
  { key: "WIFI", text: "WiFi", hint: "One hand is balled into a fist. The other hand has the index and pinky pointed upwards with the thumb tucked over the middle and ring finger. both hands bump each other twice." },
];

export const UNKNOWN = "UNKNOWN";
export const VOCAB_KEYS = [...VOCAB.map((s) => s.key), UNKNOWN];
export const POINTING_KEYS = VOCAB.filter((s) => s.points).map((s) => s.key);

export function textFor(key: string): string {
  return VOCAB.find((s) => s.key === key)?.text ?? "";
}
