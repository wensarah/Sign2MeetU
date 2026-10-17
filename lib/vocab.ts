// The fixed demo vocabulary. The model can ONLY answer with one of these
// keys (or UNKNOWN), which is what keeps accuracy usable.
//
// Person C: edit this list once you've picked your final signs, and give each
// one a short description of how it's signed. The description goes into the
// prompt and helps the model tell similar signs apart.

export type Sign = {
  key: string; // what the model returns
  text: string; // what we show and speak
  hint: string; // how it's signed (ASL), fed to the model
};

export const VOCAB: Sign[] = [
  { key: "HELLO", text: "Hello", hint: "Flat hand starts at the forehead and moves outward, like a salute." },
  { key: "THANK_YOU", text: "Thank you", hint: "Flat hand fingertips touch the chin, then move forward and down." },
  { key: "PLEASE", text: "Please", hint: "Flat hand on the chest moving in a circle." },
  { key: "SORRY", text: "Sorry", hint: "Closed fist (A handshape) on the chest moving in a circle." },
  { key: "YES", text: "Yes", hint: "Fist nods up and down at the wrist, like a nodding head." },
  { key: "NO", text: "No", hint: "Index and middle fingers snap closed against the thumb." },
  { key: "HELP", text: "Help", hint: "Thumbs-up fist resting on the open palm of the other hand; both lift together." },
  { key: "NAME", text: "Name", hint: "Index and middle fingers of both hands (H handshapes) tap together twice." },
  { key: "MY", text: "My", hint: "Flat palm placed on the chest." },
  { key: "YOU", text: "You", hint: "Index finger points at the viewer." },
  { key: "WATER", text: "Water", hint: "W handshape (three fingers up) taps the chin." },
  { key: "EAT", text: "Eat", hint: "Fingertips bunched together tap the mouth." },
  { key: "BATHROOM", text: "Bathroom", hint: "T handshape (thumb between index and middle fingers) shakes side to side." },
  { key: "MORE", text: "More", hint: "Both hands with fingertips bunched tap together at the tips." },
  { key: "STOP", text: "Stop", hint: "Edge of one flat hand chops down onto the palm of the other." },
  { key: "GOOD", text: "Good", hint: "Flat hand fingertips at the chin move down onto the other open palm." },
  { key: "BAD", text: "Bad", hint: "Flat hand fingertips at the chin, then flips palm-down moving away." },
  { key: "FRIEND", text: "Friend", hint: "Hooked index fingers interlock, then swap positions." },
  { key: "LOVE", text: "Love", hint: "Both fists crossed over the chest." },
  { key: "GOODBYE", text: "Goodbye", hint: "Open hand fingers fold down and up repeatedly, a wave." },
];

export const UNKNOWN = "UNKNOWN";
export const VOCAB_KEYS = [...VOCAB.map((s) => s.key), UNKNOWN];

export function textFor(key: string): string {
  return VOCAB.find((s) => s.key === key)?.text ?? "";
}
