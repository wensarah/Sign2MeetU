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
  { key: "SORRY", text: "Sorry", hint: "Closed fist (A handshape) on the chest moving in a circle." },
  { key: "YES", text: "Yes", hint: "Fist nods up and down at the wrist, like a nodding head." },
  { key: "NO", text: "No", hint: "Index and middle fingers snap closed against the thumb." },
  { key: "HELP", text: "Help", hint: "Thumbs-up fist resting on the open palm of the other hand; both lift together." },
  { key: "MY", text: "My", hint: "Flat palm placed on the chest." },
  { key: "YOU", text: "You", hint: "Index finger points at the viewer." },
  { key: "WATER", text: "Water", hint: "W handshape (three fingers up) taps the chin." },
  { key: "BATHROOM", text: "Bathroom", hint: "T handshape (thumb between index and middle fingers) shakes side to side." },
  { key: "STOP", text: "Stop", hint: "Edge of one flat hand chops down onto the palm of the other." },
  { key: "GOOD", text: "Good", hint: "Flat hand fingertips at the chin move down onto the other open palm." },
  { key: "BAD", text: "Bad", hint: "Flat hand fingertips at the chin, then flips palm-down moving away." },
  { key: "FRIEND", text: "Friend", hint: "Hooked index fingers interlock, then swap positions." },
  { key: "LOVE", text: "Love", hint: "Both fists crossed over the chest." },
  { key: "GOODBYE", text: "Goodbye", hint: "Open hand fingers fold down and up repeatedly, a wave." },
  { key: "TUTORING-CENTER", text:"tutoring center", hint: "Both hands form flat, open shapes (B-handshapes), palms facing down. Bend both wrists to pull the hands backward toward your chest in a small, dual tugging motion"},
  { key: "WHERE", text: "Where", hint: "Index finger shakes side to side in front of the body at chest level, with a questioning facial expression." },
  { key: "IS", text: "Is", hint: "Index finger of one hand points forward from the mouth, then moves down and forward in a small arc." },
  { key: "WIFI", text: "WiFi", hint: "Both hands with index and middle fingers extended (V handshapes) make small circles near each ear, representing signal waves." },
];

export const UNKNOWN = "UNKNOWN";
export const VOCAB_KEYS = [...VOCAB.map((s) => s.key), UNKNOWN];

export function textFor(key: string): string {
  return VOCAB.find((s) => s.key === key)?.text ?? "";
}
