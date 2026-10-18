# Sign2MeetU

Sign one word into your phone's camera, see it as text, hear it spoken.
Recognition runs on **Perceptron Mk1.5**, called directly through Perceptron's API (https://docs.perceptron.inc), with a fixed vocabulary of 20 ASL signs.

## Setup (5 minutes)

```bash
npm install
cp .env.example .env.local   # then paste in your API keys
npm run dev
```

- `PERCEPTRON_API_KEY`: create one at https://platform.perceptron.inc/ (ask the hackathon organizers about track credits).
- `PERCEPTRON_VIDEO_MODE`: `auto` (default) sends the clip inline and, if Perceptron rejects that, uploads it through the Files API and retries. Once you know which works, set it to `inline` or `upload` to skip the extra attempt.
- `GEMINI_API_KEY`: optional emergency fallback. **Leave it blank for the Perceptron track** so every result comes from Perceptron.

Quick check that your key works, before touching the app:

```bash
curl https://api.perceptron.inc/v1/chat/completions \
  -H "Authorization: Bearer $PERCEPTRON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"perceptron-mk1.5","messages":[{"role":"user","content":"Say hi"}]}'
```

Open http://localhost:3000 on your laptop to test. **Phones need HTTPS for the camera**, so for phone testing either:

1. Deploy to Vercel (recommended, do this in hour 1): push to GitHub → import in Vercel → add the env vars in Project Settings → Environment Variables. Or
2. Tunnel your local server: `npx ngrok http 3000` and open the https URL on your phone.

## Where things live

| File | What it does | Owner |
|---|---|---|
| `app/page.tsx` | Camera, 2.5s recording, results, transcript, text-to-speech | Person A |
| `app/api/translate/route.ts` | Receives the clip and calls Perceptron Mk1.5 (inline video, or Files API upload) | Person B |
| `lib/prompt.ts` | System prompt and JSON schema | Person B |
| `lib/vocab.ts` | **The sign list.** Keys, display text, and how each is signed | Person C |
| `scripts/eval.mjs` | Accuracy test over a folder of recorded clips | Person C |

## Changing the vocabulary

Edit `lib/vocab.ts`. Each entry needs a `key` (what the model returns), `text` (what's shown and spoken), and a `hint` describing the handshape, location, and movement. **Better hints = better accuracy.** Double-check every sign against a reliable source like Lifeprint (ASLU) or HandSpeak, since the default hints are a starting point.

## Measuring accuracy

1. Record clips with the app's camera conditions (front camera, ~arm's length, 2–3 seconds). The easiest way is your phone's camera app, then AirDrop/upload them.
2. Organize them as `test-clips/<SIGN_KEY>/<anything>.mp4`. Aim for 5–10 clips per sign across all three of you. A `test-clips/UNKNOWN/` folder of "not signing" clips is useful too.
3. With `npm run dev` running: `npm run eval`
