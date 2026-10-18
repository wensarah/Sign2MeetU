// Splits a continuous stream of camera frames into recordings and signs.
//
// A RECORDING (one sentence) starts automatically when hands enter the frame
// and stops when no hands are seen for SENTENCE_GAP_MS. Inside it, the signer
// lowers their hands briefly between signs:
//
//   hands enter ──► recording + sign start
//   hands down for GAP_MS ──► sign ends (its frames go to the classifier)
//   hands down for SENTENCE_GAP_MS ──► recording stops (sentence ends)
//
// Safety limits: a recording is cut off at RECORDING_MAX_MS, and one shorter
// than RECORDING_MIN_MS (a hand passing through) is thrown away.
//
// Pure logic, no browser APIs, so it can be tested with fake frames.

import type { Frame } from "./landmarks";

/** Consecutive frames with a hand before a sign counts as started (ignores flickers). */
export const START_FRAMES = 2;
/** Hands down this long ends the current sign. */
export const GAP_MS = 450;
/** A sign this long is cut off even if the hands never drop. */
export const MAX_SIGN_MS = 4000;
/** Shorter "signs" are treated as noise. */
export const MIN_SIGN_MS = 300;
/** No hands for this long stops the recording (ends the sentence). */
export const SENTENCE_GAP_MS = 2000;
/** A recording is stopped automatically at this length. */
export const RECORDING_MAX_MS = 15000;
/** Recordings shorter than this (first hand to last hand) are discarded. */
export const RECORDING_MIN_MS = 1000;

export type SegmenterEvent =
  | { type: "recordingStart" }
  | { type: "signStart" }
  | { type: "signEnd"; frames: (Frame | null)[] }
  | { type: "signCancel" } // started, but too short to be a sign
  | { type: "sentenceEnd"; reason: "pause" | "maxLength" | "manual" }
  | { type: "sentenceCancel" }; // recording too short, or no real signs in it

export class SignSegmenter {
  private frames: (Frame | null)[] | null = null; // non-null while a sign is in progress
  private prevFrame: Frame | null = null;
  private handRun = 0;
  private signStartedAt = 0;
  private lastHandAt = 0;
  private recordingStartedAt: number | null = null; // non-null while recording
  private signsInSentence = 0;

  push(frame: Frame | null, now: number): SegmenterEvent[] {
    const events: SegmenterEvent[] = [];
    const hasHand = frame !== null;
    if (hasHand) {
      this.lastHandAt = now;
      this.handRun++;
    } else {
      this.handRun = 0;
    }

    if (this.frames) {
      this.frames.push(frame);
      const handsDropped = !hasHand && now - this.lastHandAt >= GAP_MS;
      const tooLong = now - this.signStartedAt >= MAX_SIGN_MS;
      if (handsDropped || tooLong) events.push(this.endSign());
    } else if (hasHand && this.handRun >= START_FRAMES) {
      if (this.recordingStartedAt === null) {
        this.recordingStartedAt = now;
        events.push({ type: "recordingStart" });
      }
      this.frames = this.prevFrame ? [this.prevFrame, frame] : [frame];
      this.signStartedAt = now;
      events.push({ type: "signStart" });
    }
    this.prevFrame = frame;

    if (this.recordingStartedAt !== null) {
      if (now - this.recordingStartedAt >= RECORDING_MAX_MS) {
        events.push(...this.endRecording("maxLength"));
      } else if (!this.frames && !hasHand && now - this.lastHandAt >= SENTENCE_GAP_MS) {
        events.push(...this.endRecording("pause"));
      }
    }
    return events;
  }

  /** Stop recording now (the "Done" button or pausing). */
  flush(): SegmenterEvent[] {
    return this.recordingStartedAt === null ? [] : this.endRecording("manual");
  }

  /** Forget everything in progress (camera switch, clear). */
  reset() {
    this.frames = null;
    this.prevFrame = null;
    this.handRun = 0;
    this.recordingStartedAt = null;
    this.signsInSentence = 0;
  }

  /** Remove one sign from the current sentence's count (after Undo). */
  forgetLastSign() {
    this.signsInSentence = Math.max(0, this.signsInSentence - 1);
  }

  /** 0..1 progress toward the automatic stop while the hands are down, else 0. */
  sentenceProgress(now: number): number {
    if (this.recordingStartedAt === null || this.frames) return 0;
    return Math.min(1, Math.max(0, (now - this.lastHandAt) / SENTENCE_GAP_MS));
  }

  /** How long the current recording has been running, or 0. */
  recordingMs(now: number): number {
    return this.recordingStartedAt === null ? 0 : now - this.recordingStartedAt;
  }

  get recording(): boolean {
    return this.recordingStartedAt !== null;
  }

  get signing(): boolean {
    return this.frames !== null;
  }

  private endSign(): SegmenterEvent {
    const frames = this.frames!;
    this.frames = null;
    if (this.lastHandAt - this.signStartedAt < MIN_SIGN_MS) return { type: "signCancel" };
    this.signsInSentence++;
    return { type: "signEnd", frames };
  }

  private endRecording(reason: "pause" | "maxLength" | "manual"): SegmenterEvent[] {
    const events: SegmenterEvent[] = [];
    if (this.frames) events.push(this.endSign());
    const length = this.lastHandAt - this.recordingStartedAt!;
    const keep = length >= RECORDING_MIN_MS && this.signsInSentence > 0;
    events.push(keep ? { type: "sentenceEnd", reason } : { type: "sentenceCancel" });
    this.recordingStartedAt = null;
    this.signsInSentence = 0;
    return events;
  }
}
