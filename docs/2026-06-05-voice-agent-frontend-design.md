# Voice Agent — Browser ↔ Pipecat ↔ Inference Server — Design

**Date:** 2026-06-05
**Status:** Approved for implementation
**Scope:** Local (Apple Silicon / M4 / MPS). Adds a real voice loop on top of the
existing streaming TTS server. A React app in the browser captures your microphone,
a new Pipecat agent server runs Speech-to-Text → LLM → Text-to-Speech, and the
spoken reply plays back in the browser.

This document is written for a junior developer. It explains every term the first
time it appears, and every box in every diagram. No prior Pipecat knowledge assumed.

---

## 1. What we are building, in one paragraph

You open a web page. You click "Connect" and allow microphone access. You talk.
The system hears you, figures out a reply using an AI language model, turns that
reply into speech using our own Qwen text-to-speech model, and plays it back in
your browser. This is the same kind of product as Vapi or Retell: a live voice
agent you talk to and it talks back.

---

## 2. The words you need (glossary)

Read this once. Every term below is used later.

- **Client / frontend** — the React app running in your web browser. It captures
  the microphone and plays audio. It has no AI models inside it.
- **Server / backend** — a program running on your Mac (not in the browser) that
  does the heavy work. We will have **two** backend programs (see §3).
- **STT (Speech-to-Text)** — software that listens to audio of a human talking
  and produces the text of what they said. We use **Deepgram** (a cloud service).
- **LLM (Large Language Model)** — software that reads text and writes a text
  reply. We use **OpenAI** (a cloud service).
- **TTS (Text-to-Speech)** — software that reads text and produces audio of a
  voice saying it. We use **our own Qwen3-TTS model**, which already runs in the
  existing inference server.
- **Pipecat** — a Python framework for building real-time voice agents. It does
  not contain any AI model itself. Its job is to move audio and text between the
  pieces above in the right order, in real time. Think of it as the conductor of
  the four steps: receive mic audio → STT → LLM → TTS → send audio back.
- **Pipeline** — the ordered list of steps Pipecat runs. Our pipeline is literally:
  `microphone in → Deepgram STT → OpenAI LLM → Qwen TTS → speaker out`.
- **WebRTC** — the technology web browsers use to send and receive live audio and
  video with low delay. It is how the microphone audio leaves the browser and how
  the reply audio comes back.
- **SmallWebRTC** — Pipecat's built-in WebRTC support. "Small" means it runs
  entirely on your own machine and needs no third-party account, no API key, and
  no servers on the internet. The browser connects directly to your Pipecat
  program. We chose this so everything stays local and free.
- **Signaling** — before two programs can stream audio over WebRTC, they must
  first exchange a small amount of setup text (network addresses, audio formats).
  That first exchange is called signaling. Our Pipecat server has one HTTP URL
  that handles this. After signaling finishes, audio flows directly.
- **RTVI** — a small message format Pipecat uses to send non-audio information
  between the browser and the server over the same connection — for example, the
  text of what you said, the text of the reply, and connection status. The React
  SDK and the Pipecat server both speak RTVI, so you get these updates for free.
- **VAD (Voice Activity Detection)** — software that detects when you start and
  stop talking, so the system knows when your sentence is finished and it is its
  turn to reply. We use **Silero VAD**, which runs locally (no account needed).
- **Frame** — inside Pipecat, every piece of data moving through the pipeline is
  wrapped in an object called a frame. Audio travels as audio frames; text travels
  as text frames. You will see names like `TTSAudioRawFrame` in the code. A frame
  is just "one labeled piece of data flowing through the pipeline."
- **PCM** — raw, uncompressed audio: a long list of numbers, one per sample of
  sound. Our TTS server outputs PCM at 24,000 samples per second, mono.

---

## 3. The big picture: three programs, three jobs

There are **three separate programs** running at the same time. Each has exactly
one job. They are separate so that any one can crash, restart, or be replaced
without breaking the others.

```
   YOUR BROWSER                    YOUR MAC (two backend programs)
 ┌──────────────┐
 │  React app   │
 │  (frontend)  │
 │              │
 │ • mic capture│         WebRTC: live audio both ways
 │ • plays audio│◄──────────────────────────────────────────┐
 │ • shows text │                                            │
 └──────────────┘                                            │
        │  1. signaling (one HTTP request to set up WebRTC)  │
        ▼                                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  PROGRAM 2 — Pipecat agent server          (NEW, port 7860)       │
 │                                                                    │
 │  Runs the pipeline, in this exact order, in real time:             │
 │                                                                    │
 │   mic audio in                                                     │
 │      │                                                             │
 │      ▼                                                             │
 │   [Deepgram STT]  ── audio → text of what you said ──┐  (cloud)    │
 │      │                                               │            │
 │      ▼                                                            │
 │   [OpenAI LLM]    ── your text → reply text ─────────┘  (cloud)    │
 │      │                                                             │
 │      ▼                                                             │
 │   [QwenWSTTSService]  ── reply text → audio ─────────┐            │
 │      │                                               │            │
 │      ▼                                               │ WS /tts    │
 │   audio out (back to browser over WebRTC)            │            │
 └──────────────────────────────────────────────────────┼───────────┘
                                                         │
                                                         ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  PROGRAM 3 — Qwen TTS inference server  (EXISTS, port 8000)        │
 │                                                                    │
 │  Holds the Qwen3-TTS model in memory on the Mac GPU (MPS).         │
 │  Receives text, streams back PCM audio chunk by chunk.             │
 │  THIS FILE IS NOT CHANGED BY THIS PROJECT.                         │
 └──────────────────────────────────────────────────────────────────┘
```

**The most important rule:** the browser only ever talks to Program 2 (the Pipecat
server). It never talks to Program 3, never talks to Deepgram, never talks to
OpenAI. Program 2 is the only thing that knows about those. This is the property
that makes it "like Retell or Vapi": all the AI services hide behind one server,
and the browser only sees that one server.

### Why three programs and not one?

- **Program 3 (TTS) is expensive to start.** It loads a model into the GPU, which
  takes time. We start it once and leave it running. Program 2 can restart as
  often as we like without touching the model.
- **Program 2 holds your secret API keys** (Deepgram, OpenAI). The browser must
  never see those keys. Keeping them in Program 2, which the browser cannot read,
  is what keeps them secret.
- **Each program can be worked on alone.** You can edit the React app without
  restarting the model. You can change the pipeline without reloading the model.

---

## 4. What happens, step by step, when you say one sentence

This is the whole loop, in order. Numbers match the diagram below.

1. You click **Connect** in the browser. The React app and Program 2 do
   **signaling** (one HTTP request) to set up the WebRTC audio connection.
2. You speak. The browser captures your microphone and streams that audio to
   Program 2 over WebRTC.
3. **VAD** (running in Program 2) notices you started talking, and later notices
   you stopped. When you stop, Program 2 knows your turn is over.
4. The audio of your sentence goes to **Deepgram STT**, which sends back the text
   of what you said (for example, `"what time is it in Tokyo"`).
5. That text goes to the **OpenAI LLM**, which writes a reply text (for example,
   `"It's currently late evening in Tokyo."`).
6. The reply text goes to **QwenWSTTSService**. This opens a WebSocket to
   Program 3, sends the text, and receives **PCM audio chunks** streaming back.
7. Each audio chunk is sent to the browser over WebRTC **as it arrives** — you
   start hearing the reply before the whole sentence is finished generating.
8. Alongside the audio, Program 2 sends **RTVI** text messages so the browser can
   show "you said: …" and "bot said: …" on screen.

```
 BROWSER                  PROGRAM 2 (Pipecat)              CLOUD / PROGRAM 3
   │                            │                               │
   │ (1) click Connect          │                               │
   │ ── signaling HTTP ───────► │  set up WebRTC                 │
   │ ◄──────── ready ────────── │                               │
   │                            │                               │
   │ (2) mic audio ───────────► │                               │
   │                            │ (3) VAD: you stopped talking   │
   │                            │ (4) audio ──► Deepgram STT     │ (cloud)
   │                            │ ◄── "your text" ──             │
   │                            │ (5) text ──► OpenAI LLM        │ (cloud)
   │                            │ ◄── "reply text" ──            │
   │                            │ (6) reply text ──► Qwen TTS ──►│ Program 3 :8000
   │                            │ ◄──── PCM chunks ──────────────│
   │ (7) ◄── audio chunks ───── │  (streamed as they arrive)     │
   │ (8) ◄── RTVI text msgs ─── │  "you said / bot said"         │
   │     (you hear the reply)   │                               │
```

---

## 5. The frontend (React app) — what it contains

The React app is small. It uses the **official Pipecat client SDK**, which does
all the hard WebRTC work for us. We do not write raw WebRTC code.

Packages it uses (installed with npm):

- `@pipecat-ai/client-js` — core client: manages the connection and RTVI messages.
- `@pipecat-ai/client-react` — React hooks and components so we write normal React.
- `@pipecat-ai/small-webrtc-transport` — teaches the client how to connect to our
  SmallWebRTC server specifically.

The app's screen has only what is needed:

- A **Connect / Disconnect** button.
- A **microphone status** indicator (muted / live / you-are-speaking).
- A **transcript area** showing "You: …" and "Bot: …" lines as RTVI messages
  arrive. (No typing box — input is your voice.)

Project layout:

```
client/
  package.json          # lists the three SDK packages + React + Vite
  index.html            # the single page the browser loads
  vite.config.js        # dev server config (and proxy to :7860 for signaling)
  src/
    main.jsx            # React entry point; wraps app in the Pipecat provider
    App.jsx             # the UI: Connect button, mic status, transcript
    transport.js        # creates the SmallWebRTC transport pointed at :7860
```

You run it with `npm install` then `npm run dev`, which serves the page at
`http://localhost:5173`.

---

## 6. The backend (Pipecat agent server) — what it contains

This is the new Python program, Program 2. It does two things:

1. **Serves the signaling endpoint** so the browser can establish WebRTC.
2. **Runs the pipeline** for each connected browser.

It reuses the existing `QwenWSTTSService` (already in
`server_app/pipecat_app/qwen_ws_tts.py`) unchanged — that is the TTS step that
talks to Program 3.

New files:

```
server_app/pipecat_app/
  qwen_ws_tts.py        # EXISTS — unchanged (TTS step → Program 3)
  voice_agent.py        # NEW — builds and runs the pipeline (STT→LLM→TTS)
  server.py             # NEW — the :7860 web server + SmallWebRTC signaling
```

The pipeline built in `voice_agent.py`, in order:

```
transport.input()          # mic audio arriving from the browser
   → DeepgramSTTService     # audio → text
   → OpenAI context+LLM     # text → reply text (keeps conversation history)
   → QwenWSTTSService       # reply text → PCM audio (calls Program 3)
   → transport.output()     # audio sent back to the browser
```

VAD (Silero) is attached to `transport.input()` so the server knows when your
turn ends.

---

## 7. Dependencies and secrets that must be set up

These are **not yet installed / configured** and are required. This is real setup
work, not optional.

Python packages (installed into the existing `.venv`):

- `pipecat-ai[deepgram]` — the Deepgram STT service (currently missing).
- `pipecat-ai[webrtc]` — installs `aiortc`, the engine SmallWebRTC needs
  (currently missing).
- `pipecat-ai[silero]` — local VAD (the `onnxruntime` part is present; this pulls
  the rest if needed).
- OpenAI support is already importable; no extra install expected.

API keys (paid cloud accounts — you must create these):

- `DEEPGRAM_API_KEY` — from a Deepgram account.
- `OPENAI_API_KEY` — from an OpenAI account.

These go in a `.env` file that **Program 2 reads** and the browser never sees:

```
server_app/.env        # DEEPGRAM_API_KEY=...   OPENAI_API_KEY=...
```

`.env` must be git-ignored so the keys are never committed.

---

## 8. Error handling (what happens when something goes wrong)

- **Program 3 (TTS) is down** when the bot tries to speak → `QwenWSTTSService`
  already yields an `ErrorFrame`; Program 2 logs it and the browser shows a
  "bot unavailable" state instead of crashing.
- **Bad or missing API key** for Deepgram/OpenAI → the service raises on startup;
  Program 2 prints a clear message and exits so you fix the key, rather than
  failing silently mid-call.
- **Browser denies microphone permission** → the React SDK reports a connection
  error; the UI shows "microphone needed" and the Connect button stays available.
- **Browser disconnects mid-sentence** → Pipecat tears down that pipeline; the
  other two programs are unaffected.

---

## 9. How we will know each piece works (testing)

We build and verify in this order, smallest first, so a failure points at one
piece:

1. **Program 3 already proven** — `ws_client_test.py` shows text → PCM works.
2. **Pipeline without browser** — run `voice_agent.py` wired to a local audio
   transport (mic + speaker on the Mac directly), confirm: you talk → it replies.
   This proves STT → LLM → TTS with no WebRTC involved.
3. **Signaling alone** — start `server.py`, open the React app, click Connect,
   confirm WebRTC connects (status goes "connected") even before talking.
4. **Full loop** — talk in the browser, hear the reply, see the transcript.

Each step is a checkpoint. We do not move to the next until the current one works.

---

## 10. What this design does NOT include (non-goals)

- **No interruption handling / barge-in tuning** beyond Pipecat's defaults. You
  can interrupt the bot because VAD allows it, but we are not tuning that behavior.
- **No multiple simultaneous callers.** One browser, one conversation. Program 3
  already serves one request at a time (one model on the GPU).
- **No deployment to the internet.** Everything runs on localhost. SmallWebRTC is
  chosen specifically for local use; going to the public internet later would
  mean adding TURN servers or switching to the Daily transport — a separate task.
- **No change to Program 3.** The Qwen inference server and its files are untouched.
- **No custom WebRTC code.** We rely entirely on the official SDK and SmallWebRTC.

---

## 11. Summary table — who talks to whom

| From            | To                | Over            | Carries                          |
|-----------------|-------------------|-----------------|----------------------------------|
| Browser         | Pipecat (:7860)   | HTTP (once)     | Signaling to set up WebRTC       |
| Browser         | Pipecat (:7860)   | WebRTC          | Your mic audio (up)              |
| Pipecat (:7860) | Browser           | WebRTC          | Reply audio + RTVI text (down)   |
| Pipecat (:7860) | Deepgram          | HTTPS (cloud)   | Your audio → your text           |
| Pipecat (:7860) | OpenAI            | HTTPS (cloud)   | Your text → reply text           |
| Pipecat (:7860) | Qwen TTS (:8000)  | WebSocket       | Reply text → PCM audio           |

The browser row never mentions Deepgram, OpenAI, or :8000. That is the whole point.
