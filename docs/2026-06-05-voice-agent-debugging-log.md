# Voice Agent — Debugging Log (what broke, how we found it, how we fixed it)

**Date:** 2026-06-05
**Audience:** A junior developer (or future me). Plain language, no analogies.
**Context:** After building the voice agent (browser → Pipecat server :7860 → Qwen TTS server :8000),
we turned it on and hit several problems in a row. This file records each one.

Read the design doc first if you haven't: `2026-06-05-voice-agent-frontend-design.md`.

---

## The three programs (quick reminder)

1. **TTS server, port 8000** — holds the voice model, turns text into audio. Already existed.
2. **Pipecat server, port 7860** — the new program. Listens to your mic, sends your speech to
   Deepgram (speech-to-text), sends that text to OpenAI (writes a reply), sends the reply to the
   TTS server, and streams the audio back to the browser.
3. **React web app, port 5173** — the page you open. Captures your microphone and plays the audio.

When something breaks, the bug is in one of these three, or in the connection between two of them.
Most of debugging was figuring out **which** of those places was actually broken.

---

## Bug 1 — "Audio is not playing in the web app"

### What we saw
The web app connected. We talked. We saw the reply text appear on screen ("Bot: ..."). But we
heard no sound at all.

### Why this was confusing
Seeing the "Bot:" text made it look like everything worked except the very last step. But text and
audio travel on two separate channels:
- The **text** ("Bot: It's Paris") comes through a side channel called RTVI. It says what the bot
  *would* say.
- The **audio** comes through a different channel (the WebRTC audio track).

So seeing text proved the speech-to-text and the language model worked. It proved **nothing** about
whether audio was made or played. We had to find out where in the audio path it broke.

### How we investigated
We could not read the server's printouts directly (they only went to a terminal window we didn't
control). So we wrote a small temporary program (`_diag_server.py`) that did the same job as the
real Pipecat server but also wrote a log file we *could* read.

**That log file was empty at first — and that was its own bug (see Bug 2).** Once we fixed the
log, it finally showed us the truth:
- The TTS server was called. ✅
- It returned real audio bytes (for example, 19,200 bytes for "The capital of France is Paris"). ✅
- Those audio bytes were handed to the part of Pipecat that sends audio to the browser. ✅

So the audio was being **made** and **sent**. The browser was receiving it and throwing it away.

### The root cause
The Pipecat JavaScript library hands the browser the incoming audio as a raw "media track." It does
**not** automatically play it. You have to take that track, put it into an HTML `<audio>` element,
and tell it to play. Our web app never did this. There was no `<audio>` element and no code to catch
the incoming track. So the sound arrived and was silently dropped.

### The fix
In `client/src/App.jsx` we added:
1. An `<audio autoPlay>` element on the page.
2. A handler called `onTrackStarted` that runs whenever an audio track arrives. It attaches the
   bot's audio track to that `<audio>` element and plays it.
3. A guard so we only play the **bot's** track, not the copy of your **own microphone** (otherwise
   you would hear yourself).

After this, audio played.

### Lesson
"Text appears but no sound" almost always means the browser received audio but never attached it to
an audio element. The Pipecat client SDK does not auto-play; the app must handle the track.

---

## Bug 2 — The diagnostic log was empty (our measuring tool was broken)

### What we saw
Our temporary diagnostic program was supposed to write a log file. After running it and talking, the
log file had only the one startup line. Nothing else. It looked like the whole pipeline never ran.

### Why this was dangerous
A broken measuring tool is worse than no tool, because it gives you false information. We almost
concluded "nothing is running" when in fact everything was running — we just weren't recording it.

### The root cause
Pipecat's startup code calls a function (`logger.remove()`) that deletes **all** existing log
destinations and sets up its own. Our log file destination was added early, then Pipecat deleted it
a moment later. So our file got nothing.

### The fix
We re-added our log file destination **after** Pipecat had finished its startup — specifically inside
the `bot()` function, which runs later, once per connection. After that, the log filled up correctly
and showed us the evidence we needed for Bug 1.

### Lesson
Before trusting what a diagnostic tool tells you, make sure the tool itself works. An empty log can
mean "nothing happened" OR "the logger was turned off." Confirm which.

---

## Bug 3 (not actually a bug) — "Peer connection not found" in the Network tab

### What we saw
In the browser's Network tab, a request came back with `{"detail":"Peer connection not found"}`.

### What it turned out to be
When the browser sets up a live audio connection, it first sends an "offer," gets back an identifier
for that connection, and then sends follow-up messages using that identifier. If you **restart the
server** but the **browser still has the old identifier**, the new server says "I don't know that
connection" → "Peer connection not found."

This was happening because we kept stopping and starting the :7860 server while the browser tab
stayed open with its old connection.

### The fix
Not a code fix. The procedure fix is: after restarting the server, fully **close the browser tab and
open a fresh one** so the browser starts a brand-new connection. (Just reloading is not always
enough; closing the tab guarantees the old connection is gone.)

### Lesson
"Peer connection not found" after a server restart usually means the browser is holding a stale
connection. Close the tab and reconnect.

---

## Bug 4 (not a bug, a misunderstanding) — Some replies produced 0 bytes of audio

### What we saw
In the diagnostic log, some replies generated audio (good), but others showed
`audio_frames=0 audio_bytes=0` — no audio at all. For example "Hi there!" produced nothing.

### What it turned out to be
Those zero-audio replies happened right when the log also said **"broadcasting interruption"** and
**"User started speaking."** The system thought you had started talking, so it **cancelled** the
bot's speech mid-way. That cancellation is intentional: in a voice agent, when the human starts
talking, the bot stops. The 0 bytes were the bot being correctly interrupted, not a failure to make
audio.

The interruptions fired too easily (background noise, a breath, or the bot's own sound leaking into
the microphone could trigger them), which made it look broken.

### The fix
Nothing to fix in the audio. The interruptions are normal behavior. Using headphones largely avoids
the accidental triggering. (Tuning how sensitive the interruption is would be a separate, optional
task — see Bug 5 for a related dead end we cleaned up.)

### Lesson
`audio_bytes=0` is not always a failure to produce audio. Check whether the speech was **interrupted
on purpose**. Read the surrounding log lines.

---

## Bug 5 — We almost shipped a setting that did nothing (and caught it)

### What we saw
While looking at the interruption behavior, we tried to make the voice-detection less sensitive by
passing a `vad_analyzer=...` setting into the transport configuration (`TransportParams`).

### What we discovered
That setting **does not exist** in this version of Pipecat (1.3.0). The configuration object quietly
accepted it and threw it away. This meant:
- The original code's `vad_analyzer=SileroVADAnalyzer()` had never done anything either — it was
  silently ignored from the start.
- Our new "tuning" would also do nothing.

We confirmed it by checking the list of real fields on the configuration object (no `vad_analyzer`),
and by reading Pipecat's own example, which sets only the audio in/out flags. We also confirmed from
the logs that the actual turn-taking was driven by the **speech-to-text** results (Deepgram telling
the system when speech started and stopped), not by that setting.

### The fix
We **removed** the dead setting and its now-unused import, and left a clear comment explaining that
`TransportParams` has no `vad_analyzer` field in this version and that turn-taking comes from the
speech-to-text layer. We chose to delete dead code rather than keep a setting that looks like it does
something but doesn't.

### Lesson
If a setting seems to have no effect, check that it is a **real** setting first. Some libraries
silently ignore unknown options. Shipping code that looks like it configures something, but doesn't,
will mislead the next person.

---

## Bug 6 — The voice was choppy / breaking up ("i a m do ing we ll")

### What we saw
Audio now played (Bug 1 was fixed), but the words came out broken into pieces with gaps, like
"i a m do ing we ll thanks." All the words were there, just chopped up.

### How we investigated
We looked at the timing in the diagnostic log and at the project's own README. The README already
measured the model's speed on this Mac: **RTF ≈ 11 to 13.** RTF means "real-time factor." An RTF of
12 means it takes about **12 seconds of computing to produce 1 second of speech.**

### The root cause
The browser plays audio in real time — it needs 1 second of sound ready every second, without gaps.
The model produces sound about **12 times too slowly**. Because we were sending each small piece of
audio the instant it was made, the player kept running out: it played a tiny piece, then waited for
the model to compute the next piece, then played that, then waited again. Those waits are the gaps
you heard. The audio was correct; it just could not be delivered fast enough to sound continuous.

This is not a code bug. It is the known speed of this model on Apple Silicon. (The long-term plan is
to replace the slow part with a much faster GPU version later, which makes RTF go below 0.1 — faster
than real time — and removes the problem.)

### The fix (a trade-off we chose on purpose)
In `server_app/pipecat_app/qwen_ws_tts.py` we changed the behavior from "send each piece as it is
made" to "**collect the whole reply, then send it all at once.**" Because the entire reply is ready
before playback starts, the player never runs out, so the audio plays smoothly.

The cost: you wait in silence while the model finishes (several seconds, longer for longer replies),
then you hear the whole reply cleanly. We decided smooth audio was worth the wait on this hardware.

We left a comment saying: when the faster GPU version exists, switch back to sending pieces as they
are made, for lower latency.

### Lesson
If streamed audio is choppy and the words are all present, suspect that the audio source is slower
than real time. Streaming only works smoothly if the source can keep up. If it can't, buffer the
whole thing first (you trade waiting time for smoothness).

---

## Summary table

| # | Symptom | Real cause | Fix | File changed |
|---|---------|-----------|-----|--------------|
| 1 | Text shows, no sound | Browser never attached the incoming audio track to an audio element | Added `<audio>` + `onTrackStarted` handler | `client/src/App.jsx` |
| 2 | Diagnostic log empty | Pipecat startup deleted our log destination | Re-add the log file destination inside `bot()` | (temporary diag file, since removed) |
| 3 | "Peer connection not found" | Browser held a stale connection after a server restart | Close tab, open fresh one | (procedure, no code) |
| 4 | Some replies make 0 audio | The bot was interrupted on purpose when it detected speech | None needed; normal behavior | (none) |
| 5 | Voice-detection setting | `vad_analyzer` isn't a real field in Pipecat 1.3.0; it was ignored | Removed the dead setting + import, added a comment | `server_app/pipecat_app/voice_agent.py` |
| 6 | Voice choppy / breaking up | Model is ~12x slower than real time on MPS; streaming underran | Buffer the whole reply, then play it once | `server_app/pipecat_app/qwen_ws_tts.py` |

---

## The single most useful debugging habit from this session

**Find out WHERE it breaks before deciding HOW to fix it.** Most of the time was spent proving which
of the three programs (or which connection between them) was actually at fault. The fixes themselves
were small once we knew the exact spot. Guessing at fixes before locating the break would have
changed the wrong code.

A close second: **make sure your measuring tool actually works** (Bug 2). We nearly drew the wrong
conclusion from an empty log that was empty for an unrelated reason.
