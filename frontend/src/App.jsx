import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "./transport.js";

export default function App() {
  const [status, setStatus] = useState("disconnected"); // disconnected|connecting|connected
  const [botSpeaking, setBotSpeaking] = useState(false);
  const [talking, setTalking] = useState(false); // is the Speak button currently held?
  const [lines, setLines] = useState([]); // {who:"You"|"Bot", text}
  const clientRef = useRef(null);
  const audioRef = useRef(null); // <audio> element that plays the bot's voice

  const addLine = useCallback((who, text) => {
    setLines((prev) => [...prev, { who, text }]);
  }, []);

  // Attach an incoming WebRTC audio track to the <audio> element so it plays.
  // The SDK delivers audio as a raw MediaStreamTrack and does NOT play it for us.
  // Only the BOT's audio should play here — ignore the local mic track (which also
  // fires trackStarted) so the user doesn't hear themselves echoed back.
  //
  // The bot's voice is INDEPENDENT of push-to-talk: it plays whenever the bot
  // speaks, whether or not you are holding the Speak button. If the browser
  // blocks autoplay (no user gesture yet), we retry once on the next click/tap.
  const playBotTrack = useCallback((track, participant) => {
    if (track.kind !== "audio") return;
    if (participant?.local) return; // skip our own mic
    const el = audioRef.current;
    if (!el) return;
    el.srcObject = new MediaStream([track]);
    el.muted = false;
    const tryPlay = () => el.play();
    tryPlay().catch((err) => {
      console.warn("audio autoplay blocked; will retry on next interaction:", err);
      const retry = () => {
        tryPlay().catch(() => {});
        window.removeEventListener("pointerdown", retry);
        window.removeEventListener("keydown", retry);
      };
      window.addEventListener("pointerdown", retry, { once: true });
      window.addEventListener("keydown", retry, { once: true });
    });
  }, []);

  // --- Push-to-talk ---
  // The mic stays OFF. It is only enabled while the Speak button is held, so
  // nothing is sent to the bot unless you are actively pressing.
  const startTalking = useCallback(() => {
    const client = clientRef.current;
    if (!client || status !== "connected") return;
    client.enableMic(true);
    setTalking(true);
  }, [status]);

  const stopTalking = useCallback(() => {
    const client = clientRef.current;
    if (!client) return;
    client.enableMic(false);
    setTalking(false);
  }, []);

  const connect = useCallback(async () => {
    setStatus("connecting");
    // "Prime" the audio element during this click (a real user gesture) so the
    // browser will allow the bot's voice to autoplay later, even while idle.
    const el = audioRef.current;
    if (el) {
      el.muted = false;
      el.play().catch(() => {}); // empty play is fine; just registers the gesture
    }
    const client = createClient({
      onConnected: () => {
        setStatus("connected");
        // Push-to-talk: start muted. Audio is only sent while Speak is held.
        client.enableMic(false);
      },
      onDisconnected: () => {
        setStatus("disconnected");
        setTalking(false);
      },
      onBotStartedSpeaking: () => setBotSpeaking(true),
      onBotStoppedSpeaking: () => setBotSpeaking(false),
      onTrackStarted: (track, participant) => playBotTrack(track, participant),
      onUserTranscript: (data) => {
        if (data?.final && data.text) addLine("You", data.text);
      },
      onBotTranscript: (data) => {
        if (data?.text) addLine("Bot", data.text);
      },
      onError: (e) => {
        console.error(e);
        addLine("Bot", "[error] " + (e?.message || "connection failed"));
        setStatus("disconnected");
      },
    });
    clientRef.current = client;
    try {
      await client.connect();
    } catch (e) {
      console.error(e);
      setStatus("disconnected");
    }
  }, [addLine, playBotTrack]);

  const disconnect = useCallback(async () => {
    await clientRef.current?.disconnect();
    clientRef.current = null;
    setStatus("disconnected");
    setBotSpeaking(false);
    setTalking(false);
  }, []);

  // Hold the spacebar to talk (when connected and not typing in a field).
  // Also a global safety net: any pointer/key release, or the window losing
  // focus, turns the mic back off so it can never get stuck "on".
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.code !== "Space" || e.repeat) return;
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      e.preventDefault();
      startTalking();
    };
    const onKeyUp = (e) => {
      if (e.code !== "Space") return;
      stopTalking();
    };
    const onGlobalRelease = () => stopTalking();

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("pointerup", onGlobalRelease);
    window.addEventListener("blur", onGlobalRelease);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("pointerup", onGlobalRelease);
      window.removeEventListener("blur", onGlobalRelease);
    };
  }, [startTalking, stopTalking]);

  const connected = status === "connected";

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 560, margin: "40px auto", padding: 16 }}>
      <h1>Qwen Voice Agent</h1>
      <p style={{ color: "#666" }}>
        Push to talk: hold the <b>Speak</b> button (or the spacebar) while you talk,
        release when you're done. Deepgram transcribes, OpenAI replies, Qwen speaks back.
      </p>

      {/* Plays the bot's voice. autoPlay so a freshly-attached track starts immediately. */}
      <audio ref={audioRef} autoPlay />

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        {connected ? (
          <button onClick={disconnect}>Disconnect</button>
        ) : (
          <button onClick={connect} disabled={status === "connecting"}>
            {status === "connecting" ? "Connecting…" : "Connect"}
          </button>
        )}
        <span>Status: <b>{status}</b></span>
        <span>{connected ? (botSpeaking ? "🔊 bot speaking" : talking ? "🔴 sending" : "🎤 idle") : ""}</span>
      </div>

      {/* Push-to-talk button: mic is live only while this is held down. */}
      <button
        disabled={!connected}
        onPointerDown={(e) => {
          e.preventDefault();
          startTalking();
        }}
        onPointerUp={stopTalking}
        onPointerLeave={() => talking && stopTalking()}
        onContextMenu={(e) => e.preventDefault()}
        style={{
          width: "100%",
          padding: "18px",
          marginBottom: 16,
          fontSize: 18,
          fontWeight: 600,
          borderRadius: 10,
          border: "1px solid #ccc",
          cursor: connected ? "pointer" : "not-allowed",
          userSelect: "none",
          touchAction: "none",
          background: talking ? "#d33" : connected ? "#f3f3f3" : "#eee",
          color: talking ? "#fff" : "#333",
        }}
      >
        {talking ? "● Recording — release to send" : "🎙 Hold to Speak"}
      </button>

      <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12, minHeight: 200 }}>
        {lines.length === 0 ? (
          <p style={{ color: "#999" }}>Transcript will appear here…</p>
        ) : (
          lines.map((l, i) => (
            <p key={i} style={{ margin: "6px 0" }}>
              <b>{l.who}:</b> {l.text}
            </p>
          ))
        )}
      </div>
    </div>
  );
}
