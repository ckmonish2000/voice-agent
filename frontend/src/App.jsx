import { useCallback, useRef, useState } from "react";
import { createClient } from "./transport.js";

export default function App() {
  const [status, setStatus] = useState("disconnected"); // disconnected|connecting|connected
  const [botSpeaking, setBotSpeaking] = useState(false);
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
  const playBotTrack = useCallback((track, participant) => {
    if (track.kind !== "audio") return;
    if (participant?.local) return; // skip our own mic
    const el = audioRef.current;
    if (!el) return;
    el.srcObject = new MediaStream([track]);
    el.play().catch((err) => console.error("audio play() blocked:", err));
  }, []);

  const connect = useCallback(async () => {
    setStatus("connecting");
    const client = createClient({
      onConnected: () => setStatus("connected"),
      onDisconnected: () => setStatus("disconnected"),
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
  }, []);

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 560, margin: "40px auto", padding: 16 }}>
      <h1>Qwen Voice Agent</h1>
      <p style={{ color: "#666" }}>
        Talk into your mic. Deepgram transcribes, OpenAI replies, Qwen speaks it back.
      </p>

      {/* Plays the bot's voice. autoPlay so a freshly-attached track starts immediately. */}
      <audio ref={audioRef} autoPlay />

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        {status === "connected" ? (
          <button onClick={disconnect}>Disconnect</button>
        ) : (
          <button onClick={connect} disabled={status === "connecting"}>
            {status === "connecting" ? "Connecting…" : "Connect"}
          </button>
        )}
        <span>Status: <b>{status}</b></span>
        <span>{status === "connected" ? (botSpeaking ? "🔊 bot speaking" : "🎤 listening") : ""}</span>
      </div>

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
