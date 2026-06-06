// Creates the Pipecat client configured for our SmallWebRTC agent on :7860.
// The browser only ever talks to this URL — never to Deepgram/OpenAI/:8000.
import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

const AGENT_BASE_URL = "http://localhost:7860";

export function createClient(callbacks) {
  return new PipecatClient({
    transport: new SmallWebRTCTransport({
      // The runner exposes the SmallWebRTC offer/answer endpoint here.
      connectionUrl: `${AGENT_BASE_URL}/api/offer`,
    }),
    enableMic: true,
    enableCam: false,
    callbacks,
  });
}
