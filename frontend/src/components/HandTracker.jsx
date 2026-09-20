import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WS_URL = "ws://127.0.0.1:8000/ws/stream";

function HandTracker({ onPrediction, onCameraStatus, onHandStatus }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const wsRef = useRef(null);

  const [cameraStatus, setCameraStatus] = useState("Starting...");
  const [handStatus, setHandStatus] = useState("Waiting...");
  const [error, setError] = useState("");

  const onPredictionRef = useRef(onPrediction);
  const onCameraStatusRef = useRef(onCameraStatus);
  const onHandStatusRef = useRef(onHandStatus);

  useEffect(() => {
    onPredictionRef.current = onPrediction;
    onCameraStatusRef.current = onCameraStatus;
    onHandStatusRef.current = onHandStatus;
  }, [onPrediction, onCameraStatus, onHandStatus]);

  useEffect(() => {
    let mounted = true;

    async function initialize() {
      try {
        console.log("Initializing MediaPipe & WebSocket stream...");
        setCameraStatus("Starting...");
        if (onCameraStatusRef.current) onCameraStatusRef.current(false);
        setHandStatus("Waiting...");
        if (onHandStatusRef.current) onHandStatusRef.current(false);
        setError("");

        // Establish WebSocket Connection
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => console.log("WebSocket connected to backend.");
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (typeof onPredictionRef.current === "function") {
              onPredictionRef.current(data);
            }
          } catch (err) {
            console.error("Failed to parse WebSocket message:", err);
          }
        };
        ws.onerror = (err) => console.error("WebSocket connection error:", err);

        // Load MediaPipe HandLandmarker
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        if (!mounted) return;

        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!mounted) {
          handLandmarker.close();
          return;
        }

        landmarkerRef.current = handLandmarker;

        // Access Web Camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
          audio: false,
        });

        if (!mounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) throw new Error("Video stream element not found.");

        video.srcObject = stream;
        await video.play();

        setCameraStatus("Ready");
        if (onCameraStatusRef.current) onCameraStatusRef.current(true);

        detectHands();
      } catch (err) {
        console.error("Tracker initialization error:", err);
        if (mounted) {
          setCameraStatus("Error");
          if (onCameraStatusRef.current) onCameraStatusRef.current(false);
          setHandStatus("Not available");
          if (onHandStatusRef.current) onHandStatusRef.current(false);
          setError(err?.message || "Failed to initialize Camera or Hand Tracker.");
        }
      }
    }

    function detectHands() {
      if (!mounted) return;

      const video = videoRef.current;
      const landmarker = landmarkerRef.current;

      if (!video || !landmarker || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(detectHands);
        return;
      }

      try {
        const timestamp = performance.now();
        const results = landmarker.detectForVideo(video, timestamp);

        if (results?.landmarks?.length > 0) {
          setHandStatus("Detected");
          if (onHandStatusRef.current) onHandStatusRef.current(true);

          const hand = results.landmarks[0];
          const landmarks = [];

          // Wrist-Centered Landmark Normalization
          const wristX = hand[0].x;
          const wristY = hand[0].y;
          const wristZ = hand[0].z;

          for (let i = 0; i < 21; i++) {
            landmarks.push(
              hand[i].x - wristX,
              hand[i].y - wristY,
              hand[i].z - wristZ
            );
          }

          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ landmarks }));
          }
        } else {
          setHandStatus("Not Detected");
          if (onHandStatusRef.current) onHandStatusRef.current(false);

          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ landmarks: [] }));
          }
        }
      } catch (err) {
        console.error("Frame detection error:", err);
      }

      animationRef.current = requestAnimationFrame(detectHands);
    }

    initialize();

    return () => {
      mounted = false;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (landmarkerRef.current) landmarkerRef.current.close();
      if (wsRef.current) {
        if (
          wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING
        ) {
          wsRef.current.close();
        }
      }
    };
  }, []);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: "scaleX(-1)",
        }}
      />

      <div
        style={{
          position: "absolute",
          top: "16px",
          left: "16px",
          background: "rgba(0, 0, 0, 0.75)",
          color: "#ffffff",
          padding: "8px 12px",
          borderRadius: "8px",
          fontSize: "12px",
          fontFamily: "monospace",
          backdropFilter: "blur(4px)",
          zIndex: 10,
        }}
      >
        <div>Camera: <strong>{cameraStatus}</strong></div>
        <div>Hand: <strong>{handStatus}</strong></div>
      </div>

      {error && (
        <div
          style={{
            position: "absolute",
            bottom: "16px",
            left: "16px",
            right: "16px",
            padding: "10px",
            borderRadius: "6px",
            background: "rgba(220, 38, 38, 0.9)",
            color: "white",
            fontSize: "13px",
            zIndex: 10,
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}

export default HandTracker;