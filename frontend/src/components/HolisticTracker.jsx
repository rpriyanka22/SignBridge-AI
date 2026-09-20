import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

// Access MediaPipe via globally loaded CDN scripts to bypass Vite CJS/ESM bundling conflicts
const Holistic = window.Holistic;
const POSE_CONNECTIONS = window.POSE_CONNECTIONS;
const HAND_CONNECTIONS = window.HAND_CONNECTIONS;

const Camera = window.Camera;
const drawConnectors = window.drawConnectors;
const drawLandmarks = window.drawLandmarks;

const WS_URL = "ws://127.0.0.1:8000/ws/stream";

const HolisticTracker = forwardRef(function HolisticTracker(
  { onPrediction, onCameraStatus, onTrackerStatus },
  ref
) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  const [cameraStatus, setCameraStatus] = useState("Starting...");
  const [trackerStatus, setTrackerStatus] = useState("Waiting...");

  useImperativeHandle(ref, () => ({
    sendClear: () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "clear" }));
      }
    },
    sendRemoveIndex: (index) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "remove_index", index }));
      }
    },
  }));

  useEffect(() => {
    let mounted = true;
    let camera = null;

    async function initializeHolistic() {
      try {
        setCameraStatus("Connecting WebSocket...");
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (onPrediction) onPrediction(data);
          } catch (err) {
            console.error("WS Parse Error:", err);
          }
        };

        setCameraStatus("Loading Holistic Model...");
        
        const holistic = new Holistic({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
        });

        holistic.setOptions({
          modelComplexity: 0, // 0 = High performance/low CPU load
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          refineFaceLandmarks: false, // Disabled heavy facial tensor regression for max FPS
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        holistic.onResults((results) => {
          if (!mounted) return;
          
          const canvasCtx = canvasRef.current?.getContext("2d");
          if (canvasCtx && results.image) {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            
            // Draw webcam feed frame
            canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);
            
            // Draw Pose Skeleton (Lightweight)
            if (results.poseLandmarks) {
              drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 3 });
            }
            
            // Draw Hands (High Contrast)
            if (results.leftHandLandmarks) {
              drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#CC0000", lineWidth: 4 });
            }
            if (results.rightHandLandmarks) {
              drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#00CC00", lineWidth: 4 });
            }
            
            canvasCtx.restore();
          }

          const isTrackingAnything = results.leftHandLandmarks || results.rightHandLandmarks || results.poseLandmarks;
          setTrackerStatus(isTrackingAnything ? "Tracking Active" : "Searching...");
          if (onTrackerStatus) onTrackerStatus(!!isTrackingAnything);

          // Package and transmit multi-modal data to Python backend
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              left_hand: results.leftHandLandmarks || [],
              right_hand: results.rightHandLandmarks || [],
              pose: results.poseLandmarks || [],
              face: results.faceLandmarks || []
            }));
          }
        });

        const videoElement = videoRef.current;
        if (!videoElement) return;

        camera = new Camera(videoElement, {
          onFrame: async () => {
            if (mounted && videoElement) {
              await holistic.send({ image: videoElement });
            }
          },
          width: 640,
          height: 480,
        });

        await camera.start();
        setCameraStatus("Ready");
        if (onCameraStatus) onCameraStatus(true);

      } catch (error) {
        console.error("Holistic Init Error:", error);
        setCameraStatus("Error");
        if (onCameraStatus) onCameraStatus(false);
      }
    }

    initializeHolistic();

    return () => {
      mounted = false;
      if (camera) {
        try { camera.stop(); } catch(e) {}
      }
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Hidden video element for raw feed processing */}
      <video ref={videoRef} autoPlay muted playsInline style={{ display: "none" }} />
      
      {/* Canvas displays the feed + AI skeletal overlay */}
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
      />

      <div style={{
        position: "absolute", top: "16px", left: "16px", background: "rgba(0,0,0,0.75)",
        color: "#fff", padding: "8px 12px", borderRadius: "8px", fontSize: "12px",
        fontFamily: "monospace", zIndex: 10
      }}>
        <div>Camera: <strong>{cameraStatus}</strong></div>
        <div>Engine: <strong>{trackerStatus}</strong></div>
      </div>
    </div>
  );
});

export default HolisticTracker;