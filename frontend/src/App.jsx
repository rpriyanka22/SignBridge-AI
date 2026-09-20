import { useCallback, useEffect, useRef, useState } from "react";
import HolisticTracker from "./components/HolisticTracker";
import "./App.css";

const SIGN_TRANSLATIONS = {
  "open palm": "Hello",
  "thumbs up": "Yes",
  "thumbs down": "No",
  "peace": "Peace",
  "victory": "Victory",
  "okay": "Okay",
  "closed fist": "Closed Fist",
  "index point": "Index Point",
  "i love you": "I Love You",
  "crossed fingers": "Crossed Fingers",
  "call me": "Call Me",
  "rock on": "Rock On",
  "three": "Three",
  "four": "Four",
  "l shape": "L Shape",
  "pinch": "Pinch",
};

function cleanPrediction(prediction) {
  if (!prediction) return "";
  return prediction.toString().replace(/^\d+\.\s*/, "").trim();
}

function getConfidenceLabel(confidence) {
  if (confidence >= 80) return "High confidence";
  if (confidence >= 60) return "Medium confidence";
  return "Low confidence";
}

function speakWord(text) {
  if (!("speechSynthesis" in window) || !text) return;
  window.speechSynthesis.cancel();
  const speech = new SpeechSynthesisUtterance(text);
  speech.lang = "en-US";
  speech.rate = 0.9;
  speech.pitch = 1.0;
  window.speechSynthesis.speak(speech);
}

export default function App() {
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [translatedText, setTranslatedText] = useState("Waiting...");
  const [sentence, setSentence] = useState([]);
  const [constructedSentence, setConstructedSentence] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [trackerActive, setTrackerActive] = useState(false);

  const trackerRef = useRef(null);
  const activeUtteranceRef = useRef(null);
  const lastSpokenRef = useRef("");

  const handlePrediction = useCallback(
    (result) => {
      if (!result) return;

      const rawPrediction = result.detected_sign || result.prediction || "";
      const currentConfidence = Number(result.confidence || 0);

      setPrediction(rawPrediction);
      setConfidence(currentConfidence);

      if (result.constructed_sentence) {
        setConstructedSentence(result.constructed_sentence);
      }

      if (result.sentence && Array.isArray(result.sentence)) {
        setSentence(result.sentence);
      }

      const cleanSign = cleanPrediction(rawPrediction);
      const lookupKey = cleanSign.toLowerCase();

      const translated = SIGN_TRANSLATIONS[lookupKey] || result.translated_text || cleanSign;

      if (currentConfidence >= 60 && cleanSign && !["idle", "waiting...", "--", "uncertain"].includes(lookupKey)) {
        setTranslatedText(translated);

        if (result.is_new_sign && autoSpeak && lastSpokenRef.current !== translated) {
          speakWord(translated);
          lastSpokenRef.current = translated;
        }
      }
    },
    [autoSpeak]
  );

  const speakFullSentence = useCallback(() => {
    const textToSpeak = constructedSentence || sentence.join(" ");
    if (!textToSpeak || textToSpeak === "Waiting for gestures...") return;

    if (!("speechSynthesis" in window)) {
      alert("Text-to-speech is not supported in this browser.");
      return;
    }

    window.speechSynthesis.cancel();
    const speech = new SpeechSynthesisUtterance(textToSpeak);
    activeUtteranceRef.current = speech;

    speech.lang = "en-US";
    speech.rate = 0.85;
    speech.pitch = 1.0;

    speech.onstart = () => setIsSpeaking(true);
    speech.onend = () => {
      activeUtteranceRef.current = null;
      setIsSpeaking(false);
    };
    speech.onerror = () => {
      activeUtteranceRef.current = null;
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(speech);
  }, [constructedSentence, sentence]);

  const stopSpeaking = () => {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    activeUtteranceRef.current = null;
    setIsSpeaking(false);
  };

  const clearSentence = () => {
    // Notify WebSocket backend to reset memory state
    if (trackerRef.current?.sendClear) {
      trackerRef.current.sendClear();
    }
    setSentence([]);
    setConstructedSentence("");
    setTranslatedText("Waiting...");
    setPrediction("");
    setConfidence(0);
    lastSpokenRef.current = "";
    stopSpeaking();
  };

  const removeWord = (index) => {
    // Notify WebSocket backend to remove index from array
    if (trackerRef.current?.sendRemoveIndex) {
      trackerRef.current.sendRemoveIndex(index);
    }
    setSentence((previousSentence) => {
      const updated = previousSentence.filter((_, wordIndex) => wordIndex !== index);
      if (updated.length === 0) setConstructedSentence("");
      return updated;
    });
  };

  useEffect(() => {
    return () => {
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    };
  }, []);

  const confidenceLabel = getConfidenceLabel(confidence);

  return (
    <div className="app">
      <header className="top-header">
        <div className="brand-section">
          <div className="brand-icon">✋</div>
          <div>
            <h1>SIGNBRIDGE AI</h1>
            <p>Full-Body Sign Language & Intent Translation Engine</p>
          </div>
        </div>

        <div className="header-controls">
          <button
            type="button"
            className={`auto-speech-toggle-btn ${autoSpeak ? "active" : "inactive"}`}
            onClick={() => setAutoSpeak(!autoSpeak)}
            title="Toggle automatic speech output on gesture detection"
          >
            <span className="toggle-icon">{autoSpeak ? "🔊" : "🔇"}</span>
            <span className="toggle-text">
              Auto Speech: <strong>{autoSpeak ? "ON" : "OFF"}</strong>
            </span>
          </button>

          <div className="system-status">
            <span className="status-dot"></span> SYSTEM ONLINE
          </div>
        </div>
      </header>

      <main className="dashboard">
        <section className="camera-panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">LIVE INPUT</span>
              <h2>Webcam</h2>
            </div>
            <div className="live-badge">
              <span></span> LIVE
            </div>
          </div>

          <div className="camera-wrapper">
            <HolisticTracker
              ref={trackerRef}
              onPrediction={handlePrediction}
              onCameraStatus={setCameraReady}
              onTrackerStatus={setTrackerActive}
            />
            <div className="camera-corners corner-tl"></div>
            <div className="camera-corners corner-tr"></div>
            <div className="camera-corners corner-bl"></div>
            <div className="camera-corners corner-br"></div>
            <div className="scan-line"></div>
          </div>

          <div className="camera-status">
            <div className="status-item">
              <span className={`mini-dot ${cameraReady ? "active" : "inactive"}`}></span>
              <span>Camera {cameraReady ? "Ready" : "Starting..."}</span>
            </div>
            <div className="status-item">
              <span className={`mini-dot ${trackerActive ? "active" : "inactive"}`}></span>
              <span>{trackerActive ? "Signer Tracked" : "No Subject"}</span>
            </div>
          </div>
        </section>

        <section className="recognition-panel">
          <div className="recognition-card">
            <div className="card-label">DETECTED SIGN</div>
            <div className="detected-sign">
              {prediction ? cleanPrediction(prediction) : "--"}
            </div>
            <div className="recognition-subtitle">
              {prediction ? "Sign detected by AI" : "Show a sign to the camera"}
            </div>
          </div>

          <div className="recognition-card">
            <div className="card-title-row">
              <div className="card-label">CONFIDENCE</div>
              <div className="confidence-number">{confidence.toFixed(0)}%</div>
            </div>
            <div className="confidence-track">
              <div
                className="confidence-progress"
                style={{ width: `${Math.min(confidence, 100)}%` }}
              ></div>
            </div>
            <div className="confidence-footer">
              <span>{confidenceLabel}</span>
              <span>AI Recognition</span>
            </div>
          </div>

          <div className="recognition-card highlight">
            <div className="card-label">SMART INFERRED INTENT</div>
            <div className="translated-result">
              {constructedSentence || "Waiting for gestures..."}
            </div>
          </div>

          <div className="sentence-card">
            <div className="sentence-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div className="card-label">RAW GESTURE STREAM</div>
                <div className="sentence-count">
                  {sentence.length} word{sentence.length !== 1 ? "s" : ""}
                </div>
              </div>
              <button
                type="button"
                className="field-clear-btn"
                onClick={clearSentence}
                disabled={sentence.length === 0}
                title="Clear detected gesture stream"
              >
                <span>🗑 Clear Stream</span>
              </button>
            </div>
            <div className="sentence-content">
              {sentence.length === 0 ? (
                <div className="empty-sentence">Detected words will stack here step-by-step...</div>
              ) : (
                sentence.map((word, index) => (
                  <div className="word-chip" key={`${word}-${index}`}>
                    <span>{word}</span>
                    <button onClick={() => removeWord(index)} title="Remove word">
                      ×
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="actions">
            {!isSpeaking ? (
              <button
                className="speak-button"
                onClick={speakFullSentence}
                disabled={sentence.length === 0}
              >
                <span className="button-icon">🔊</span>
                <span>Speak Smart Sentence</span>
              </button>
            ) : (
              <button className="stop-button" onClick={stopSpeaking}>
                <span className="button-icon">■</span>
                <span>Stop Speaking</span>
              </button>
            )}

            <button className="clear-button" onClick={clearSentence}>
              <span className="button-icon">🗑</span>
              <span>Clear All</span>
            </button>
          </div>
        </section>
      </main>

      <footer className="dashboard-footer">
        <div className="footer-status">
          <span className="footer-dot"></span> Recognition Engine Active
        </div>
        <div className="footer-divider"></div>
        <div>
          Signs Detected: <strong>{sentence.length}</strong>
        </div>
        <div className="footer-divider"></div>
        <div>
          Current Confidence: <strong>{confidence.toFixed(0)}%</strong>
        </div>
        <div className="footer-version">SignBridge AI v3.0 (Holistic)</div>
      </footer>
    </div>
  );
}