import time
import threading
from diva_mic import MicVAD
from diva_stt import transcribe_audio, transcription_queue
from diva_brain import DivaBrain
from diva_tts import speak_chunk, stop_speaking

# ---------------- GLOBAL ----------------
brain = DivaBrain(debug=True)   # LLM streaming debug
current_job_id = 0
lock = threading.Lock()
running = True
exit_phrases = ["bye", "goodbye", "exit", "quit", "see you"]

# ---------------- HELPER ----------------
def is_exit(text: str) -> bool:
    text = text.lower().strip()
    return any(phrase in text for phrase in exit_phrases)

# ---------------- PROCESS ----------------
def process(text: str, job_id: int):
    """
    Stream tokens from LLM and push to TTS immediately in parallel.
    """
    start_time = time.time()
    for token in brain.stream(text):
        if job_id != current_job_id:
            brain.stop()  # Interrupt LLM streaming if new job arrives
            return
        words = token.strip().split()
        for w in words:
            print(f"[MAIN] Sending to TTS: {w}")
            speak_chunk(w)
    print(f"[PROCESS DONE] Job #{job_id} ({time.time()-start_time:.2f}s)")

# ---------------- CALLBACK ----------------
def on_speech(audio_np, sample_rate):
    global current_job_id, running
    text = transcribe_audio(audio_np)
    print(f"\n[STT RESULT]: {text}")

    if not text or len(text.strip()) < 2:
        print("[STT] Ignored: too short/noisy")
        return

    if is_exit(text):
        print("\n[SYSTEM] Exit detected. Shutting down...")
        running = False
        stop_speaking()
        return

    with lock:
        current_job_id += 1
        job_id = current_job_id

    threading.Thread(target=process, args=(text, job_id), daemon=True).start()

# ---------------- TRANSCRIPTION WORKER ----------------
def transcription_worker_loop():
    while running:
        try:
            utt_id, audio_np = transcription_queue.get(timeout=0.5)
            print(f"\n[WORKER] Processing utterance #{utt_id}...")
            on_speech(audio_np, sample_rate=16000)
        except:
            continue

# ---------------- MAIN ----------------
def main():
    global running
    print("="*50)
    print("DIVA REAL-TIME STT → LLM → TTS STREAMING")
    print("="*50)

    threading.Thread(target=transcription_worker_loop, daemon=True).start()

    vad = MicVAD(on_speech_ready=on_speech)
    vad.start()

    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C pressed")
    finally:
        vad.stop()
        stop_speaking()
        print("[SYSTEM] Stopped")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()