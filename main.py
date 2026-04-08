import time
import threading
from diva_mic import MicVAD
from diva_stt import transcribe_audio, transcription_queue
from diva_brain import DivaBrain

brain = DivaBrain(debug=False, use_mcp=True)
current_job_id = 0
lock = threading.Lock()
running = True
exit_phrases = ["bye", "goodbye", "exit", "quit", "see you"]

def is_exit(text: str) -> bool:
    text = text.lower().strip()
    return any(phrase in text for phrase in exit_phrases)

def process(text: str, job_id: int):
    start_time = time.time()
    print(f"\n[LLM] Processing: {text}")
    
    for token in brain.stream(text):
        if job_id != current_job_id:
            brain.stop()
            return
        print(token, end="", flush=True)
    
    print(f"\n[LLM] Response ready ({time.time()-start_time:.2f}s)")

def on_speech(audio_np, sample_rate):
    global current_job_id, running
    text = transcribe_audio(audio_np)
    print(f"\n[STT] {text}")

    if not text or len(text.strip()) < 2:
        return

    if is_exit(text):
        print("\n[SYSTEM] Exit...")
        running = False
        return

    with lock:
        current_job_id += 1
        job_id = current_job_id

    threading.Thread(target=process, args=(text, job_id), daemon=True).start()

def transcription_worker_loop():
    while running:
        try:
            utt_id, audio_np = transcription_queue.get(timeout=0.5)
            on_speech(audio_np, sample_rate=16000)
        except:
            continue

def main():
    global running
    print("="*60)
    print("DIVA - Voice Assistant with MCP")
    print("="*60)
    threading.Thread(target=transcription_worker_loop, daemon=True).start()

    vad = MicVAD(on_speech_ready=on_speech)
    vad.start()

    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    finally:
        vad.stop()

if __name__ == "__main__":
    main()
