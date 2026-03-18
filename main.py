from diva_mic import MicVAD, SAMPLE_RATE
from diva_stt import on_speech_ready

def main():
    print("\n" + "="*70)
    print("DIVA: Voice → Intent → Speech (100% LOCAL)")
    print("="*70)
    print("\nInitializing...\n")

    # initialize VAD with callback
    vad = MicVAD(on_speech_ready=on_speech_ready)
    vad.start()

    try:
        while True:
            pass  # all processing handled by callback
    except KeyboardInterrupt:
        print("\n\n[STOP] shutting down...")
    finally:
        vad.stop()
        print("[STOP] complete\n")


if __name__ == "__main__":
    main()