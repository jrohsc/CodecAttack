#!/usr/bin/env python3
"""
LOCAL SCRIPT — Run this on your laptop (with speaker + microphone).

Plays adversarial audio through the speaker while simultaneously recording
from the microphone, capturing the over-the-air (OTA) signal.

Requirements (install on laptop):
    pip install sounddevice soundfile numpy

Usage:
    # List audio devices first
    python ota_local_record.py --list-devices

    # Play and record (default devices)
    python ota_local_record.py --play adversarial.wav --output recorded.wav

    # Specify devices
    python ota_local_record.py --play adversarial.wav --output recorded.wav \
                               --speaker 1 --mic 2

    # Just record from mic (you play from YouTube/Spotify manually)
    python ota_local_record.py --record-only --output recorded.wav --duration 12
"""

import argparse
import sys
import time

import numpy as np
import soundfile as sf


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print(f"\nDefault input  (mic):     {sd.default.device[0]}")
    print(f"Default output (speaker): {sd.default.device[1]}")


def play_and_record(play_path, output_path, speaker_id=None, mic_id=None,
                    record_sr=24000, extra_seconds=1.5):
    import sounddevice as sd

    # Load audio to play
    play_audio, play_sr = sf.read(play_path)
    play_duration = len(play_audio) / play_sr
    record_duration = play_duration + extra_seconds

    print(f"Audio:    {play_path}")
    print(f"Duration: {play_duration:.1f}s (recording {record_duration:.1f}s)")
    print(f"Play SR:  {play_sr} Hz | Record SR: {record_sr} Hz")
    if speaker_id is not None:
        print(f"Speaker:  device {speaker_id}")
    if mic_id is not None:
        print(f"Mic:      device {mic_id}")
    print()

    input("Press ENTER to start (make sure volume is up)...")
    print("Playing + Recording...", flush=True)

    # Start recording (non-blocking)
    recording = sd.rec(
        int(record_duration * record_sr),
        samplerate=record_sr,
        channels=1,
        dtype="float32",
        device=mic_id,
    )

    # Small delay to ensure recording has started
    time.sleep(0.1)

    # Play audio (blocking)
    sd.play(play_audio, play_sr, device=speaker_id)
    sd.wait()

    # Wait for remaining recording time
    remaining = record_duration - play_duration - 0.1
    if remaining > 0:
        time.sleep(remaining)

    sd.stop()
    recording = recording.squeeze()

    # Trim silence from start (find first sample above threshold)
    threshold = 0.005
    start_idx = 0
    for i in range(len(recording)):
        if abs(recording[i]) > threshold:
            # Back up a tiny bit to not cut the start
            start_idx = max(0, i - int(0.05 * record_sr))
            break

    recording = recording[start_idx:]

    # Save
    sf.write(output_path, recording, record_sr)
    print(f"\nSaved recording: {output_path}")
    print(f"Recording length: {len(recording)/record_sr:.2f}s")
    print(f"\nNext step: copy this file to the server and run:")
    print(f"  python ota_eval.py --audio {output_path} --target '<your target text>' --model qwen2_audio")


def record_only(output_path, mic_id=None, duration=12.0, record_sr=24000):
    import sounddevice as sd

    print(f"Duration: {duration}s | SR: {record_sr} Hz")
    if mic_id is not None:
        print(f"Mic: device {mic_id}")
    print()
    print("Play your adversarial audio from YouTube/Spotify now,")
    input("then press ENTER to start recording...")
    print(f"Recording for {duration}s...", flush=True)

    recording = sd.rec(
        int(duration * record_sr),
        samplerate=record_sr,
        channels=1,
        dtype="float32",
        device=mic_id,
    )
    sd.wait()
    recording = recording.squeeze()

    sf.write(output_path, recording, record_sr)
    print(f"\nSaved: {output_path} ({len(recording)/record_sr:.2f}s)")
    print(f"\nNext step: scp this file to the server and run:")
    print(f"  python ota_eval.py --audio {output_path} --target '<your target text>' --model qwen2_audio")


def main():
    parser = argparse.ArgumentParser(description="OTA play & record (run on laptop)")

    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--play", type=str, help="Path to adversarial WAV to play through speaker")
    parser.add_argument("--record-only", action="store_true",
                        help="Only record from mic (play from YouTube/Spotify yourself)")
    parser.add_argument("--output", "-o", type=str, default="ota_recording.wav",
                        help="Output WAV path (default: ota_recording.wav)")
    parser.add_argument("--speaker", type=int, default=None, help="Speaker device ID")
    parser.add_argument("--mic", type=int, default=None, help="Microphone device ID")
    parser.add_argument("--duration", type=float, default=12.0,
                        help="Recording duration for --record-only (default: 12s)")
    parser.add_argument("--sr", type=int, default=24000, help="Recording sample rate (default: 24000)")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.play:
        play_and_record(args.play, args.output, args.speaker, args.mic, args.sr)
    elif args.record_only:
        record_only(args.output, args.mic, args.duration, args.sr)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python ota_local_record.py --list-devices")
        print("  python ota_local_record.py --play adversarial.wav -o recorded.wav")
        print("  python ota_local_record.py --record-only -o recorded.wav --duration 12")


if __name__ == "__main__":
    main()
