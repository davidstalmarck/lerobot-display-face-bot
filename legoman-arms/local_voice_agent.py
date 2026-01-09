#!/usr/bin/env python3
"""
Local Voice Agent for SO-101 Robot
Uses: Whisper (STT) + Ollama (LLM) + pyttsx3 (TTS)
No API keys required - fully local!
"""

import whisper
import pyaudio
import wave
import tempfile
import requests
import pyttsx3
import threading
import time
from pathlib import Path

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:latest"  # Fast conversational model
CONTROL_SERVER_URL = "http://localhost:5001"  # Use simple_control_server instead

# Audio settings
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5  # Max recording length

class LocalVoiceAgent:
    def __init__(self):
        print("Initializing Local Voice Agent...")

        # Load Whisper model (tiny is fast, base is more accurate)
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("tiny")
        print("✓ Whisper loaded")

        # Initialize TTS
        print("Initializing TTS...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)  # Speed
        self.tts_engine.setProperty('volume', 0.9)
        print("✓ TTS ready")

        # Initialize audio
        self.audio = pyaudio.PyAudio()
        print("✓ Audio ready")

        # Conversation context
        self.context = []

        print("\n✓ Local Voice Agent ready!")
        print("Press SPACE to talk, or 'q' to quit")

    def record_audio(self):
        """Record audio from microphone until silence"""
        print("\n🎤 Listening... (speak now)")

        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        frames = []
        silence_threshold = 500  # Adjust based on your mic
        silence_chunks = 0
        max_silence = 20  # Stop after 20 silent chunks

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Simple silence detection
            audio_data = int.from_bytes(data[:2], byteorder='little', signed=True)
            if abs(audio_data) < silence_threshold:
                silence_chunks += 1
                if silence_chunks > max_silence:
                    break
            else:
                silence_chunks = 0

        stream.stop_stream()
        stream.close()

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return temp_file.name

    def transcribe(self, audio_file):
        """Transcribe audio using Whisper"""
        print("🔍 Transcribing...")
        result = self.whisper_model.transcribe(audio_file)
        text = result["text"].strip()
        print(f"You said: {text}")
        return text

    def chat_with_ollama(self, user_input):
        """Get response from Ollama"""
        print("🤖 Thinking...")

        # Build prompt with context
        system_prompt = """You are SO-101, a friendly robot receptionist!

Your personality: Cheerful, helpful, and quirky. Keep responses SHORT (1-2 sentences max).

You can perform gestures:
- Say "[HELLO]" when greeting to wave
- Say "[FORWARD]" when showing interest
- Say "[NEUTRAL]" to return to neutral position

Be conversational and natural!"""

        # Add context
        prompt = system_prompt + "\n\n"
        for msg in self.context[-4:]:  # Last 4 exchanges
            prompt += f"{msg}\n"
        prompt += f"Human: {user_input}\nSO-101:"

        # Call Ollama
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100  # Keep responses short
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            ai_response = response.json()["response"].strip()

            # Update context
            self.context.append(f"Human: {user_input}")
            self.context.append(f"SO-101: {ai_response}")

            return ai_response
        else:
            return "Sorry, I'm having trouble thinking right now!"

    def trigger_gesture(self, gesture_name):
        """Trigger robot gesture by playing a recording"""
        # Map gesture names to your recordings
        gesture_map = {
            "hello": "noah",  # Use your recordings
            "wave": "noah",
            "forward": "noah",
            "neutral": "noah"
        }

        recording_name = gesture_map.get(gesture_name, "noah")

        try:
            response = requests.post(
                f"{CONTROL_SERVER_URL}/play",
                json={"name": recording_name, "speed": 1.5},  # Faster playback
                timeout=2
            )
            if response.ok:
                print(f"✓ Gesture: {gesture_name}")
            else:
                print(f"⚠ Gesture failed: {response.text}")
        except Exception as e:
            print(f"⚠ Gesture failed: {e}")

    def speak(self, text):
        """Convert text to speech and trigger gestures"""
        print(f"SO-101: {text}")

        # Check for gesture commands
        if "[HELLO]" in text:
            self.trigger_gesture("hello")
            text = text.replace("[HELLO]", "")
        if "[FORWARD]" in text:
            self.trigger_gesture("forward")
            text = text.replace("[FORWARD]", "")
        if "[NEUTRAL]" in text:
            self.trigger_gesture("neutral")
            text = text.replace("[NEUTRAL]", "")

        # Speak
        self.tts_engine.say(text.strip())
        self.tts_engine.runAndWait()

    def run(self):
        """Main conversation loop"""
        # Welcome message
        self.speak("[HELLO] Hi! I'm SO-101, your robot assistant. How can I help you today?")

        print("\n=== Controls ===")
        print("Press ENTER to record and speak")
        print("Type 'q' and ENTER to quit\n")

        try:
            while True:
                # Wait for user input
                user_input = input("\n[Press ENTER to speak, or 'q' to quit]: ").strip()

                if user_input.lower() == 'q':
                    self.speak("[NEUTRAL] Goodbye! Have a great day!")
                    break

                # Record audio
                audio_file = self.record_audio()

                # Transcribe
                text = self.transcribe(audio_file)

                # Clean up temp file
                Path(audio_file).unlink()

                if not text:
                    print("(didn't catch that)")
                    continue

                # Get response
                response = self.chat_with_ollama(text)

                # Speak response
                self.speak(response)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.audio.terminate()

if __name__ == "__main__":
    agent = LocalVoiceAgent()
    agent.run()
