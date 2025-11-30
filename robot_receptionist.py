#!/usr/bin/env python3
"""
SO-101 Hackathon Receptionist Robot with Web Face

Zero2Demo Hackathon Reception Assistant

Features:
- HTML/CSS animated face display in browser
- Voice-based reception assistance via OpenAI Realtime API
- Can trigger robot gestures during conversation
- WebSocket communication between backend and face
"""

import asyncio
import websockets
import json
import pyaudio
import base64
import os
import requests
from dotenv import load_dotenv
from aiohttp import web
import webbrowser
from pathlib import Path

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in .env file!")

# Configuration
GESTURE_SERVER_URL = "http://localhost:8080"
WEB_PORT = 3000
WEBSOCKET_PORT = 8765

# Audio configuration
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

class GestureAPI:
    """Interface to robot gesture server"""

    @staticmethod
    async def call_gesture(gesture_name):
        """Call a gesture on the robot"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{GESTURE_SERVER_URL}/{gesture_name}", timeout=2)
            )
            print(f"✓ Triggered gesture: {gesture_name}")
            return response.json()
        except Exception as e:
            print(f"✗ Error calling gesture {gesture_name}: {e}")
            return None

class ReceptionistBot:
    """Main receptionist bot with web face and voice"""

    def __init__(self):
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        self.openai_ws = None
        self.face_ws_clients = set()
        self.is_recording = False
        self.current_state = "idle"

    async def broadcast_state(self, state, audio_level=0):
        """Broadcast state to all connected web face clients"""
        self.current_state = state
        message = json.dumps({"state": state, "audio_level": audio_level})

        # Send to all connected clients
        if self.face_ws_clients:
            await asyncio.gather(
                *[client.send(message) for client in self.face_ws_clients],
                return_exceptions=True
            )

    async def face_websocket_handler(self, websocket):
        """Handle WebSocket connections from web face (websockets v14+)"""
        # If you need the path, it lives on the websocket object now
        path = getattr(websocket, "path", None)

        self.face_ws_clients.add(websocket)
        print(f"Face connected{f' on {path}' if path else ''}. Total clients: {len(self.face_ws_clients)}")

        try:
            # Send current state
            await websocket.send(json.dumps({"state": self.current_state}))

            # Keep connection alive
            async for message in websocket:
                pass
        finally:
            self.face_ws_clients.remove(websocket)
            print(f"Face disconnected. Total clients: {len(self.face_ws_clients)}")


    def initialize_audio(self):
        """Initialize PyAudio"""
        print("Initializing audio...")
        self.audio = pyaudio.PyAudio()

        self.input_stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        self.output_stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )
        print("Audio ready!")

    async def connect_openai(self):
        """Connect to OpenAI Realtime API with function calling"""
        print("Connecting to OpenAI Realtime API...")

        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        self.openai_ws = await websockets.connect(url, additional_headers=headers)
        print("Connected to OpenAI!")

        # Configure session with function calling for gestures
        hackathon_instructions = """You are SO-101, a friendly robot receptionist at the Zero2Demo hackathon!

Your role:
- Welcome attendees warmly and enthusiastically
- Provide information about the hackathon
- You can perform physical gestures! Use the available functions when appropriate:
  - do_hello_gesture: Wave hello when greeting someone
  - do_forward_gesture: Lean forward when showing interest or curiosity
  - return_to_neutral: Return to neutral position after gestures

Personality: Cheerful, helpful, quirky, and encouraging. Keep responses SHORT (1-2 sentences).

Key Info:
- Event: Zero2Demo Hackathon
- Focus: Building demos and prototypes
- Check-in: 9 AM | Demos: Afternoon
- Prizes available | Food & Wi-Fi provided

Use gestures naturally in conversation to be more expressive and engaging!"""

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": hackathon_instructions,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "do_hello_gesture",
                        "description": "Make the robot wave hello. Use when greeting someone or saying hi.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "type": "function",
                        "name": "do_forward_gesture",
                        "description": "Make the robot lean forward curiously. Use when showing interest or being inquisitive.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "type": "function",
                        "name": "return_to_neutral",
                        "description": "Return robot to neutral position. Use after performing gestures.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                ],
                "tool_choice": "auto"
            }
        }

        await self.openai_ws.send(json.dumps(session_config))

    async def send_audio(self):
        """Send audio from microphone to OpenAI"""
        try:
            while self.is_recording:
                audio_data = self.input_stream.read(CHUNK, exception_on_overflow=False)
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }

                await self.openai_ws.send(json.dumps(message))
                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Error sending audio: {e}")

    async def handle_function_call(self, function_name, call_id):
        """Handle function calls from OpenAI to trigger gestures"""
        print(f"🤖 Function called: {function_name}")

        result = None
        if function_name == "do_hello_gesture":
            result = await GestureAPI.call_gesture("hello")
        elif function_name == "do_forward_gesture":
            result = await GestureAPI.call_gesture("forward")
        elif function_name == "return_to_neutral":
            result = await GestureAPI.call_gesture("neutral")

        # Send function result back to OpenAI
        response = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": result is not None})
            }
        }
        await self.openai_ws.send(json.dumps(response))

        # Trigger response generation
        await self.openai_ws.send(json.dumps({"type": "response.create"}))

    async def receive_responses(self):
        """Receive responses from OpenAI"""
        try:
            async for message in self.openai_ws:
                data = json.loads(message)
                event_type = data.get("type")

                if event_type == "response.audio.delta":
                    # Play audio
                    audio_b64 = data.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        self.output_stream.write(audio_data)
                        await self.broadcast_state("speaking", audio_level=5000)

                elif event_type == "response.audio.done":
                    await self.broadcast_state("idle")

                elif event_type == "input_audio_buffer.speech_started":
                    await self.broadcast_state("listening")

                elif event_type == "response.created":
                    await self.broadcast_state("thinking")

                elif event_type == "response.function_call_arguments.done":
                    # Function call detected
                    function_name = data.get("name")
                    call_id = data.get("call_id")
                    await self.handle_function_call(function_name, call_id)

                elif event_type == "error":
                    print(f"OpenAI Error: {data}")

        except Exception as e:
            print(f"Error receiving responses: {e}")

    async def start_websocket_server(self):
        """Start WebSocket server for face"""
        server = await websockets.serve(
            self.face_websocket_handler,
            "localhost",
            WEBSOCKET_PORT
        )
        print(f"WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
        return server

    async def start_web_server(self):
        """Start web server for face HTML"""
        app = web.Application()

        # Serve static files
        static_dir = Path(__file__).parent / "static"

        async def index(request):
            """Serve index.html"""
            index_path = static_dir / "index.html"
            return web.FileResponse(index_path)

        app.router.add_get('/', index)
        app.router.add_static('/static/', static_dir, name='static')

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', WEB_PORT)
        await site.start()

        print(f"Web server running on http://localhost:{WEB_PORT}")
        print(f"Open http://localhost:{WEB_PORT} to see the face!")

        return runner

    async def run(self):
        """Main run loop"""
        print("\n" + "=" * 70)
        print("SO-101 Hackathon Receptionist")
        print("=" * 70)

        # Start servers
        ws_server = await self.start_websocket_server()
        web_runner = await self.start_web_server()

        # Initialize audio
        self.initialize_audio()

        # Connect to OpenAI if key available
        if OPENAI_API_KEY:
            await self.connect_openai()
            print("\n✓ All systems ready!")
            print("\nControls:")
            print("  - Say 'hello' to start a conversation")
            print("  - I can wave, lean forward, and more!")
            print("  - Press Ctrl+C to quit")
            print("=" * 70)

            # Open browser
            webbrowser.open(f"http://localhost:{WEB_PORT}")

            # Start receiving
            tasks = [
                asyncio.create_task(self.receive_responses()),
            ]

            # Auto-start recording for voice activation
            self.is_recording = True
            tasks.append(asyncio.create_task(self.send_audio()))

            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                print("\n\nShutting down...")
        else:
            print("Running without OpenAI (face display only)")
            print(f"Open http://localhost:{WEB_PORT} to see the face")
            try:
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                print("\n\nShutting down...")

        # Cleanup
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.audio:
            self.audio.terminate()
        if self.openai_ws:
            await self.openai_ws.close()

        ws_server.close()
        await ws_server.wait_closed()
        await web_runner.cleanup()

        print("Goodbye!")


if __name__ == "__main__":
    bot = ReceptionistBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
