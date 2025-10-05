import os
import asyncio
import io
import traceback
import json
import time
import datetime

import cv2
import pyaudio
import PIL.Image
import mss
import numpy as np

import argparse

from google import genai
from google.genai import types
from face_reid import FaceReID


# --- DEBUG FLAG ---
# Set to True to enable detailed logging, False to disable.
DEBUG = True

def DEBUG_PRINT(message):
    """Helper function for debug printing with timestamps."""
    if DEBUG:
        print(f"[{datetime.datetime.now().isoformat()}] [DEBUG] {message}")

# Try to import WebRTC VAD
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    DEBUG_PRINT("WebRTC VAD not available. Voice detection will be volume-based only.")

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# NOTE: The model name in the original script was for a preview version.
# Using a standard, available model is recommended for stability.
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

DEFAULT_MODE = "camera"

# It's better practice to handle API key errors gracefully
#api_key = os.getenv('GEMINI_API_KEY')
#if not api_key:
    #raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key="", http_options={"api_version": "v1alpha"})

def Give_Sample():
    """Dummy function for the tool call."""
    DEBUG_PRINT("Executing tool: Give_Sample\n\n____________________\n\n")
    return {"status": "Sample delivered successfully", "item": "nut bar", 'name': 'done'}

tools = [
    {"function_declarations": [
        {"name": "Give_Sample", "description": "By calling this function, you open a hatch that deposits one sample."}
    ]}
]

CONFIG = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": "Zephyr"}
        }
    },
    "proactivity": {'proactive_audio': True},
    "system_instruction": """Your role is a food server at Costco who hands out samples.

You will receive continuous audio and video feeds. Based on what you see and hear:

- Proactively greet customers who approach or look interested in samples
- If someone is speaking to you, respond appropriately
- Each person should only get ONE sample
- The sample is a nut bar
- Be friendly and conversational, but keep responses concise
- You can see people even when they're not talking - feel free to initiate conversation!

You may also receive messages tagged with [Operator Guidance]. Treat these as internal instructions: think through them silently, decide whether customer-facing action is required, and only speak when you choose to engage. Never repeat guidance verbatim to customers.""",
    "tools": tools,
}

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        DEBUG_PRINT(f"Initializing AudioLoop with video_mode: {video_mode}")
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.output_stream = None
        self.running = True
        
        # WebRTC VAD setup
        self.vad = None
        if WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Slightly more sensitive to speech
            DEBUG_PRINT("WebRTC VAD initialized.")
        
        # Echo prevention state
        self.is_ai_speaking = False
        self.ai_stop_time = 0
        self.min_delay_after_ai = 0.5  # 500ms delay to prevent feedback
        
        # Timers for continuous data sending
        self.last_video_send = 0
        self.base_video_send_interval = 5.0  # ~0.2 FPS when idle
        self.active_video_send_interval = 1.0  # ~1 FPS shortly after motion
        self.motion_state_decay = 3.0  # seconds staying in active mode after motion
        self.video_send_interval = self.base_video_send_interval
        self.last_audio_send = 0
        self.ambient_audio_interval = 1.0  # How often to send ambient sound
        self.speaking_audio_interval = 0.3  # How often to send audio when someone is talking
        self.min_send_bytes = int(SEND_SAMPLE_RATE * 0.25 * 2)  # Min audio data to send immediately
        
        # Voice detection state
        self.min_volume_threshold = 0.0035
        self.noise_floor = self.min_volume_threshold / 2
        self.dynamic_threshold_ratio = 1.6
        self.is_person_speaking = False

        # Visual proactivity state
        self.last_frame_gray = None
        self.motion_pixel_threshold = 25
        self.motion_trigger_ratio = 0.02
        self.motion_cooldown = 0.25
        self.last_motion_event_time = 0.0
        self.last_proactive_prompt_time = 0.0
        self.proactive_prompt_cooldown = 0.25

        self.reid = FaceReID(max_age_s=600, match_threshold=0.42)  # 10 min memory window
        
        if WEBRTC_AVAILABLE:
            self.frame_duration_ms = 30
            self.frame_size = int(SEND_SAMPLE_RATE * self.frame_duration_ms / 1000)
            self.audio_buffer = bytearray()
        
        DEBUG_PRINT("AudioLoop initialized successfully.")

    async def trigger_proactive_prompt(self, text):
        """Send a brief proactive hint to the model when we detect visual events."""
        if not self.session:
            DEBUG_PRINT("No active session; skipping proactive prompt.")
            return

        now = time.time()
        if now - self.last_proactive_prompt_time < self.proactive_prompt_cooldown:
            DEBUG_PRINT("Skipping proactive prompt due to cooldown.")
            return

        self.last_proactive_prompt_time = now
        prompt_text = f"[Operator Guidance] {text}"
        DEBUG_PRINT(f"Triggering proactive prompt: {prompt_text}")
        try:
            await self.session.send_client_content(
                turns={
                    "parts": [
                        {
                            "text": prompt_text
                        }
                    ]
                }
            )
        except Exception as e:
            DEBUG_PRINT(f"Failed to send proactive prompt: {e}")

    def calculate_volume(self, audio_data):
        """Calculate RMS volume of audio data."""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_np) == 0: return 0.0
            rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
            return rms / 32768.0
        except Exception as e:
            DEBUG_PRINT(f"Error calculating volume: {e}")
            return 0.0

    def is_voice_detected(self, audio_data):
        """Detect voice using volume and/or WebRTC VAD."""
        volume = self.calculate_volume(audio_data)

        # Update noise floor using exponential moving average when AI isn't talking
        if not self.is_ai_speaking:
            alpha = 0.05
            self.noise_floor = max(self.min_volume_threshold / 2,
                                   (1 - alpha) * self.noise_floor + alpha * volume)

        adaptive_threshold = max(self.min_volume_threshold,
                                 self.dynamic_threshold_ratio * self.noise_floor)
        volume_detected = volume > adaptive_threshold
        
        webrtc_detected = False
        if self.vad and WEBRTC_AVAILABLE:
            try:
                self.audio_buffer.extend(audio_data)
                while len(self.audio_buffer) >= self.frame_size * 2:
                    frame = self.audio_buffer[:self.frame_size * 2]
                    self.audio_buffer = self.audio_buffer[self.frame_size * 2:]
                    if self.vad.is_speech(frame, SEND_SAMPLE_RATE):
                        webrtc_detected = True
                        break
            except Exception as e:
                DEBUG_PRINT(f"Error in WebRTC VAD processing: {e}")
        
        return volume_detected or webrtc_detected

    async def send_text(self):
        """Task to send user-typed text messages to the model."""
        DEBUG_PRINT("Starting send_text task.")
        while self.running:
            try:
                text = await asyncio.to_thread(input, "message > ")
                if text.lower() == "q":
                    DEBUG_PRINT("User requested exit with 'q'.")
                    self.running = False
                    break
                DEBUG_PRINT(f"Sending text to model: '{text}'")
                await self.session.send_client_content(
                    turns={
                        "parts": [
                            {
                                "text": text if text else "."
                            }
                        ]
                    }
                )
            except EOFError:
                break
        DEBUG_PRINT("send_text task finished.")

    def _choose_focus_pid(self, reid_results, width, height):
        if not reid_results:
            self.current_focus_pid = None
            return
        cx0, cy0 = width / 2.0, height / 2.0
        best_pid, best_d = None, 1e18
        for r in reid_results:
            x1, y1, x2, y2 = r["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = (cx - cx0) ** 2 + (cy - cy0) ** 2
            if d < best_d:
                best_d, best_pid = d, r["pid"]
        self.current_focus_pid = best_pid

    def _get_frame(self, cap, detect_motion=False):
        """Helper to capture and process one camera frame.
        Returns: (blob, motion_detected, guidance_text)
        """
        ret, frame = cap.read()
        if not ret:
            return (None, False, None) if detect_motion else (None, None, None)

        motion_detected = False
        guidance_msgs = []

        # --- Motion detection (thread-safe; no asyncio here) ---
        if detect_motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.last_frame_gray is None:
                self.last_frame_gray = gray
            else:
                frame_delta = cv2.absdiff(self.last_frame_gray, gray)
                thresh = cv2.threshold(frame_delta, self.motion_pixel_threshold, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                motion_ratio = np.sum(thresh > 0) / thresh.size
                if motion_ratio > self.motion_trigger_ratio and (time.time() - self.last_motion_event_time) > self.motion_cooldown:
                    motion_detected = True
                    self.last_motion_event_time = time.time()
                    # don't schedule here—collect a message to return
                    guidance_msgs.append(
                        "Motion observed near the counter by generic image motion detector. "
                        "See if there are people. Evaluate what to do depending on conversational context."
                    )
                self.last_frame_gray = gray

        # --- ReID & focus selection (full-frame pipeline) ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        reid_results = self.reid.process_frame(frame_rgb)
        self._choose_focus_pid(reid_results, W, H)

        # Compose internal guidance based on ReID state
        interesting = []
        for r in reid_results:
            pid = r["pid"]
            if r["served"]:
                interesting.append(f"Person #{pid} appears already served.")
            else:
                person = self.reid.people.get(pid, {})
                if person.get("seen_count", 0) == 1:
                    interesting.append(f"New person near counter: #{pid}.")

        if interesting:
            guidance_msgs.append(
                " ".join(interesting) + " Greet those who seem interested; avoid offering duplicates."
            )

        # --- Optional debug drawing (local only) ---
        for r in reid_results:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{r['pid']}{'✓' if r['served'] else ''}"
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



        # --- Encode & return ---
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([600, 338])
        with io.BytesIO() as image_io:
            img.save(image_io, format="jpeg", quality=80)
            blob = types.Blob(mime_type="image/jpeg", data=image_io.getvalue())

        guidance_text = " ".join(guidance_msgs) if guidance_msgs else None
        return (blob, motion_detected, guidance_text) if detect_motion else (blob, None, guidance_text)


    async def get_frames(self):
        """Task to periodically capture camera frames (schedules prompts on main loop)."""
        DEBUG_PRINT("Starting get_frames (camera) task.")
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        if not cap.isOpened():
            DEBUG_PRINT("Error: Could not open camera.")
            self.running = False
            return

        # optional warm-up reads
        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                break
            await asyncio.sleep(0.05)

        while self.running:
            if time.time() - self.last_video_send >= self.video_send_interval:
                DEBUG_PRINT("Capturing camera frame.")
                blob, motion_detected, guidance_text = await asyncio.to_thread(self._get_frame, cap, True)

                if blob:
                    if self.out_queue.full():
                        try:
                            _ = self.out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.out_queue.put(blob)
                    self.last_video_send = time.time()

                if motion_detected:
                    self.last_motion_event_time = time.time()
                    self.video_send_interval = self.active_video_send_interval
                elif (time.time() - self.last_motion_event_time) > self.motion_state_decay:
                    self.video_send_interval = self.base_video_send_interval

                # Schedule the proactive prompt here (on the real event loop)
                if guidance_text and (time.time() - self.last_proactive_prompt_time) > self.proactive_prompt_cooldown:
                    asyncio.create_task(self.trigger_proactive_prompt(guidance_text))

            await asyncio.sleep(0.05)

        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        DEBUG_PRINT("get_frames (camera) task finished.")


    def _get_screen(self):
        """Helper to capture and process one screen grab."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img.thumbnail([1024, 1024])

            with io.BytesIO() as image_io:
                img.save(image_io, format="jpeg", quality=80)
                return types.Blob(mime_type="image/jpeg", data=image_io.getvalue())

    async def get_screen(self):
        """Task to periodically capture the screen."""
        DEBUG_PRINT("Starting get_screen task.")
        while self.running:
            if time.time() - self.last_video_send >= self.video_send_interval:
                DEBUG_PRINT("Capturing screen frame.")
                blob = await asyncio.to_thread(self._get_screen)
                if blob:
                    if self.out_queue.full():
                        try:
                            _ = self.out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.out_queue.put(blob)
                    self.last_video_send = time.time()
            await asyncio.sleep(0.05)
        DEBUG_PRINT("get_screen task finished.")

    async def send_realtime(self):
        """Task to send queued data (audio/video) to the model."""
        DEBUG_PRINT("Starting send_realtime task.")
        while self.running:
            try:
                blob = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                if blob.mime_type.startswith("audio"):
                    DEBUG_PRINT(f"Sending audio data ({len(blob.data)} bytes).")
                    await self.session.send_realtime_input(audio=blob)
                elif blob.mime_type.startswith("image"):
                    DEBUG_PRINT(f"Sending image data ({len(blob.data)} bytes). Categorized as: video frame")
                    await self.session.send_realtime_input(video=blob)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                DEBUG_PRINT(f"Error in send_realtime: {e}")
        DEBUG_PRINT("send_realtime task finished.")

    async def listen_audio(self):
        """Task for continuous audio monitoring and smart sending."""
        DEBUG_PRINT("Starting listen_audio task.")
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
            input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE
        )
        DEBUG_PRINT("Microphone stream opened.")
        
        audio_accumulator = bytearray()
        while self.running:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, False)
                current_time = time.time()
                can_send = not self.is_ai_speaking and (current_time - self.ai_stop_time > self.min_delay_after_ai)

                if not can_send:
                    if len(audio_accumulator) > 0:
                        DEBUG_PRINT("Dropping buffered audio while AI is speaking.")
                    audio_accumulator.clear()
                    if WEBRTC_AVAILABLE and hasattr(self, "audio_buffer"):
                        self.audio_buffer.clear()
                    if self.is_person_speaking:
                        self.is_person_speaking = False
                    await asyncio.sleep(0.05)
                    continue

                audio_accumulator.extend(data)
                voice_detected = self.is_voice_detected(data)

                if voice_detected and not self.is_person_speaking:
                    self.is_person_speaking = True
                    DEBUG_PRINT("Person started speaking. Categorized as: speaking")
                    if len(audio_accumulator) >= self.min_send_bytes and can_send:
                        DEBUG_PRINT("Immediate send on speech start.")
                        await self.out_queue.put(types.Blob(data=bytes(audio_accumulator), mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"))
                        audio_accumulator.clear()
                        self.last_audio_send = current_time

                elif not voice_detected and self.is_person_speaking:
                    self.is_person_speaking = False
                    DEBUG_PRINT("Person stopped speaking. Categorized as: not speaking")
                    if len(audio_accumulator) > 0 and can_send:
                        DEBUG_PRINT("Sending final audio chunk.")
                        await self.out_queue.put(types.Blob(data=bytes(audio_accumulator), mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"))
                        audio_accumulator.clear()
                        self.last_audio_send = current_time

                send_interval = self.speaking_audio_interval if self.is_person_speaking else self.ambient_audio_interval
                if current_time - self.last_audio_send >= send_interval and len(audio_accumulator) > 0 and can_send:
                    DEBUG_PRINT(f"Periodic send ({'speaking' if self.is_person_speaking else 'ambient'}).")
                    await self.out_queue.put(types.Blob(data=bytes(audio_accumulator), mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"))
                    audio_accumulator.clear()
                    self.last_audio_send = current_time
                
            except Exception as e:
                DEBUG_PRINT(f"Error in listen_audio loop: {e}")
        DEBUG_PRINT("listen_audio task finished.")

    async def receive_audio(self):
        """Task to handle all incoming responses from the model."""
        DEBUG_PRINT("Starting receive_audio task.")
        while self.running:
            try:
                async for chunk in self.session.receive():
                    if hasattr(chunk, "data") and chunk.data:
                        await self.audio_in_queue.put(chunk.data)

                    if hasattr(chunk, "text") and chunk.text:
                        print(f"\n[Assistant]: {chunk.text}")
                        DEBUG_PRINT(f"Received text from model: '{chunk.text}'")

                    if hasattr(chunk, "tool_call") and hasattr(chunk.tool_call, "function_calls"):
                        for fc in chunk.tool_call.function_calls:
                            print(f"\n[Tool Call]: {fc.name}")
                            DEBUG_PRINT(f"Model called tool: {fc.name} with ID: {fc.id}")
                            if fc.name == "Give_Sample":
                                result = Give_Sample()
                                # Pick person with highest seen_count in the last ~2s as the focus
                                # Prefer the current focus; fallback to most-seen if no focus
                                pid_to_mark = getattr(self, "current_focus_pid", None)
                                if pid_to_mark is None and self.reid.people:
                                    pid_to_mark = max(self.reid.people.items(), key=lambda kv: kv[1]["seen_count"])[0]

                                if pid_to_mark is not None:
                                    self.reid.mark_served(pid_to_mark)
                                    DEBUG_PRINT(f"Marked person #{pid_to_mark} as served.")
                                else:
                                    DEBUG_PRINT("No focus PID available to mark as served.")

                                fr = types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                                DEBUG_PRINT(f"Sending tool response for {fc.name}: {result}")
                                await self.session.send_tool_response(function_responses=[fr])
            except Exception as e:
                DEBUG_PRINT(f"Error in receive_audio: {e}")
        DEBUG_PRINT("receive_audio task finished.")

    async def play_audio(self):
        """Task to play audio received from the model."""
        DEBUG_PRINT("Starting play_audio task.")
        self.output_stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE,
            output=True, frames_per_buffer=CHUNK_SIZE * 4
        )
        DEBUG_PRINT("Audio output stream opened.")
        buffered = bytearray()
        while self.running:
            try:
                bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
                if not self.is_ai_speaking:
                    self.is_ai_speaking = True
                    DEBUG_PRINT("AI started speaking. Recording is paused.")
                buffered.extend(bytestream)

                if len(buffered) >= int(RECEIVE_SAMPLE_RATE * 0.1 * 2):
                    await asyncio.to_thread(self.output_stream.write, bytes(buffered))
                    buffered.clear()
            except asyncio.TimeoutError:
                if self.is_ai_speaking:
                    if len(buffered) > 0:
                        await asyncio.to_thread(self.output_stream.write, bytes(buffered))
                        buffered.clear()
                    self.is_ai_speaking = False
                    self.ai_stop_time = time.time()
                    DEBUG_PRINT(f"AI stopped speaking. Recording will resume after {self.min_delay_after_ai}s delay.")
            except Exception as e:
                DEBUG_PRINT(f"Error in play_audio: {e}")
        DEBUG_PRINT("play_audio task finished.")

    async def cleanup(self):
        """Clean up all resources."""
        DEBUG_PRINT("Starting cleanup.")
        self.running = False
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            DEBUG_PRINT("Input audio stream closed.")
        if self.output_stream and self.output_stream.is_active():
            self.output_stream.stop_stream()
            self.output_stream.close()
            DEBUG_PRINT("Output audio stream closed.")
        DEBUG_PRINT("Cleanup finished.")

    async def run(self):
        """Main execution function."""
        try:
            DEBUG_PRINT(f"Connecting to model: {MODEL}")
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session, \
                       asyncio.TaskGroup() as tg:
                DEBUG_PRINT("Model session started.")
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=50)

                tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                DEBUG_PRINT("All tasks created.")
        except Exception as e:
            DEBUG_PRINT(f"An exception occurred in run: {e}")
            traceback.print_exc()
        finally:
            await self.cleanup()
            pya.terminate()
            DEBUG_PRINT("PyAudio terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        help="pixels to stream from", choices=["camera", "screen", "none"])
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    
    try:
        DEBUG_PRINT("Starting application run loop.")
        asyncio.run(main.run())
    except KeyboardInterrupt:
        DEBUG_PRINT("KeyboardInterrupt received. Shutting down.")
    finally:
        DEBUG_PRINT("Application finished.")
