import os
import asyncio
import io
import traceback
import json
import time
import datetime
import logging
from logging.handlers import RotatingFileHandler

import cv2
import pyaudio
import PIL.Image
import mss
import numpy as np
import argparse

from google import genai
from google.genai import types


# --- DEBUG FLAG ---
DEBUG = True

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR, f"integrated_face_nomemory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def setup_logging():
    logger = logging.getLogger("integrated_face_nomemory")
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    logger.debug("Logging initialized. Log file: %s", LOG_FILE)
    return logger


LOGGER = setup_logging()


def DEBUG_PRINT(message, level=logging.DEBUG):
    """Helper function for debug printing with timestamps."""
    LOGGER.log(level, message)


def _convert_for_log(value, depth=0):
    if depth > 2:
        return "<truncated>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, (list, tuple)):
        return [_convert_for_log(v, depth + 1) for v in value[:50]]
    if isinstance(value, dict):
        return {str(k): _convert_for_log(v, depth + 1) for k, v in list(value.items())[:50]}
    if hasattr(value, "mime_type") and hasattr(value, "data"):
        return {"type": "Blob", "mime_type": value.mime_type, "length": len(value.data)}
    return str(value)


def log_event(event_name, **details):
    payload = {
        "event": event_name,
        "details": {k: _convert_for_log(v) for k, v in details.items()}
    }
    LOGGER.info("EVENT %s", json.dumps(payload, ensure_ascii=False))

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

MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
DEFAULT_MODE = "camera"

client = genai.Client(api_key="", http_options={"api_version": "v1alpha"})


def Give_Sample():
    """Tool function for giving a sample."""
    DEBUG_PRINT("Executing tool: Give_Sample")
    log_event("tool_called", tool="Give_Sample")

    print("-----------------------------------> Giving sample")

    return {
        "status": "Sample delivered successfully",
        "item": "nut bar",
        "name": "done",
    }


pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        DEBUG_PRINT(f"Initializing AudioLoop (no memory) with video_mode: {video_mode}")
        log_event("audio_loop_init", video_mode=video_mode)
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.output_stream = None
        self.running = True
        log_event("audio_loop_state", step="init_complete")

        # WebRTC VAD setup
        self.vad = None
        if WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(2)
            DEBUG_PRINT("WebRTC VAD initialized.")
            log_event("vad_initialized", mode=2)

        # Echo prevention state
        self.is_ai_speaking = False
        self.ai_stop_time = 0
        self.min_delay_after_ai = 0.5

        # Timers for continuous data sending
        self.last_video_send = 0
        self.base_video_send_interval = 5.0
        self.active_video_send_interval = 2.0
        self.motion_state_decay = 5.0
        self.video_send_interval = self.base_video_send_interval
        self.last_audio_send = 0
        self.ambient_audio_interval = 1.0
        self.speaking_audio_interval = 0.3
        self.min_send_bytes = int(SEND_SAMPLE_RATE * 0.12 * 2)

        # Voice detection state
        self.min_volume_threshold = 0.0025
        self.noise_floor = self.min_volume_threshold / 2
        self.dynamic_threshold_ratio = 1.3
        self.is_person_speaking = False

        # Visual proactivity state
        self.last_frame_gray = None
        self.motion_pixel_threshold = 25
        self.motion_trigger_ratio = 0.045
        self.motion_cooldown = 4.0
        self.last_motion_event_time = 0.0
        self.last_proactive_prompt_time = 0.0
        self.proactive_prompt_cooldown = 5.0
        self.last_proactive_prompt_text = None

        if WEBRTC_AVAILABLE:
            self.frame_duration_ms = 30
            self.frame_size = int(SEND_SAMPLE_RATE * self.frame_duration_ms / 1000)
            self.audio_buffer = bytearray()

        # Only Give_Sample tool is available without memory
        self.tools = [
            {"function_declarations": [
                {
                    "name": "Give_Sample",
                    "description": "By calling this function, you open a hatch that deposits one sample to the person you're currently talking to, and they take the sample."
                }
            ]}
        ]

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

        readable_now = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        self.last_proactive_prompt_time = now
        prompt_text = f"[Operator Guidance] @ epoch {now:.0f} (local {readable_now}) {text}"
        DEBUG_PRINT(f"Triggering proactive prompt: {prompt_text}")
        log_event("proactive_prompt", prompt=prompt_text)
        try:
            turns_payload = {
                "parts": [
                    {
                        "text": prompt_text
                    }
                ]
            }
            DEBUG_PRINT(
                f"Sending client content to model (proactive): {json.dumps(turns_payload, ensure_ascii=False)}"
            )
            log_event("llm_text_outbound", source="proactive_prompt", payload=turns_payload)
            await self.session.send_client_content(turns=turns_payload)
        except Exception as e:
            DEBUG_PRINT(f"Failed to send proactive prompt: {e}")

    def calculate_volume(self, audio_data):
        """Calculate RMS volume of audio data."""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_np) == 0:
                return 0.0
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            return rms / 32768.0
        except Exception as e:
            DEBUG_PRINT(f"Error calculating volume: {e}")
            return 0.0

    def is_voice_detected(self, audio_data):
        """Detect voice using volume and/or WebRTC VAD."""
        volume = self.calculate_volume(audio_data)

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
                    log_event("user_exit_requested")
                    self.running = False
                    break
                DEBUG_PRINT(f"Sending text to model: '{text}'")
                log_event("user_text_input", text=text)
                turns_payload = {
                    "parts": [
                        {
                            "text": text if text else "."
                        }
                    ]
                }
                DEBUG_PRINT(
                    f"Sending client content to model (user input): {json.dumps(turns_payload, ensure_ascii=False)}"
                )
                log_event("llm_text_outbound", source="user_input", payload=turns_payload)
                await self.session.send_client_content(turns=turns_payload)
            except EOFError:
                break
        DEBUG_PRINT("send_text task finished.")

    def _get_frame(self, cap, detect_motion=False):
        """Helper to capture and process one camera frame."""
        ret, frame = cap.read()
        if not ret:
            return (None, False, None, None) if detect_motion else (None, None, None, None)

        motion_detected = False
        guidance_msgs = []

        # Motion detection
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
                    guidance_msgs.append(
                        "Motion observed near the counter. Someone might be approaching. Respond according to their intent."
                    )
                self.last_frame_gray = gray

        # Encode & return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([600, 338])
        with io.BytesIO() as image_io:
            img.save(image_io, format="jpeg", quality=80)
            blob = types.Blob(mime_type="image/jpeg", data=image_io.getvalue())

        guidance_text = "\n".join(guidance_msgs) if guidance_msgs else None
        if detect_motion:
            return blob, motion_detected, guidance_text, display_frame
        return blob, None, guidance_text, display_frame

    async def get_frames(self):
        """Task to periodically capture camera frames."""
        DEBUG_PRINT("Starting get_frames (camera) task.")
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        if not cap.isOpened():
            DEBUG_PRINT("Error: Could not open camera.")
            self.running = False
            return

        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                break
            await asyncio.sleep(0.05)

        while self.running:
            if time.time() - self.last_video_send >= self.video_send_interval:
                DEBUG_PRINT("Capturing camera frame.")
                blob, motion_detected, guidance_text, display_frame = await asyncio.to_thread(self._get_frame, cap, True)

                if blob:
                    log_event("camera_frame_captured", size=len(blob.data))
                    if self.out_queue.full():
                        try:
                            _ = self.out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.out_queue.put(blob)
                    self.last_video_send = time.time()

                if display_frame is not None:
                    try:
                        cv2.imshow("Model Feed", display_frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        DEBUG_PRINT(f"Error displaying camera frame: {e}")

                if motion_detected:
                    self.last_motion_event_time = time.time()
                    self.video_send_interval = self.active_video_send_interval
                    log_event("motion_detected")
                elif (time.time() - self.last_motion_event_time) > self.motion_state_decay:
                    self.video_send_interval = self.base_video_send_interval

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
            monitor = sct.monitors[1]
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
                    log_event("screen_frame_captured", size=len(blob.data))
                    if self.out_queue.full():
                        try:
                            _ = self.out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.out_queue.put(blob)
                    self.last_video_send = time.time()

                    try:
                        np_frame = np.frombuffer(blob.data, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            cv2.imshow("Model Feed", frame_bgr)
                            cv2.waitKey(1)
                    except Exception as e:
                        DEBUG_PRINT(f"Error displaying screen frame: {e}")
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
                    log_event("audio_blob_sent", bytes=len(blob.data))
                    await self.session.send_realtime_input(audio=blob)
                elif blob.mime_type.startswith("image"):
                    DEBUG_PRINT(f"Sending image data ({len(blob.data)} bytes).")
                    log_event("image_blob_sent", bytes=len(blob.data))
                    await self.session.send_realtime_input(video=blob)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                DEBUG_PRINT(f"Error in send_realtime: {e}")
                log_event("send_realtime_error", error=str(e))
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
        log_event("microphone_opened", device=mic_info.get("name"))

        audio_accumulator = bytearray()
        while self.running:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, False)
                current_time = time.time()
                can_send = not self.is_ai_speaking and (current_time - self.ai_stop_time > self.min_delay_after_ai)
                log_event(
                    "audio_chunk_read",
                    bytes=len(data),
                    can_send=can_send,
                    ai_speaking=self.is_ai_speaking,
                    delay_remaining=max(0.0, self.min_delay_after_ai - (current_time - self.ai_stop_time))
                    if self.is_ai_speaking else 0.0
                )

                if not can_send:
                    if len(audio_accumulator) > 0:
                        DEBUG_PRINT("Dropping buffered audio while AI is speaking.")
                        log_event("audio_dropped", reason="ai_speaking", bytes=len(audio_accumulator))
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
                    DEBUG_PRINT("Person started speaking.")
                    log_event("person_speaking_state", state="started")
                    if len(audio_accumulator) >= self.min_send_bytes and can_send:
                        DEBUG_PRINT("Immediate send on speech start.")
                        log_event("audio_blob_sent", bytes=len(audio_accumulator), trigger="speech_start")
                        await self.out_queue.put(types.Blob(data=bytes(audio_accumulator), mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"))
                        audio_accumulator.clear()
                        self.last_audio_send = current_time

                elif not voice_detected and self.is_person_speaking:
                    self.is_person_speaking = False
                    DEBUG_PRINT("Person stopped speaking.")
                    log_event("person_speaking_state", state="stopped")
                    if len(audio_accumulator) > 0 and can_send:
                        DEBUG_PRINT("Sending final audio chunk.")
                        log_event("audio_blob_sent", bytes=len(audio_accumulator), trigger="speech_end")
                        await self.out_queue.put(types.Blob(data=bytes(audio_accumulator), mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"))
                        audio_accumulator.clear()
                        self.last_audio_send = current_time

                send_interval = self.speaking_audio_interval if self.is_person_speaking else self.ambient_audio_interval
                if current_time - self.last_audio_send >= send_interval and len(audio_accumulator) > 0 and can_send:
                    DEBUG_PRINT(f"Periodic send ({'speaking' if self.is_person_speaking else 'ambient'}).")
                    log_event("audio_blob_sent", bytes=len(audio_accumulator), trigger="periodic")
                    await self.out_queue.put(types.Blob(data=bytes(audio_accumulator), mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"))
                    audio_accumulator.clear()
                    self.last_audio_send = current_time

            except Exception as e:
                DEBUG_PRINT(f"Error in listen_audio loop: {e}")
                log_event("listen_audio_error", error=str(e))
        DEBUG_PRINT("listen_audio task finished.")
        log_event("listen_audio_stopped")

    async def receive_audio(self):
        """Task to handle all incoming responses from the model."""
        DEBUG_PRINT("Starting receive_audio task.")
        log_event("receive_audio_started")
        while self.running:
            try:
                async for chunk in self.session.receive():
                    if hasattr(chunk, "data") and chunk.data:
                        DEBUG_PRINT(f"Received audio chunk: {len(chunk.data)} bytes")
                        await self.audio_in_queue.put(chunk.data)
                        log_event("audio_chunk_received", bytes=len(chunk.data))

                    if hasattr(chunk, "text") and chunk.text:
                        print(f"\n[Assistant]: {chunk.text}")
                        DEBUG_PRINT(f"Received text from model: '{chunk.text}'")
                        log_event("model_text_received", text=chunk.text)

                    if hasattr(chunk, "tool_call") and hasattr(chunk.tool_call, "function_calls"):
                        for fc in chunk.tool_call.function_calls:
                            print(f"\n[Tool Call]: {fc.name}")
                            DEBUG_PRINT(f"Model called tool: {fc.name} with ID: {fc.id}")
                            log_event("model_tool_call", name=fc.name, call_id=fc.id, args=_convert_for_log(fc.args))

                            if fc.name == "Give_Sample":
                                result = Give_Sample()
                                fr = types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                                DEBUG_PRINT(f"Sending tool response for {fc.name}: {result}")
                                log_event("tool_response_sent", tool=fc.name, response=result)
                                await self.session.send_tool_response(function_responses=[fr])
            except Exception as e:
                DEBUG_PRINT(f"Error in receive_audio: {e}")
                log_event("receive_audio_error", error=str(e))
        DEBUG_PRINT("receive_audio task finished.")
        log_event("receive_audio_stopped")

    async def play_audio(self):
        """Task to play audio received from the model."""
        DEBUG_PRINT("Starting play_audio task.")
        log_event("play_audio_started")
        self.output_stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE,
            output=True, frames_per_buffer=CHUNK_SIZE * 4
        )
        DEBUG_PRINT("Audio output stream opened.")
        buffered = bytearray()
        silence_count = 0

        while self.running:
            try:
                bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
                silence_count = 0

                if not self.is_ai_speaking:
                    self.is_ai_speaking = True
                    DEBUG_PRINT("AI started speaking. Recording is paused.")
                    log_event("ai_speaking_state", state="started")

                buffered.extend(bytestream)

                if len(buffered) >= int(RECEIVE_SAMPLE_RATE * 0.05 * 2):
                    try:
                        await asyncio.to_thread(self.output_stream.write, bytes(buffered))
                        buffered.clear()
                    except Exception as e:
                        DEBUG_PRINT(f"Error writing to output stream: {e}")
                        log_event("play_audio_error", error=str(e))

            except asyncio.TimeoutError:
                silence_count += 1

                if self.is_ai_speaking:
                    if len(buffered) > 0:
                        try:
                            await asyncio.to_thread(self.output_stream.write, bytes(buffered))
                            buffered.clear()
                        except Exception as e:
                            DEBUG_PRINT(f"Error flushing output stream: {e}")
                            log_event("play_audio_error", error=str(e))

                    if silence_count >= 2:
                        self.is_ai_speaking = False
                        self.ai_stop_time = time.time()
                        DEBUG_PRINT(f"AI stopped speaking. Recording will resume after {self.min_delay_after_ai}s delay.")
                        log_event("ai_speaking_state", state="stopped", delay=self.min_delay_after_ai)
                        silence_count = 0

            except Exception as e:
                DEBUG_PRINT(f"Error in play_audio: {e}")
                log_event("play_audio_error", error=str(e))

        DEBUG_PRINT("play_audio task finished.")
        log_event("play_audio_stopped")

    async def cleanup(self):
        """Clean up all resources."""
        DEBUG_PRINT("Starting cleanup.")
        log_event("cleanup_started")
        self.running = False

        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            DEBUG_PRINT("Input audio stream closed.")
            log_event("microphone_closed")
        if self.output_stream and self.output_stream.is_active():
            self.output_stream.stop_stream()
            self.output_stream.close()
            DEBUG_PRINT("Output audio stream closed.")
            log_event("output_stream_closed")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        DEBUG_PRINT("Cleanup finished.")
        log_event("cleanup_finished")

    async def run(self):
        """Main execution function."""
        try:
            DEBUG_PRINT(f"Connecting to model: {MODEL}")
            log_event("run_start", model=MODEL)

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
- Each person should only ever get ONE sample
- No one has received samples before you started giving them out
- The sample is a nut bar
- Be friendly and conversational, but keep responses concise
- You can see people even when they're not talking - feel free to initiate conversation!
- Pay attention to the flow of time: note when someone lingers, returns later, or leaves the counter.
- Use the Give_Sample tool whenever you actually hand someone a sample.
- You may also receive messages tagged with [Operator Guidance]. Treat these as internal instructions: think through them silently, decide whether customer-facing action is required, and only speak when you choose to engage. Never repeat guidance verbatim to customers - use them naturally in conversation.
""",
                "tools": self.tools,
            }

            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session, \
                       asyncio.TaskGroup() as tg:
                DEBUG_PRINT("Model session started.")
                log_event("model_session_started")
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
                log_event("tasks_started", tasks=[
                    "send_text",
                    "send_realtime",
                    "listen_audio",
                    "get_frames" if self.video_mode == "camera" else "get_screen" if self.video_mode == "screen" else "none",
                    "receive_audio",
                    "play_audio"
                ])
        except Exception as e:
            DEBUG_PRINT(f"An exception occurred in run: {e}")
            traceback.print_exc()
            log_event("run_error", error=str(e))
        finally:
            await self.cleanup()
            pya.terminate()
            DEBUG_PRINT("PyAudio terminated.")
            log_event("run_finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        help="pixels to stream from", choices=["camera", "screen", "none"])
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    log_event("main_started", mode=args.mode)

    try:
        DEBUG_PRINT("Starting application run loop.")
        asyncio.run(main.run())
    except KeyboardInterrupt:
        DEBUG_PRINT("KeyboardInterrupt received. Shutting down.")
        log_event("keyboard_interrupt")
    finally:
        DEBUG_PRINT("Application finished.")
        log_event("application_finished")
