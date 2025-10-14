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
from face_reid import FaceReID


# --- DEBUG FLAG ---
DEBUG = True

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR, f"integrated_face_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def setup_logging():
    logger = logging.getLogger("integrated_face")
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
MEMORY_CACHE_FILE = "person_memory_cache.json"
RESET_MEMORY_ON_START = True

client = genai.Client(api_key="", http_options={"api_version": "v1alpha"})


class PersonMemoryCache:
    """Persistent memory cache for storing information about each person."""
    
    def __init__(self, cache_file=MEMORY_CACHE_FILE, reset_on_start=RESET_MEMORY_ON_START):
        self.cache_file = cache_file
        if reset_on_start and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                DEBUG_PRINT(f"Resetting memory cache file at startup: {self.cache_file}")
                log_event("memory_cache_reset", cache_file=self.cache_file)
            except Exception as e:
                DEBUG_PRINT(f"Failed to reset memory cache: {e}")
                log_event("memory_cache_reset_error", cache_file=self.cache_file, error=str(e))
        self.memories = {}  # pid -> memory dict
        self.load_cache()
    
    def load_cache(self):
        """Load existing memory cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.memories = json.load(f)
                DEBUG_PRINT(f"Loaded memory cache with {len(self.memories)} people.")
                log_event("memory_cache_loaded", cache_file=self.cache_file, people=len(self.memories))
            except Exception as e:
                DEBUG_PRINT(f"Error loading cache: {e}")
                log_event("memory_cache_load_error", cache_file=self.cache_file, error=str(e))
                self.memories = {}
        else:
            DEBUG_PRINT("No existing memory cache found. Starting fresh.")
            log_event("memory_cache_not_found", cache_file=self.cache_file)
    
    def save_cache(self):
        """Save memory cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
            DEBUG_PRINT(f"Saved memory cache with {len(self.memories)} people.")
            log_event("memory_cache_saved", cache_file=self.cache_file, people=len(self.memories))
        except Exception as e:
            DEBUG_PRINT(f"Error saving cache: {e}")
            log_event("memory_cache_save_error", cache_file=self.cache_file, error=str(e))
    
    def get_memory(self, pid):
        """Get memory for a person ID."""
        pid_str = str(pid)
        if pid_str not in self.memories:
            self.memories[pid_str] = {
                "pid": pid,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "total_interactions": 0,
                "served_count": 0,
                "last_served": None,
                "conversation_history": [],
                "preferences": {},
                "observations": [],
                "state": "unknown"  # e.g., "interested", "already_served", "just_arrived"
            }
            log_event("memory_created", person_id=pid)
        return self.memories[pid_str]
    
    def update_memory(self, pid, updates):
        """Update memory for a person with new information."""
        memory = self.get_memory(pid)
        memory["last_seen"] = time.time()
        
        # Merge updates
        for key, value in updates.items():
            if key == "conversation_history" or key == "observations":
                # Append to lists
                if value:
                    memory[key].append({
                        "timestamp": time.time(),
                        "content": value
                    })
            elif key == "preferences":
                # Merge preferences dict
                memory["preferences"].update(value)
            else:
                memory[key] = value
        
        self.save_cache()
        log_event("memory_updated", person_id=pid, updates=updates)
        return memory
    
    def format_memory_for_prompt(self, pid):
        """Format memory as text for inclusion in prompts."""
        memory = self.get_memory(pid)
        
        # Calculate time since last seen
        time_since_last = time.time() - memory["last_seen"]
        if time_since_last < 60:
            recency = "just now"
        elif time_since_last < 3600:
            recency = f"{int(time_since_last/60)} minutes ago"
        else:
            recency = f"{int(time_since_last/3600)} hours ago"
        
        # Format conversation history (keep last 5)
        recent_convos = memory["conversation_history"][-5:]
        convo_text = ""
        if recent_convos:
            convo_text = "Recent interactions:\n"
            for c in recent_convos:
                dt = datetime.datetime.fromtimestamp(c["timestamp"]).strftime("%H:%M:%S")
                convo_text += f"  - [{dt}] {c['content']}\n"
        
        # Format observations (keep last 3)
        recent_obs = memory["observations"][-3:]
        obs_text = ""
        if recent_obs:
            obs_text = "Observations:\n"
            for o in recent_obs:
                dt = datetime.datetime.fromtimestamp(o["timestamp"]).strftime("%H:%M:%S")
                obs_text += f"  - [{dt}] {o['content']}\n"
        
        prompt = f"""[Person Memory - ID #{pid}]
Last seen: {recency}
Total interactions: {memory["total_interactions"]}
Served samples: {memory["served_count"]}{f' (last: {datetime.datetime.fromtimestamp(memory["last_served"]).strftime("%H:%M:%S")})' if memory["last_served"] else ''}
Current state: {memory["state"]}
{convo_text}{obs_text}"""
        
        if memory["preferences"]:
            prompt += f"Preferences: {json.dumps(memory['preferences'])}\n"
        
        return prompt


def Give_Sample(memory_cache, focus_pid):
    """Tool function for giving a sample."""
    DEBUG_PRINT(f"Executing tool: Give_Sample for person #{focus_pid}")
    log_event("tool_called", tool="Give_Sample", focus_pid=focus_pid)
    
    if focus_pid:
        memory_cache.update_memory(focus_pid, {
            "served_count": memory_cache.get_memory(focus_pid)["served_count"] + 1,
            "last_served": time.time(),
            "state": "served"
        })
        log_event("sample_served", person_id=focus_pid)
    
    return {
        "status": "Sample delivered successfully",
        "item": "nut bar",
        "name": "done",
        "recipient_id": focus_pid
    }


pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        DEBUG_PRINT(f"Initializing AudioLoop with video_mode: {video_mode}")
        log_event("audio_loop_init", video_mode=video_mode)
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.output_stream = None
        self.running = True
        log_event("audio_loop_state", step="init_complete")
        
        # Memory cache
        self.memory_cache = PersonMemoryCache()
        
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
        self.active_video_send_interval = 2.0  # Increased from 1.0
        self.motion_state_decay = 5.0  # Increased from 3.0
        self.video_send_interval = self.base_video_send_interval
        self.last_audio_send = 0
        self.ambient_audio_interval = 1.0
        self.speaking_audio_interval = 0.3
        self.min_send_bytes = int(SEND_SAMPLE_RATE * 0.25 * 2)
        
        # Voice detection state
        self.min_volume_threshold = 0.0035
        self.noise_floor = self.min_volume_threshold / 2
        self.dynamic_threshold_ratio = 1.6
        self.is_person_speaking = False

        # Visual proactivity state
        self.last_frame_gray = None
        self.motion_pixel_threshold = 25
        self.motion_trigger_ratio = 0.02
        self.motion_cooldown = 2.0  # Increased from 0.25
        self.last_motion_event_time = 0.0
        self.last_proactive_prompt_time = 0.0
        self.proactive_prompt_cooldown = 5.0  # Increased from 0.25

        # Face ReID with memory integration
        self.reid = FaceReID(max_age_s=600, match_threshold=0.42)
        self.current_focus_pid = None
        self.last_memory_update = {}  # pid -> timestamp of last update
        
        if WEBRTC_AVAILABLE:
            self.frame_duration_ms = 30
            self.frame_size = int(SEND_SAMPLE_RATE * self.frame_duration_ms / 1000)
            self.audio_buffer = bytearray()
        
        # Configure tools with memory context
        self.tools = [
            {"function_declarations": [
                {
                    "name": "Give_Sample",
                    "description": "By calling this function, you open a hatch that deposits one sample to the person you're currently focused on."
                },
                {
                    "name": "Update_Person_Memory",
                    "description": "Update your memory about a person. Use this to record observations, preferences, or state changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "person_id": {
                                "type": "integer",
                                "description": "The person ID to update"
                            },
                            "observation": {
                                "type": "string",
                                "description": "New observation about the person"
                            },
                            "state": {
                                "type": "string",
                                "description": "Current state (e.g., 'interested', 'served', 'browsing', 'hesitant')"
                            },
                            "preferences": {
                                "type": "object",
                                "description": "Any preferences mentioned (e.g., dietary restrictions, likes/dislikes)"
                            }
                        },
                        "required": ["person_id"]
                    }
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

        self.last_proactive_prompt_time = now
        prompt_text = f"[Operator Guidance] {text}"
        DEBUG_PRINT(f"Triggering proactive prompt: {prompt_text}")
        log_event("proactive_prompt", prompt=prompt_text)
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
        """Choose which person to focus on (usually closest to center)."""
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
        """Helper to capture and process one camera frame with memory integration."""
        ret, frame = cap.read()
        if not ret:
            return (None, False, None) if detect_motion else (None, None, None)

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
                        "Motion observed near the counter. Check who's approaching."
                    )
                self.last_frame_gray = gray

        # ReID & focus selection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        reid_results = self.reid.process_frame(frame_rgb)
        self._choose_focus_pid(reid_results, W, H)

        # Build memory-aware guidance
        if reid_results:
            memory_context = []
            for r in reid_results:
                pid = r["pid"]
                memory = self.memory_cache.get_memory(pid)
                
                # Check if this is a new sighting (first time in this session)
                if memory["total_interactions"] == 0:
                    memory_context.append(f"New person (ID #{pid}) detected at the counter.")
                    self.memory_cache.update_memory(pid, {
                        "total_interactions": 1,
                        "state": "just_arrived",
                        "observations": "First appearance at the counter"
                    })
                else:
                    # Update last seen
                    time_since = time.time() - memory["last_seen"]
                    if time_since > 300:  # 5 minutes
                        memory_context.append(f"Person #{pid} returned after {int(time_since/60)} minutes.")
                    
                    # Include their full memory
                    memory_text = self.memory_cache.format_memory_for_prompt(pid)
                    memory_context.append(memory_text)
                
                # Update interaction count
                self.memory_cache.update_memory(pid, {
                    "total_interactions": memory["total_interactions"] + 1
                })
            
            if memory_context:
                guidance_msgs.extend(memory_context)

        # Optional debug drawing
        for r in reid_results:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            memory = self.memory_cache.get_memory(r["pid"])
            label = f"ID:{r['pid']} ({memory['state']})"
            if memory["served_count"] > 0:
                label += f" ✓{memory['served_count']}"
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode & return
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([600, 338])
        with io.BytesIO() as image_io:
            img.save(image_io, format="jpeg", quality=80)
            blob = types.Blob(mime_type="image/jpeg", data=image_io.getvalue())

        guidance_text = "\n".join(guidance_msgs) if guidance_msgs else None
        return (blob, motion_detected, guidance_text) if detect_motion else (blob, None, guidance_text)

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
                blob, motion_detected, guidance_text = await asyncio.to_thread(self._get_frame, cap, True)

                if blob:
                    log_event("camera_frame_captured", size=len(blob.data))
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
                                result = Give_Sample(self.memory_cache, self.current_focus_pid)
                                
                                # Mark person as served in ReID too
                                if self.current_focus_pid is not None:
                                    self.reid.mark_served(self.current_focus_pid)
                                    DEBUG_PRINT(f"Marked person #{self.current_focus_pid} as served.")
                                    log_event("person_marked_served", person_id=self.current_focus_pid)
                                
                                fr = types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                                DEBUG_PRINT(f"Sending tool response for {fc.name}: {result}")
                                log_event("tool_response_sent", tool=fc.name, response=result)
                                await self.session.send_tool_response(function_responses=[fr])
                            
                            elif fc.name == "Update_Person_Memory":
                                args = json.loads(fc.args) if isinstance(fc.args, str) else fc.args
                                pid = args.get("person_id")
                                
                                updates = {}
                                if "observation" in args:
                                    updates["observations"] = args["observation"]
                                if "state" in args:
                                    updates["state"] = args["state"]
                                if "preferences" in args:
                                    updates["preferences"] = args["preferences"]
                                
                                if pid and updates:
                                    self.memory_cache.update_memory(pid, updates)
                                    DEBUG_PRINT(f"Updated memory for person #{pid}: {updates}")
                                
                                result = {"status": "Memory updated", "person_id": pid}
                                fr = types.FunctionResponse(id=fc.id, name=fc.name, response=result)
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
                silence_count = 0  # Reset silence counter when we get data
                
                if not self.is_ai_speaking:
                    self.is_ai_speaking = True
                    DEBUG_PRINT("AI started speaking. Recording is paused.")
                    log_event("ai_speaking_state", state="started")
                
                buffered.extend(bytestream)

                # Play immediately when buffer has enough data
                if len(buffered) >= int(RECEIVE_SAMPLE_RATE * 0.05 * 2):  # 50ms buffer
                    try:
                        await asyncio.to_thread(self.output_stream.write, bytes(buffered))
                        buffered.clear()
                    except Exception as e:
                        DEBUG_PRINT(f"Error writing to output stream: {e}")
                        log_event("play_audio_error", error=str(e))
                        
            except asyncio.TimeoutError:
                silence_count += 1
                
                if self.is_ai_speaking:
                    # Flush any remaining audio
                    if len(buffered) > 0:
                        try:
                            await asyncio.to_thread(self.output_stream.write, bytes(buffered))
                            buffered.clear()
                        except Exception as e:
                            DEBUG_PRINT(f"Error flushing output stream: {e}")
                            log_event("play_audio_error", error=str(e))
                    
                    # Only mark as stopped after 2 consecutive timeouts (1 second)
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
        
        # Save memory cache one final time
        self.memory_cache.save_cache()
        
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
- You have persistent memory of every person you interact with, including their ID number
- Always reference your memory when you see someone - acknowledge if they've been here before
- Each person should only get ONE sample per visit
- The sample is a nut bar
- Be friendly and conversational, but keep responses concise
- You can see people even when they're not talking - feel free to initiate conversation!
- Use the Update_Person_Memory tool to record observations, preferences, and state changes about people
- When you observe something noteworthy about a person (interests, dietary needs, mood), update their memory

You may also receive messages tagged with [Operator Guidance] or [Person Memory]. These provide context about people in view. Treat these as internal information: think through them silently, and only speak when you choose to engage customers. Never repeat guidance or memory details verbatim to customers - use them naturally in conversation.""",
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