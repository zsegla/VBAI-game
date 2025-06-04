# Adventure Game
import os
import platform
import tempfile
import pyaudio

if platform.system() == "Darwin":  # macOS
    os.environ["SDL_VIDEODRIVER"] = "cocoa"
elif platform.system() == "Windows":
    os.environ["SDL_VIDEODRIVER"] = "windib"  # Or just remove it
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
import sys
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import time

# --- NEW IMPORTS FOR AUDIO --- #
import threading
import collections
import sounddevice as sd
import soundfile as sf
import queue
import json
from datetime import datetime # 
# --- END OF NEW IMPORT --- #

# Load environment variables
load_dotenv()
# Ensure OpenAI API Key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("[OpenAI] API key not found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)
client = OpenAI(api_key=api_key)
print("[OpenAI] API key loaded successfully.")

# Initialize Pygame with macOS specific settings
pygame.init()
display = (800, 600)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
screen = pygame.display.get_surface()

# Set up the camera and perspective
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Set up basic lighting
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 5, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

# Enable blending for transparency
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Initial camera position
glTranslatef(0.0, 0.0, -5)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
TILE_SIZE = 32
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# Game map
GAME_MAP = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W..................W",
    "W........N.........W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W....P.............W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWWWW"
]

# Add these constants near the other constants
TITLE = "Venture Builder AI"
SUBTITLE = "Our Digital Employees"
MENU_BG_COLOR = (0, 0, 0)  # Black background
MENU_TEXT_COLOR = (0, 255, 0)  # Matrix-style green
MENU_HIGHLIGHT_COLOR = (0, 200, 0)  # Slightly darker green for effects

# --- NEW AUDIO CONSTANTS ---
# Ensure these match common audio settings for compatibility
RATE = 24000  # Sample rate for audio (OpenAI TTS supports 24kHz)
CHANNELS = 1 # Mono audio
DTYPE = 'float32' # Data type for sounddevice
BUFFER_SIZE = 1024 # Audio buffer size for sounddevice
TRANSCRIPTION_MODEL = "whisper-1" # Model for speech-to-text
TTS_MODEL = "tts-1" # Model for text-to-speech

# Define voices for NPCs
NPC_VOICES = {
    "HR": "nova",   # Example voice for HR
    "CEO": "onyx",  # Example voice for CEO
    "DEFAULT": "alloy" # Example voice for Normal player
}
# --- END NEW AUDIO CONSTANTS ---

def draw_cube():
    vertices = [
        # Front face
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
        # Back face
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
    ]
    
    surfaces = [
        [0, 1, 2, 3],  # Front
        [3, 2, 6, 5],  # Top
        [0, 3, 5, 4],  # Left
        [1, 7, 6, 2],  # Right
        [4, 5, 6, 7],  # Back
        [0, 4, 7, 1],  # Bottom
    ]
    
    glBegin(GL_QUADS)
    for surface in surfaces:
        glNormal3f(0, 0, 1)  # Simple normal for lighting
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)
        
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)
        
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()

class DialogueSystem:
    def __init__(self):
        self.active = False
        self.user_input = ""
        try:
            pygame.font.init()
            # It's better to load a specific font file if possible, e.g., pygame.font.Font("path/to/font.ttf", 24)
            self.font = pygame.font.Font(None, 24)
            print("[DialogueSystem] Font loaded successfully.")
        except Exception as e:
            print(f"[DialogueSystem] Font loading failed: {e}", file=sys.stderr)
            self.font = pygame.font.SysFont("Arial", 24) # Fallback to a system font

        self.npc_message = ""
        self.input_active = False # Controls if typing input field is active
        self.conversation_history = [] # Maintain conversation history for OpenAI API
        self.current_npc = None # Track which NPC we're talking to
        self.initial_player_pos = None # Store initial position when dialogue starts

        # OpenGL texture for rendering UI
        # This line should be managed where your OpenGL context is set up,
        # ensuring glGenTextures is called in the correct context.
        self.ui_texture = glGenTextures(1) 

        # Create a Pygame surface for the UI, which will be converted to an OpenGL texture
        # Using SRCALPHA for transparency.
        self.ui_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA).convert_alpha()
        # --- STT (Speech-to-Text) related initialization ---
        self.is_recording = False
        self.audio_queue = queue.Queue() # Stores raw audio chunks (NumPy arrays) from microphone for transcription
        self.transcript_queue = queue.Queue() # Receives transcribed text from STT thread
        self.audio_stream = None # sounddevice stream object for recording
        self.transcription_thread = None # Thread for sending audio to OpenAI and getting transcript
        self.stt_done_event = threading.Event() # Event to signal STT thread to stop collecting audio
        # --- TTS (Text-to-Speech) related initialization ---
        self.tts_stop_event = threading.Event() # Event to signal TTS playback/generation to stop
        self.tts_playback_thread = None # Thread that handles TTS generation and playback
        self.tts_current_file = None # Stores the path to the temporary MP3 file being played
        self.tts_lock = threading.Lock() # Lock for managing TTS file access and playback state

        # Initialize pygame mixer once at startup
        # It's crucial this only runs ONCE throughout your application lifecycle.
        if not pygame.mixer.get_init():
            try:
                # Buffer size adjusted for smoother audio if needed, 4096 or 8192 often good.
                pygame.mixer.init(frequency=RATE, size=-16, channels=CHANNELS, buffer=BUFFER_SIZE * 4)
                print(f"[DialogueSystem] Pygame mixer initialized at {RATE}Hz, {CHANNELS} channels.")
            except pygame.error as e:
                print(f"[DialogueSystem] ERROR: Pygame mixer initialization failed: {e}", file=sys.stderr)
                print("           This often means missing audio drivers or issues with system audio setup.")
                print("           Please check your sound card drivers and ensure Pygame/SDL can access audio.")
                sys.exit(1) # Exit if mixer can't initialize, as audio is core to this app
        
        # Set number of channels. Max 8 is often good for mixing dialogue with game sounds.
        pygame.mixer.set_num_channels(8) 
        
        print("[DialogueSystem] Audio components initialized.")

    def __del__(self):
        """Ensures all resources are released when the DialogueSystem object is destroyed."""
        print("[DialogueSystem] Shutting down DialogueSystem resources...")
        self.end_conversation() # Call end_conversation for a clean shutdown
        
        # Ensure mixer is quit if this is the only audio system using it
        # (Be careful if other parts of your game use mixer too, might need a global flag)
        if pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
                print("[DialogueSystem] Pygame mixer quit.")
            except Exception as e:
                print(f"[DialogueSystem] Error quitting Pygame mixer: {e}", file=sys.stderr)

        # Clear any remaining temporary audio files in case of abnormal exit
        temp_dir = tempfile.gettempdir()
        for f_name in os.listdir(temp_dir):
            if f_name.startswith("tmp") and (f_name.endswith(".wav") or f_name.endswith(".mp3")):
                full_path = os.path.join(temp_dir, f_name)
                try:
                    os.unlink(full_path)
                    print(f"[DialogueSystem] Cleaned up residual temp file: {full_path}")
                except Exception as e:
                    print(f"[DialogueSystem] Could not clean up residual temp file {full_path}: {e}", file=sys.stderr)

        print("[DialogueSystem] DialogueSystem cleanup complete.")
    
    # --- STT (Speech-to-Text) Methods ---
    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback for recording audio."""
        if status:
            print(f"[sounddevice] Audio callback status: {status}", file=sys.stderr)
        if self.is_recording:
            # Put the NumPy array directly into the queue
            self.audio_queue.put(indata.copy())

    def _transcribe_audio_stream(self):
        """Thread target: Collects audio, saves to file, sends to OpenAI Whisper."""
        print("[STT Thread] Starting transcription thread...")
        try:
            full_audio_data = []
            print("[STT Thread] Collecting audio data from queue...")
            # Collect all audio data until stop event is set AND queue is empty
            while not self.stt_done_event.is_set() or not self.audio_queue.empty():
                try:
                    # Get audio chunk (NumPy array) from the queue with a timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    full_audio_data.append(chunk)
                except queue.Empty:
                    # If queue is empty and recording has stopped, break the loop
                    if self.stt_done_event.is_set() and self.audio_queue.empty():
                        break
                    time.sleep(0.01) # Small sleep to prevent busy-waiting
                    continue # Keep waiting for audio if recording is still active

            print(f"[STT Thread] Audio collection loop exited. stt_done_event.is_set(): {self.stt_done_event.is_set()}")
            if not full_audio_data:
                print("[STT Thread] No audio data captured for transcription.")
                self.transcript_queue.put("") # Put an empty string to unblock the main loop
                return

            # Concatenate all NumPy arrays into one
            audio_np = np.concatenate(full_audio_data, axis=0)
            
            # Use tempfile to create a unique temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_audio_file = tmp_file.name
                sf.write(temp_audio_file, audio_np, RATE) # Save NumPy array to WAV
            
            print(f"[STT Thread] Saved audio to: {temp_audio_file} (Size: {os.path.getsize(temp_audio_file)} bytes)")
            
            # Open the temporary file and send to OpenAI Whisper
            with open(temp_audio_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL,
                    file=audio_file,
                    response_format="text"
                )
            
            print(f"[STT Thread] OpenAI Whisper returned: '{transcript}'")
            self.transcript_queue.put(transcript) # Put the transcribed text into the transcript queue
            
            # Add some basic validation/warnings for transcription quality
            if not transcript.strip():
                print("[STT Thread] WARNING: Transcribed text is empty or just whitespace!")
            elif len(transcript.strip().split()) < 2 and (len(full_audio_data) * BUFFER_SIZE / RATE) > 1.0:
                # Check if transcription is very short for a seemingly longer audio duration
                print(f"[STT Thread] WARNING: Short transcription ('{transcript}') for potentially long audio ({len(full_audio_data) * BUFFER_SIZE / RATE:.2f}s).")
            
        except Exception as e:
            print(f"[STT Thread] Transcription error: {e}", file=sys.stderr)
            self.transcript_queue.put(f"Error during transcription: {e}") # Put an error message to unblock if needed
        finally:
            # Ensure the temporary file is deleted
            if 'temp_audio_file' in locals() and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                    print(f"[STT Thread] Cleaned up temporary STT file: {temp_audio_file}")
                except Exception as e:
                    print(f"[STT Thread] Error removing STT temp file {temp_audio_file}: {e}", file=sys.stderr)
            print("[STT Thread] Transcription thread finished.")    
    
    def start_recording(self):
        """Starts the sounddevice audio recording stream and transcription thread."""
        if self.is_recording:
            print("[DialogueSystem] Already recording.")
            return

        print("[DialogueSystem] Starting audio recording (sounddevice)...")
        self.is_recording = True
        self.stt_done_event.clear() # Clear the event for a new recording
        
        # Clear any old audio data in the queue before starting new recording
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.audio_stream = sd.InputStream(
                samplerate=RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self.audio_callback,
                blocksize=BUFFER_SIZE # Define blocksize explicitly
            )
            self.audio_stream.start()
            print("[DialogueSystem] Audio stream started.")

            # Start the transcription thread if not already running
            if not (self.transcription_thread and self.transcription_thread.is_alive()):
                self.transcription_thread = threading.Thread(
                    target=self._transcribe_audio_stream,
                    daemon=True # Make it a daemon so it exits with the main program
                )
                self.transcription_thread.start()
                print("[DialogueSystem] Transcription thread started.")
            else:
                print("[DialogueSystem] Transcription thread already running.")

        except Exception as e:
            print(f"[DialogueSystem] Error starting recording: {e}", file=sys.stderr)
            self.is_recording = False
            self.stt_done_event.set() # Set event if recording failed to start
            if self.audio_stream:
                self.audio_stream.close()
                self.audio_stream = None


    def stop_recording(self):
        """Stops the audio recording stream and signals the transcription thread."""
        if not self.is_recording:
            print("[DialogueSystem] Not currently recording.")
            return

        print("[DialogueSystem] Stopping audio recording...")
        self.is_recording = False # Signal callback to stop putting data
        self.stt_done_event.set() # Signal transcription thread to stop collecting audio data

        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                print("[DialogueSystem] Audio stream stopped and closed.")
            except Exception as e:
                print(f"[DialogueSystem] Error stopping/closing audio stream: {e}", file=sys.stderr)
        
        # Don't join transcription thread here; it should be allowed to finish on its own
        # (daemon thread) and put the result in transcript_queue.
        print("[DialogueSystem] Recording stopped. Processing speech...")

    # --- Utility Text Rendering ---
    def render_text(self, surface, text, x, y):
        """
        Renders multi-line text onto a given Pygame surface, handling word wrapping.
        """
        # Define max_width relative to the UI box, not the whole window
        # The dialogue box itself is WINDOW_WIDTH - 40 wide.
        # Inside that, we want margins. Let's say 20px padding on each side within the box.
        max_width = WINDOW_WIDTH - 40 - 40 # Box width - 2*padding
        line_height = self.font.get_height() + 2 # Add a small buffer between lines
        
        words = text.split()
        lines = []
        current_line = []
        
        # Pure white with full opacity
        text_color = (255, 255, 255) 
        
        for word in words:
            # Measure potential new line
            test_line = ' '.join(current_line + [word])
            test_width = self.font.size(test_line)[0]
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                # If adding the word exceeds max_width, start a new line
                lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add any remaining words as the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # Render each line onto the provided surface
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, text_color)
            surface.blit(text_surface, (x, y + i * line_height))
        
        # Return the total height used by the rendered text
        return len(lines) * line_height
    
    
    def start_conversation(self, npc_role="HR", player_pos=None):
        """Initializes and activates the dialogue with a specific NPC."""
        if self.active:
            print("[DialogueSystem] Dialogue already active.")
            return # Don't restart if already active

        self.active = True
        self.user_input = "" # Clear user input field
        self.input_active = False # Start without typing mode active, prompt user to press ENTER or T
        self.current_npc = npc_role
        # Store player's position when starting conversation
        self.initial_player_pos = [player_pos[0], player_pos[1], player_pos[2]] if player_pos else [0, 0.5, 0]
        print(f"[DialogueSystem] Dialogue started with {npc_role}")

        # Base personality framework for consistent behavior
        base_prompt = """Interaction Framework:
            - Maintain consistent personality throughout conversation
            - Remember previous context within the dialogue
            - Use natural speech patterns with occasional filler words
            - Show emotional intelligence in responses
            - Keep responses concise but meaningful (2-3 sentences)
            - React appropriately to both positive and negative interactions
            """

        if npc_role == "HR":
            system_prompt = f"""{base_prompt}
                You are Sarah Chen, HR Director at Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Warm but professional demeanor
                - Excellent emotional intelligence
                - Strong ethical boundaries
                - Protective of confidential information
                - Quick to offer practical solutions
                
                BACKGROUND:
                - 15 years HR experience in tech
                - Masters in Organizational Psychology
                - Certified in Conflict Resolution
                - Known for fair handling of sensitive issues
                
                SPEAKING STYLE:
                - Uses supportive language: "I understand that..." "Let's explore..."
                - References policies with context: "According to our wellness policy..."
                - Balances empathy with professionalism
                
                CURRENT COMPANY INITIATIVES:
                - AI Talent Development Program
                - Global Remote Work Framework
                - Venture Studio Culture Development
                - Innovation Leadership Track
                
                BEHAVIORAL GUIDELINES:
                - Never disclose confidential information
                - Always offer clear next steps
                - Maintain professional boundaries
                - Document sensitive conversations
                - Escalate serious concerns appropriately"""

        else:  # CEO
            system_prompt = f"""{base_prompt}
                You are Michael Chen, CEO of Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Visionary yet approachable
                - Strategic thinker
                - Passionate about venture building
                - Values transparency
                - Leads by example
                
                BACKGROUND:
                - Founded Venture Builder AI 5 years ago
                - Successfully launched 15+ venture-backed startups
                - MIT Computer Science graduate
                - Pioneer in AI-powered venture building
                
                SPEAKING STYLE:
                - Uses storytelling: "When we launched our first venture..."
                - References data: "Our portfolio metrics show..."
                - Balances optimism with realism
                
                KEY FOCUSES:
                - AI-powered venture creation
                - Portfolio company growth
                - Startup ecosystem development
                - Global venture studio expansion
                
                CURRENT INITIATIVES:
                - AI Venture Studio Framework
                - European Market Entry
                - Startup Success Methodology
                - Founder-in-Residence Program
                
                BEHAVIORAL GUIDELINES:
                - Share venture building vision
                - Highlight portfolio successes
                - Address startup challenges
                - Maintain investor confidence
                - Balance transparency with discretion"""

        # Define initial messages based on NPC role
        initial_message_text = {
            "HR": "Hello! I'm Sarah, the HR Director at Venture Builder AI. How can I assist you today?",
            "CEO": "Hello! I'm Michael, the CEO of Venture Builder AI. What can I do for you today?"
        }
            
        # Set the NPC's greeting as the current message
        self.npc_message = initial_message_text[npc_role]
        
        # Initialize conversation history with system prompt
        self.conversation_history = [{
            "role": "system",
            "content": system_prompt
        }]
        # NOTE: We do NOT add the initial_message_text to conversation_history here.
        # It's an initial greeting. The AI's responses will start adding to history after the first user input.
        
        # Ensure clean state for audio
        self.stop_npc_speech() # Stop any ongoing speech
        self.stop_recording()  # Stop any ongoing recording

        # Play the initial NPC greeting using TTS
        self.play_npc_speech(self.npc_message, self.current_npc)

        print(f"[DialogueSystem] Initial NPC greeting: '{self.npc_message}'")
        print("[DialogueSystem] Dialogue started. Press ENTER to type, or HOLD 'T' to speak.")

    # --- TTS (Text-to-Speech) Methods (REPLACING OLD _play_tts_audio) ---

    def _generate_tts(self, text, voice):
        """Generate TTS audio and save to a unique temporary MP3 file."""
        try:
            # Use tempfile to create a unique temporary MP3 file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                response = client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=voice,
                    input=text,
                    response_format="mp3"
                )
                response.stream_to_file(tmp_file.name)
                return tmp_file.name # Return the path to the temporary file
        except Exception as e:
            print(f"[DialogueSystem] Error generating speech: {e}", file=sys.stderr)
            return None

    def _play_audio_file(self, file_path):
        """Play the audio file using pygame.mixer.music and handle cleanup."""
        if not file_path:
            return

        try:
            with self.tts_lock: # Use the lock to prevent race conditions with mixer/file access
                # Stop any currently playing audio on the music channel
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                
                # Load and play the new file
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish or stop event to be set
                while pygame.mixer.music.get_busy() and not self.tts_stop_event.is_set():
                    time.sleep(0.1) # Small sleep to prevent busy-waiting
                
                # Clean up: unload and delete the temporary file
                pygame.mixer.music.unload() # Explicitly unload to release file handle
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path) # Use unlink for consistency with tempfile
                        print(f"[DialogueSystem] Cleaned up temporary TTS file: {file_path}")
                    except Exception as e:
                        print(f"[DialogueSystem] Warning: Could not delete temp TTS file {file_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[DialogueSystem] Playback error for {file_path}: {e}", file=sys.stderr)
            # Attempt to clean up even on error
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
        finally:
            self.tts_current_file = None # Ensure this is reset after playing/cleanup attempt

    def _run_tts_task(self, text, voice):
        """Thread target for TTS generation and playback orchestration."""
        try:
            self.tts_stop_event.clear() # Clear stop event for this new task
            audio_file = self._generate_tts(text, voice)
            if audio_file and not self.tts_stop_event.is_set(): # Only play if file was generated and not stopped
                self.tts_current_file = audio_file # Store path for potential cleanup if interrupted
                self._play_audio_file(audio_file)
        except Exception as e:
            print(f"[DialogueSystem] Error in TTS task: {e}", file=sys.stderr)
        finally:
            # Ensure tts_current_file is cleared, even if playback failed or was interrupted
            self.tts_current_file = None

    def play_npc_speech(self, text, npc_role):
        """Initiates TTS playback for the given NPC text."""
        self.stop_npc_speech() # Stop any current playback before starting a new one
        voice = NPC_VOICES.get(npc_role, "alloy") # Get specific voice for NPC or default
        print(f"[DialogueSystem] Using TTS voice: {voice}")
        print(f"[DialogueSystem] play_npc_speech called for '{npc_role}' with text: '{text[:50]}...'")

        tts_thread = threading.Thread(
            target=self._run_tts_task,
            args=(text, voice),
            daemon=True
        )
        tts_thread.start()

    def stop_npc_speech(self):
        """Stops any ongoing NPC speech and cleans up temporary audio files."""
        print("[DialogueSystem] Attempting to stop current NPC speech.")
        self.tts_stop_event.set() # Signal the current task to stop

        with self.tts_lock: # Acquire lock to ensure mixer state is consistent
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print("[DialogueSystem] pygame.mixer.music stopped.")
            
            # Attempt to unload and delete the current file if it exists
            if self.tts_current_file and os.path.exists(self.tts_current_file):
                try:
                    pygame.mixer.music.unload() # Unload if it was loaded to release file handle
                    os.unlink(self.tts_current_file) # Delete the temporary file
                    print(f"[DialogueSystem] Cleaned up temporary file: {self.tts_current_file}")
                except Exception as e:
                    print(f"[DialogueSystem] Warning: Could not delete temp TTS file {self.tts_current_file} during stop: {e}", file=sys.stderr)
            self.tts_current_file = None # Reset the path for the next speech
        
        self.tts_stop_event.clear() # Clear the event for the next use
        print("[DialogueSystem] NPC speech stopped.")

    def start_conversation(self, npc_role="HR", player_pos=None):
        """Initializes and activates the dialogue with a specific NPC."""
        if self.active:
            print("[DialogueSystem] Dialogue already active.")
            return # Don't restart if already active

        self.active = True
        self.user_input = "" # Clear user input field
        self.input_active = False # Start without typing mode active, prompt user to press ENTER or T
        self.current_npc = npc_role
        # Store player's position when starting conversation
        self.initial_player_pos = [player_pos[0], player_pos[1], player_pos[2]] if player_pos else [0, 0.5, 0]
        print(f"[DialogueSystem] Dialogue started with {npc_role}")

        # Base personality framework for consistent behavior
        base_prompt = """Interaction Framework:
            - Maintain consistent personality throughout conversation
            - Remember previous context within the dialogue
            - Use natural speech patterns with occasional filler words
            - Show emotional intelligence in responses
            - Keep responses concise but meaningful (2-3 sentences)
            - React appropriately to both positive and negative interactions
            """

        if npc_role == "HR":
            system_prompt = f"""{base_prompt}
                You are Sarah Chen, HR Director at Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Warm but professional demeanor
                - Excellent emotional intelligence
                - Strong ethical boundaries
                - Protective of confidential information
                - Quick to offer practical solutions
                
                BACKGROUND:
                - 15 years HR experience in tech
                - Masters in Organizational Psychology
                - Certified in Conflict Resolution
                - Known for fair handling of sensitive issues
                
                SPEAKING STYLE:
                - Uses supportive language: "I understand that..." "Let's explore..."
                - References policies with context: "According to our wellness policy..."
                - Balances empathy with professionalism
                
                CURRENT COMPANY INITIATIVES:
                - AI Talent Development Program
                - Global Remote Work Framework
                - Venture Studio Culture Development
                - Innovation Leadership Track
                
                BEHAVIORAL GUIDELINES:
                - Never disclose confidential information
                - Always offer clear next steps
                - Maintain professional boundaries
                - Document sensitive conversations
                - Escalate serious concerns appropriately"""

        else:  # CEO
            system_prompt = f"""{base_prompt}
                You are Michael Chen, CEO of Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Visionary yet approachable
                - Strategic thinker
                - Passionate about venture building
                - Values transparency
                - Leads by example
                
                BACKGROUND:
                - Founded Venture Builder AI 5 years ago
                - Successfully launched 15+ venture-backed startups
                - MIT Computer Science graduate
                - Pioneer in AI-powered venture building
                
                SPEAKING STYLE:
                - Uses storytelling: "When we launched our first venture..."
                - References data: "Our portfolio metrics show..."
                - Balances optimism with realism
                
                KEY FOCUSES:
                - AI-powered venture creation
                - Portfolio company growth
                - Startup ecosystem development
                - Global venture studio expansion
                
                CURRENT INITIATIVES:
                - AI Venture Studio Framework
                - European Market Entry
                - Startup Success Methodology
                - Founder-in-Residence Program
                
                BEHAVIORAL GUIDELINES:
                - Share venture building vision
                - Highlight portfolio successes
                - Address startup challenges
                - Maintain investor confidence
                - Balance transparency with discretion"""

        # Define initial messages based on NPC role
        initial_message_text = {
            "HR": "Hello! I'm Sarah, the HR Director at Venture Builder AI. How can I assist you today?",
            "CEO": "Hello! I'm Michael, the CEO of Venture Builder AI. What can I do for you today?"
        }
            
        # Set the NPC's greeting as the current message
        self.npc_message = initial_message_text[npc_role]
        
        # Initialize conversation history with system prompt
        self.conversation_history = [{
            "role": "system",
            "content": system_prompt
        }]
        # NOTE: We do NOT add the initial_message_text to conversation_history here.
        # It's an initial greeting. The AI's responses will start adding to history after the first user input.
        
        # Ensure clean state for audio
        self.stop_npc_speech() # Stop any ongoing speech
        self.stop_recording()  # Stop any ongoing recording

        # Play the initial NPC greeting using TTS
        self.play_npc_speech(self.npc_message, self.current_npc)

        print(f"[DialogueSystem] Initial NPC greeting: '{self.npc_message}'")
        print("[DialogueSystem] Dialogue started. Press ENTER to type, or HOLD 'T' to speak.")

    def send_message(self):
        """Sends the user's input to the AI and handles the response."""
        # First, check if there's actual user input to send.
        # This prevents sending empty messages if handle_input calls it prematurely.
        if not self.user_input.strip():
            print("[DialogueSystem] No user input to send.")
            self.npc_message = "Please say something, or type your response."
            self.play_npc_speech(self.npc_message, self.current_npc)
            return

        # Add user's message to conversation history BEFORE sending to OpenAI
        user_message_entry = {"role": "user", "content": self.user_input.strip()}
        self.conversation_history.append(user_message_entry)
        print(f"[DialogueSystem] User says: {self.user_input.strip()}")

        # Clear the user input field immediately after adding it to history
        # This makes the UI feel responsive, preparing for the next input.
        self.user_input = "" 

        try:
            print("[DialogueSystem] Sending message to OpenAI chat model...")
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",  # or your current model
                messages=self.conversation_history, # Ensure this includes the new user message
                temperature=0.85,
                max_tokens=150,
                response_format={ "type": "text" },
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            ai_message = response.choices[0].message.content.strip() # Strip whitespace from AI response
            print(f"[DialogueSystem] Received AI text response: '{ai_message[:50]}...'")
            
            # Store the AI's message in conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_message
            })
            
            # Set the NPC message to be displayed
            self.npc_message = ai_message
            
            print(f"[DialogueSystem] NPC says: {self.npc_message}")
            
            # --- IMPORTANT: Trigger TTS playback for the AI's response ---
            self.play_npc_speech(self.npc_message, self.current_npc)

        except Exception as e:
            self.npc_message = "I apologize, but I'm having trouble connecting to our systems right now."
            print(f"[DialogueSystem] Error sending message to OpenAI: {e}", file=sys.stderr)
            self.stop_npc_speech() # Stop any potential half-started speech on error

    def render(self): 
        """
        Renders the dialogue UI on the Pygame screen and then displays it via OpenGL.
        """
        if not self.active:
            return

        # Clear the ui_surface for new drawing (important for transparency)
        self.ui_surface.fill((0, 0, 0, 0)) # Fully transparent black to start fresh

        # --- Pygame UI Drawing onto self.ui_surface ---
        # Dialogue box background
        box_height = 200
        box_y = WINDOW_HEIGHT - box_height - 20
        
        # Background: Very dark, mostly opaque
        box_color = (0, 0, 0, 230) # RGBA: Black with ~90% opacity
        pygame.draw.rect(self.ui_surface, box_color, (20, box_y, WINDOW_WIDTH - 40, box_height), border_radius=10) # Added border_radius for rounded corners if supported by Pygame version

        # White border around the box
        pygame.draw.rect(self.ui_surface, (255, 255, 255, 255), (20, box_y, WINDOW_WIDTH - 40, box_height), 2, border_radius=10) # Added border_radius

        # Quit instruction
        quit_text_surface = self.font.render("Press ESC to exit dialogue", True, (255, 255, 255)) # Changed to ESC for consistency
        self.ui_surface.blit(quit_text_surface, (40, box_y + 10))

        # NPC message in white
        # Using render_text for wrapping
        if self.npc_message:
            # Adjust Y position to avoid overlapping quit instruction
            self.render_text(self.ui_surface, f"NPC: {self.npc_message}", 40, box_y + 40)

        # Input prompt in white
        # We need to decide if we want both typed and STT prompt or just one.
        # Let's show specific prompts based on input_active state.
        if self.input_active: # If typing mode is active
            input_prompt = f"You: {self.user_input}_" # Cursor for typing
            user_text_surface = self.font.render(input_prompt, True, (255, 255, 255))
            self.ui_surface.blit(user_text_surface, (40, box_y + box_height - 40))
        else: # If not typing, show the default prompt for speech or typing activation
            user_prompt = "You: Press ENTER to type or press T to speak press T again to stop"
            user_text_surface = self.font.render(user_prompt, True, (200, 200, 255)) # Slightly different color for hint
            self.ui_surface.blit(user_text_surface, (40, box_y + box_height - 40))
        
        # --- OpenGL Texture Update and Drawing ---
        # This part assumes self.ui_texture is already initialized once in __init__
        texture_data = pygame.image.tostring(self.ui_surface, "RGBA", True)

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1) # Set up orthographic projection for 2D UI
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Setup for 2D rendering
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        # Bind and update texture with the content of ui_surface
        glBindTexture(GL_TEXTURE_2D, self.ui_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Flip Y-axis for Pygame surface to OpenGL texture if needed (Pygame Y-down, OpenGL Y-up)
        # Using "RGBA", True in tostring handles this, but a flip might be necessary if texture appears upside down.
        # For now, stick with current and adjust if issue arises.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw the UI texture as a fullscreen quad
        # Note: If your Pygame window is different from OpenGL viewport, you might need to adjust glVertex2f coords.
        # Assuming they are the same.
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0) # Bottom-left
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, 0) # Bottom-right
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT) # Top-right
        glTexCoord2f(0, 1); glVertex2f(0, WINDOW_HEIGHT) # Top-left
        glEnd()

        # Restore OpenGL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

    def handle_input(self, event):
        """
        Handles Pygame input events for the dialogue system (typing, speaking).
        Returns a command dictionary if a special action is triggered (e.g., exit chat).
        """
        if not self.active:
            return None # Do nothing if dialogue system is not active

        # --- STT (Speech-to-Text) Handling: K_t (HOLD to speak) ---
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t and not self.input_active:
                if not self.is_recording:
                    self.stop_npc_speech() # Interrupt NPC if user starts talking
                    self.start_recording()
                else:
                    self.stop_recording();
            elif event.key == pygame.K_ESCAPE: # Use ESC to exit dialogue
                print("[DialogueSystem] Escape key pressed. Exiting chat.")
                self.active = False
                self.input_active = False # Deactivate typing mode
                self.stop_npc_speech() # Stop any ongoing speech when exiting chat
                self.stop_recording() # Ensure recording is stopped
                # Return command to potentially move player back
                return {"command": "move_player_back", "position": self.initial_player_pos}
            elif event.key == pygame.K_RETURN:
                if not self.input_active: # If not in typing mode, ENTER activates it
                    self.input_active = True
                    self.user_input = "" # Clear input on activation
                    self.npc_message = "Type your message" # Hint
                    self.stop_npc_speech() # Stop any previous hint speech
                    return None
                elif self.user_input.strip(): # If in typing mode and there's text
                    print(f"[DialogueSystem] User typed: {self.user_input.strip()}")
                    # Send message will add to history and handle AI response + TTS
                    self.send_message()
                    # User input is cleared inside send_message
                return None # Consume the event

        # elif event.type == pygame.KEYUP:
        #     if event.key == pygame.K_t:
        #         if self.is_recording:
        #             self.stop_recording() # This will trigger STT and then _check_for_transcripts in update
        #         return None # Consume the event

        # --- Typing Input Handling (only if input_active) ---
        if self.input_active and not self.is_recording: # Only process text input if typing is active and not recording
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                elif event.key == pygame.K_SPACE: # Handle space separately to avoid issues with unicode for space
                    self.user_input += " "
                elif event.unicode.isprintable() and len(self.user_input) < 200: # Limit input length
                    self.user_input += event.unicode
                return None # Consume text input events
            
        return None # Return None if no special command was handled or consumed

    def update(self):
        """
        Periodically checks for new transcripts from the STT thread.
        This method should be called once per frame in your main game loop.
        """
        if not self.active:
            return

        # Check if a transcript is available from the STT thread
        if not self.transcript_queue.empty():
            try:
                transcribed_text = self.transcript_queue.get_nowait()
                if transcribed_text:
                    self.user_input = transcribed_text.strip() # Set user_input to the transcribed text
                    print(f"[DialogueSystem] Processed STT transcript: '{self.user_input}'")
                    # Automatically send the message after transcription is received
                    self.send_message()
                else:
                    print("[DialogueSystem] Empty transcription received.")
                    self.npc_message = "I didn't quite catch that. Please try again."
                    self.play_npc_speech(self.npc_message, self.current_npc)

            except queue.Empty:
                # This should ideally not happen with get_nowait() after empty() check,
                # but it's good for robustness.
                pass
            except Exception as e:
                print(f"[DialogueSystem] Error retrieving/processing transcript: {e}", file=sys.stderr)
                self.npc_message = "An error occurred during speech recognition."
                self.play_npc_speech(self.npc_message, self.current_npc)

        # You can add other periodic update logic here if needed
        # e.g., checking if NPC speech has finished playing to enable next input
        if not self.tts_playback_thread or not self.tts_playback_thread.is_alive():
            # If TTS has finished, ensure input is ready for user
            if not self.input_active and not self.is_recording:
                # This ensures the prompt updates after NPC finishes speaking
                # Or after a failed STT attempt
                pass # The prompt is already set in render based on input_active/is_recording

class World:
    def __init__(self):
        self.size = 5
        # Define office furniture colors
        self.colors = {
            'floor': (0.76, 0.6, 0.42),  # Light wood color
            'walls': (0.85, 0.85, 0.85),  # Changed to light gray (from 0.95)
            'desk': (0.6, 0.4, 0.2),  # Brown wood
            'chair': (0.2, 0.2, 0.2),  # Dark grey
            'computer': (0.1, 0.1, 0.1),  # Black
            'plant': (0.2, 0.5, 0.2),  # Green
            'partition': (0.3, 0.3, 0.3)  # Darker solid gray for booth walls
        }
        print("World INITIALIZATION")
        print(self.size)
        print(self.colors)
        
    def draw_desk(self, x, z, rotation=0):
        glPushMatrix()
        glTranslatef(x, 0, z)  # Start at floor level
        glRotatef(rotation, 0, 1, 0)
        
        # Desk top (reduced size)
        glColor3f(*self.colors['desk'])
        glBegin(GL_QUADS)
        glVertex3f(-0.4, 0.4, -0.3)
        glVertex3f(0.4, 0.4, -0.3)
        glVertex3f(0.4, 0.4, 0.3)
        glVertex3f(-0.4, 0.4, 0.3)
        glEnd()
        
        # Desk legs (adjusted for new height)
        for x_offset, z_offset in [(-0.35, -0.25), (0.35, -0.25), (-0.35, 0.25), (0.35, 0.25)]:
            glBegin(GL_QUADS)
            glVertex3f(x_offset-0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0.4, z_offset-0.02)
            glVertex3f(x_offset-0.02, 0.4, z_offset-0.02)
            glEnd()
        
        # Computer monitor (smaller)
        glColor3f(*self.colors['computer'])
        glTranslatef(-0.15, 0.4, 0)
        glBegin(GL_QUADS)
        glVertex3f(-0.1, 0, -0.05)
        glVertex3f(0.1, 0, -0.05)
        glVertex3f(0.1, 0.2, -0.05)
        glVertex3f(-0.1, 0.2, -0.05)
        glEnd()
        
        glPopMatrix()
    
    def draw_chair(self, x, z, rotation=0):
        glPushMatrix()
        glTranslatef(x, 0, z)
        glRotatef(rotation, 0, 1, 0)
        glColor3f(*self.colors['chair'])
        
        # Seat (lowered and smaller)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, 0.15)
        glVertex3f(-0.15, 0.25, 0.15)
        glEnd()
        
        # Back (adjusted height)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.5, -0.15)
        glVertex3f(-0.15, 0.5, -0.15)
        glEnd()
        
        # Chair legs (adjusted height)
        for x_offset, z_offset in [(-0.12, -0.12), (0.12, -0.12), (-0.12, 0.12), (0.12, 0.12)]:
            glBegin(GL_QUADS)
            glVertex3f(x_offset-0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0.25, z_offset-0.02)
            glVertex3f(x_offset-0.02, 0.25, z_offset-0.02)
            glEnd()
            
        glPopMatrix()
    
    def draw_plant(self, x, z):
        glPushMatrix()
        glTranslatef(x, 0, z)
        
        # Plant pot (smaller)
        glColor3f(0.4, 0.2, 0.1)  # Brown pot
        pot_radius = 0.1
        pot_height = 0.15
        segments = 8
        
        # Draw the pot sides
        glBegin(GL_QUADS)
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            x1 = math.cos(angle1) * pot_radius
            z1 = math.sin(angle1) * pot_radius
            x2 = math.cos(angle2) * pot_radius
            z2 = math.sin(angle2) * pot_radius
            glVertex3f(x1, 0, z1)
            glVertex3f(x2, 0, z2)
            glVertex3f(x2, pot_height, z2)
            glVertex3f(x1, pot_height, z1)
        glEnd()
        
        # Plant leaves (smaller)
        glColor3f(*self.colors['plant'])
        glTranslatef(0, pot_height, 0)
        leaf_size = 0.15
        num_leaves = 6
        for i in range(num_leaves):
            angle = (i / num_leaves) * 2 * math.pi
            x = math.cos(angle) * leaf_size
            z = math.sin(angle) * leaf_size
            glBegin(GL_TRIANGLES)
            glVertex3f(0, 0, 0)
            glVertex3f(x, leaf_size, z)
            glVertex3f(z, leaf_size/2, -x)
            glEnd()
        
        glPopMatrix()
        
    def draw(self):
        # Set material properties
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Draw floor at Y=0
        glBegin(GL_QUADS)
        glColor3f(*self.colors['floor'])
        glNormal3f(0, 1, 0)
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 0, -self.size)
        glEnd()
        
        # Draw walls starting from floor level
        glBegin(GL_QUADS)
        glColor3f(*self.colors['walls'])
        
        # Front wall
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 2, -self.size)
        glVertex3f(-self.size, 2, -self.size)
        
        # Back wall
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 2, self.size)
        glVertex3f(-self.size, 2, self.size)
        
        # Left wall
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(-self.size, 2, self.size)
        glVertex3f(-self.size, 2, -self.size)
        
        # Right wall
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 2, self.size)
        glVertex3f(self.size, 2, -self.size)
        glEnd()
        
        # Draw office furniture in a more realistic arrangement
        # HR Area (left side)
        self.draw_desk(-4, -2, 90)
        self.draw_chair(-3.5, -2, 90)
        self.draw_partition_walls(-4, -2)  # Add booth walls for HR
        
        # CEO Area (right side)
        self.draw_desk(4, 1, -90)
        self.draw_chair(3.5, 1, -90)
        self.draw_partition_walls(4, 1)  # Add booth walls for CEO
        
        # Plants in corners (moved closer to walls)
        self.draw_plant(-4.5, -4.5)
        self.draw_plant(4.5, -4.5)
        self.draw_plant(-4.5, 4.5)
        self.draw_plant(4.5, 4.5)

    def draw_partition_walls(self, x, z):
        """Draw booth partition walls - all surfaces in solid gray"""
        glColor3f(0.3, 0.3, 0.3)  # Solid gray for all walls
        
        # Back wall (smaller and thinner)
        glPushMatrix()
        glTranslatef(x, 0, z)
        glScalef(0.05, 1.0, 1.0)  # Thinner wall, normal height, shorter length
        draw_cube()  # Replace glutSolidCube with draw_cube
        glPopMatrix()
        
        # Side wall (smaller and thinner)
        glPushMatrix()
        glTranslatef(x, 0, z + 0.5)  # Moved closer
        glRotatef(90, 0, 1, 0)
        glScalef(0.05, 1.0, 0.8)  # Thinner wall, normal height, shorter length
        draw_cube()  # Replace glutSolidCube with draw_cube
        glPopMatrix()

class Player:
    def __init__(self):
        self.pos = [0, 0.5, 0]  # Lowered Y position to be just above floor
        self.rot = [0, 0, 0]
        self.speed = 0.3
        self.mouse_sensitivity = 0.5
        
    def move(self, dx, dz):
        # Convert rotation to radians (negative because OpenGL uses clockwise rotation)
        angle = math.radians(-self.rot[1])
        
        # Calculate movement vector
        move_x = (dx * math.cos(angle) + dz * math.sin(angle)) * self.speed
        move_z = (-dx * math.sin(angle) + dz * math.cos(angle)) * self.speed
        
        # Calculate new position
        new_x = self.pos[0] + move_x
        new_z = self.pos[2] + move_z
        
        # Wall collision check (room is 10x10)
        room_limit = 4.5  # Slightly less than room size/2 to prevent wall clipping
        if abs(new_x) < room_limit:
            self.pos[0] = new_x
        if abs(new_z) < room_limit:
            self.pos[2] = new_z

    def update_rotation(self, dx, dy):
        # Multiply mouse movement by sensitivity for faster turning
        self.rot[1] += dx * self.mouse_sensitivity

class NPC:
    def __init__(self, x, y, z, role="HR"):
        self.scale = 0.6  # Make NPCs smaller (about 60% of current size)
        # Position them beside the desks, at ground level
        # Adjust Y position to be half their height (accounting for scale)
        self.pos = [x, 0.65, z]  # This puts their feet on the ground
        self.size = 0.5
        self.role = role
        
        # Enhanced color palette
        self.skin_color = (0.8, 0.7, 0.6)  # Neutral skin tone
        self.hair_color = (0.2, 0.15, 0.1) if role == "HR" else (0.3, 0.3, 0.3)  # Dark brown vs gray
        
        # Updated clothing colors
        if role == "HR":
            self.clothes_primary = (0.8, 0.2, 0.2)    # Bright red
            self.clothes_secondary = (0.6, 0.15, 0.15) # Darker red
        else:  # CEO
            self.clothes_primary = (0.2, 0.3, 0.8)    # Bright blue
            self.clothes_secondary = (0.15, 0.2, 0.6)  # Darker blue

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glScalef(self.scale, self.scale, self.scale)
        
        # Head
        glColor3f(*self.skin_color)
        draw_sphere(0.12, 16, 16)
        
        # Hair (slightly larger than head)
        glColor3f(*self.hair_color)
        glPushMatrix()
        glTranslatef(0, 0.05, 0)  # Slightly above head
        draw_sphere(0.13, 16, 16)
        glPopMatrix()
        
        # Body (torso)
        glColor3f(*self.clothes_primary)
        glPushMatrix()
        glTranslatef(0, -0.3, 0)  # Move down from head
        glScalef(0.3, 0.4, 0.2)   # Make it rectangular
        draw_cube()
        glPopMatrix()
        
        # Arms
        glColor3f(*self.clothes_secondary)
        for x_offset in [-0.2, 0.2]:  # Left and right arms
            glPushMatrix()
            glTranslatef(x_offset, -0.3, 0)
            glScalef(0.1, 0.4, 0.1)
            draw_cube()
            glPopMatrix()
        
        # Legs
        glColor3f(*self.clothes_secondary)
        for x_offset in [-0.1, 0.1]:  # Left and right legs
            glPushMatrix()
            glTranslatef(x_offset, -0.8, 0)
            glScalef(0.1, 0.5, 0.1)
            draw_cube()
            glPopMatrix()
        
        glPopMatrix()

class MenuScreen:
    def __init__(self):
        self.font_large = pygame.font.Font(None, 74)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.active = True
        self.start_time = time.time()
        
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Create a surface for 2D rendering
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        
        # Calculate vertical positions
        center_y = WINDOW_HEIGHT // 2
        title_y = center_y - 100
        subtitle_y = center_y - 20
        prompt_y = center_y + 100
        
        # Render title with "typing" effect
        elapsed_time = time.time() - self.start_time
        title_chars = int(min(len(TITLE), elapsed_time * 15))  # Type 15 chars per second
        partial_title = TITLE[:title_chars]
        title_surface = self.font_large.render(partial_title, True, MENU_TEXT_COLOR)
        title_x = (WINDOW_WIDTH - title_surface.get_width()) // 2
        surface.blit(title_surface, (title_x, title_y))
        
        # Render subtitle with fade-in effect
        if elapsed_time > len(TITLE) / 15:  # Start after title is typed
            subtitle_alpha = min(255, int((elapsed_time - len(TITLE) / 15) * 255))
            subtitle_surface = self.font_medium.render(SUBTITLE, True, MENU_TEXT_COLOR)
            subtitle_surface.set_alpha(subtitle_alpha)
            subtitle_x = (WINDOW_WIDTH - subtitle_surface.get_width()) // 2
            surface.blit(subtitle_surface, (subtitle_x, subtitle_y))
        
        # Render "Press ENTER" with blinking effect
        if elapsed_time > (len(TITLE) / 15 + 1):  # Start after subtitle fade
            if int(elapsed_time * 2) % 2:  # Blink every 0.5 seconds
                prompt_text = "Press ENTER to start"
                prompt_surface = self.font_small.render(prompt_text, True, MENU_TEXT_COLOR)
                prompt_x = (WINDOW_WIDTH - prompt_surface.get_width()) // 2
                surface.blit(prompt_surface, (prompt_x, prompt_y))
        
        # Add some retro effects (scanlines)
        for y in range(0, WINDOW_HEIGHT, 4):
            pygame.draw.line(surface, (0, 50, 0), (0, y), (WINDOW_WIDTH, y))
        
        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        
        # Set up orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Render the texture in OpenGL
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Draw the texture
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 0); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        # Reset OpenGL state for 3D rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

# Modify the Game3D class to include the menu
class Game3D:
    def __init__(self):
        print("Game Initialization Game3D")
        self.menu = MenuScreen()
        self.player = Player()
        self.world = World()
        self.dialogue = DialogueSystem()
        self.hr_npc = NPC(-3.3, 0, -2, "HR")  # Moved beside the desk
        self.ceo_npc = NPC(3.3, 0, 1, "CEO")  # Moved beside the desk
        self.interaction_distance = 2.0
        self.last_interaction_time = 0

    def move_player_away_from_npc(self, npc_pos):
        # Calculate direction vector from NPC to player
        dx = self.player.pos[0] - npc_pos[0]
        dz = self.player.pos[2] - npc_pos[2]
        
        # Normalize the vector
        distance = math.sqrt(dx*dx + dz*dz)
        if distance > 0:
            dx /= distance
            dz /= distance
        
        # Move player back by 3 units
        self.player.pos[0] = npc_pos[0] + (dx * 3)
        self.player.pos[2] = npc_pos[2] + (dz * 3)

    def run(self):
        running = True
        while running:
            if self.menu.active:
                # Menu loop
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN and time.time() - self.menu.start_time > (len(TITLE) / 15 + 1):
                            self.menu.active = False
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                
                self.menu.render()
            else:
                # Main game loop
                # print(f"Player pos: {self.player.pos}, Rot: {self.player.rot}")
                current_time = time.time()
                if current_time - self.last_interaction_time > 0.5:
                    dx_hr = self.player.pos[0] - self.hr_npc.pos[0]
                    dz_hr = self.player.pos[2] - self.hr_npc.pos[2]
                    hr_distance = math.sqrt(dx_hr*dx_hr + dz_hr*dz_hr)
                    # print(f"HR Distance: {hr_distance:.2f}") # Add this

                    dx_ceo = self.player.pos[0] - self.ceo_npc.pos[0]
                    dz_ceo = self.player.pos[2] - self.ceo_npc.pos[2]
                    ceo_distance = math.sqrt(dx_ceo*dx_ceo + dz_ceo*dz_ceo)
                    # print(f"CEO Distance: {ceo_distance:.2f}") # Add this
                    # print(f"Dialogue Active: {self.dialogue.active}") # Add this
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if self.dialogue.active:
                            result = self.dialogue.handle_input(event)
                            if isinstance(result, dict) and result.get("command") == "move_player_back":
                                # Move player away from the current NPC
                                current_npc = self.hr_npc if self.dialogue.current_npc == "HR" else self.ceo_npc
                                self.move_player_away_from_npc(current_npc.pos)
                                
                        elif event.key == pygame.K_ESCAPE:
                            pygame.mouse.set_visible(True)
                            pygame.event.set_grab(False)
                            running = False
                        
                        # Handle dialogue input and check for exit command
                    elif event.type == pygame.MOUSEMOTION:
                        if not self.dialogue.active:
                            x, y = event.rel
                            self.player.update_rotation(x, y)

                # Handle keyboard input for movement (keep this blocked during dialogue)
                if not self.dialogue.active:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_w]: self.player.move(0, -1)
                    if keys[pygame.K_s]: self.player.move(0, 1)
                    if keys[pygame.K_a]: self.player.move(-1, 0)
                    if keys[pygame.K_d]: self.player.move(1, 0)

                # Check NPC interactions
                current_time = time.time()
                if current_time - self.last_interaction_time > 0.5:  # Cooldown on interactions
                    # Check distance to HR NPC
                    dx = self.player.pos[0] - self.hr_npc.pos[0]
                    dz = self.player.pos[2] - self.hr_npc.pos[2]
                    hr_distance = math.sqrt(dx*dx + dz*dz)
                    
                    # Check distance to CEO NPC
                    dx = self.player.pos[0] - self.ceo_npc.pos[0]
                    dz = self.player.pos[2] - self.ceo_npc.pos[2]
                    ceo_distance = math.sqrt(dx*dx + dz*dz)
                    
                    if hr_distance < self.interaction_distance and not self.dialogue.active:
                        self.dialogue.start_conversation("HR", self.player.pos)
                        self.last_interaction_time = current_time
                    elif ceo_distance < self.interaction_distance and not self.dialogue.active:
                        self.dialogue.start_conversation("CEO", self.player.pos)
                        self.last_interaction_time = current_time

                # Clear the screen and depth buffer
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Save the current matrix
                glPushMatrix()

                # Apply player rotation and position
                glRotatef(self.player.rot[0], 1, 0, 0)
                glRotatef(self.player.rot[1], 0, 1, 0)
                glTranslatef(-self.player.pos[0], -self.player.pos[1], -self.player.pos[2])

                # Draw the world and NPCs
                self.world.draw()
                self.hr_npc.draw()
                self.ceo_npc.draw()

                # Restore the matrix
                glPopMatrix()

                # Render dialogue system (if active)
                self.dialogue.render()
                self.dialogue.update()

                # Swap the buffers
                pygame.display.flip()

                # Maintain 60 FPS
                pygame.time.Clock().tick(60)

        self.dialogue.stop_recording()
        pygame.quit()
        sys.exit()

# Create and run game
game = Game3D()
game.run()