import os
import pyaudio
import wave
import tempfile
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables automatically
print("\n=== Starting Whisper Chat Example... ===")

# First try to find .env file automatically
dotenv_path = find_dotenv(usecwd=True)

# If not found, look in specific locations
if not dotenv_path:
    # Try project root (up to 4 levels)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(5):  # Check current and 4 parent levels
        potential_env = os.path.join(current_dir, '.env')
        if os.path.exists(potential_env):
            dotenv_path = potential_env
            print(f"Found .env file at: {dotenv_path}")
            break
        # Move up one directory
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root directory
            break
        current_dir = parent_dir

# Load the found .env file
if dotenv_path:
    print(f"Loading environment from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print("No .env file found. Will try to use environment variables directly.")

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in a .env file or as an environment variable.")
print("OpenAI API key loaded successfully")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
print("OpenAI client initialized")
print("=== Startup complete ===\n")

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Default recording duration

def record_audio(filename, duration=RECORD_SECONDS):
    """Record audio and save to file"""
    audio = pyaudio.PyAudio()
    
    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording... Press Ctrl+C to stop")
    frames = []
    
    # Create an event to mark recording stop
    stop_event = threading.Event()
    
    def listen_for_keyboard_interrupt():
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
    
    # Start listening thread
    keyboard_thread = threading.Thread(target=listen_for_keyboard_interrupt)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    # Record audio
    try:
        start_time = time.time()
        while not stop_event.is_set() and (time.time() - start_time < duration):
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    finally:
        print("Recording finished")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save recording file
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

def transcribe_audio(audio_file):
    """Transcribe audio using Whisper API"""
    try:
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=file
            )
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def get_chatgpt_response(prompt):
    """Get response from ChatGPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Or use other available models
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting ChatGPT response: {e}")
        return None

def main():
    print("Welcome to the Voice Chat Program!")
    print("Speak to chat with ChatGPT. Each conversation will record for 5 seconds")
    print("Press Ctrl+C during recording to end early")
    print("Enter 'q' to quit the program")
    
    while True:
        # Wait for the user to be ready to start recording
        command = input("\nPress Enter to start recording (or enter 'q' to quit): ")
        if command.lower() == 'q':
            break
        
        # Create temporary file for saving the recording
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        try:
            # Record audio
            record_audio(temp_filename)
            
            # Transcribe audio using Whisper
            print("Processing your speech...")
            transcript = transcribe_audio(temp_filename)
            
            if transcript:
                print(f"\nYou said: {transcript}")
                
                # Get ChatGPT response
                print("Waiting for ChatGPT response...")
                response = get_chatgpt_response(transcript)
                
                if response:
                    print(f"\nChatGPT: {response}")
                else:
                    print("Unable to get ChatGPT response")
            else:
                print("Unable to transcribe audio")
                
        finally:
            # Delete temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

if __name__ == "__main__":
    main() 