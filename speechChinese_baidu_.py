import os
import sys
import time
import wave
import tempfile
import platform
from typing import Optional, Tuple, Dict, Any
import logging
import threading
import queue
import json
import pyaudio
from aip import AipSpeech

class BaiduSpeechRecognizer:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Baidu Speech Recognition client"""
        # Setup
        self._setup_logging()
        
        # Security warning
        self._show_security_warning()
        
        # Load config from environment or config file
        self.config = self._load_config(config_path)
        self.APP_ID = self.config['app_id']
        self.API_KEY = self.config['api_key']
        self.SECRET_KEY = self.config['secret_key']
        
        # Validate config
        if not all([self.APP_ID, self.API_KEY, self.SECRET_KEY]):
            raise ValueError("Incomplete API configuration, please set environment variables or config file")
        
        # Initialize client
        self.client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        
        # Audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.MAX_DURATION = 60  # Maximum recording duration (seconds)
        self.SILENCE_THRESHOLD = 300  # Silence threshold
        self.SILENCE_DURATION = 2  # Wait time after detecting silence (seconds)
        
        # Error code mapping
        self.ERROR_MAP = {
            3300: 'Input parameter error',
            3301: 'Audio quality too poor',
            3302: 'Authentication failed',
            3303: 'Internal server error',
            3304: 'Request limit exceeded',
            3305: 'Service not enabled',
            3307: 'Recognition engine busy',
            3308: 'Audio too long (exceeds 60 seconds)',
            3309: 'Audio data problem',
            3310: 'Invalid audio format',
            3311: 'Unsupported sample rate',
            3312: 'Audio decoding failed',
            3313: 'Speech too short',
            3314: 'Engine no response'
        }
        
        # Recording state
        self._is_recording = False
        self._audio_queue = queue.Queue()
        
        # Recognition callback
        self._callback = None

    def _setup_logging(self):
        """Configure logging system"""
        self.logger = logging.getLogger('baidu_speech')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter and add to handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(ch)

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration, priority: environment variables > config file > defaults"""
        config = {
            'app_id': os.getenv('BAIDU_APP_ID', '6965317'),
            'api_key': os.getenv('BAIDU_API_KEY', 'K8lDyto3dXTlz6hI0AdhzbtV'),
            'secret_key': os.getenv('BAIDU_SECRET_KEY', 'Bj8Ajgj0GcfdzKLUywAjtWXafOibnZYM'),
            'timeout': int(os.getenv('BAIDU_TIMEOUT', 10)),  # API request timeout
            'retry_times': int(os.getenv('BAIDU_RETRY_TIMES', 3)),  # Retry attempts
            'retry_delay': float(os.getenv('BAIDU_RETRY_DELAY', 1.0)),  # Retry delay (seconds)
        }
        
        # If config path provided, try loading from file
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                    self.logger.info(f"Loaded config from file: {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
        
        return config

    def _show_security_warning(self):
        """Display security warning"""
        warning = """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SECURITY WARNING: Protect your API keys!
        1. Never commit keys to version control
        2. Recommended to use environment variables or config files
        3. Rotate your keys regularly
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        self.logger.warning(warning)
        time.sleep(1)  # Ensure user sees the warning

    def recognize_file(self, file_path: str, add_punctuation: bool = True, 
                       dev_pid: int = 1537) -> Optional[str]:
        """
        Recognize speech in audio file
        :param file_path: Path to audio file
        :param add_punctuation: Whether to add punctuation
        :param dev_pid: Recognition model ID, default is Mandarin Chinese
        :return: Recognized text
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None

        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Validate audio length
            if len(audio_data) > 10 * 1024 * 1024:  # 10MB limit
                self.logger.error("Audio file too large (exceeds 10MB)")
                return None
            
            # Get file format
            file_format = self._get_file_format(file_path)
            
            # Call recognition API with retry
            result = None
            for attempt in range(self.config['retry_times']):
                try:
                    result = self.client.asr(
                        audio_data, 
                        file_format, 
                        self.RATE, 
                        {
                            'dev_pid': dev_pid,
                            'ptt': 1 if add_punctuation else 0,  # Add punctuation
                            'cuid': self._generate_cuid(),  # Generate unique device ID
                        }
                    )
                    break
                except Exception as e:
                    if attempt < self.config['retry_times'] - 1:
                        self.logger.warning(f"Recognition attempt {attempt+1} failed: {str(e)}, retrying...")
                        time.sleep(self.config['retry_delay'])
                    else:
                        self.logger.error(f"All retries failed: {str(e)}")
                        return None
            
            return self._process_result(result)
        
        except Exception as e:
            self.logger.error(f"File recognition failed: {str(e)}")
            return None

    def _get_file_format(self, file_path: str) -> str:
        """Get valid file format"""
        ext = os.path.splitext(file_path)[1][1:].lower()
        valid_formats = ['wav', 'pcm', 'amr', 'm4a']
        return ext if ext in valid_formats else 'wav'

    def _generate_cuid(self) -> str:
        """Generate unique device ID"""
        import uuid
        return str(uuid.getnode())

    def record_and_recognize(self, duration: int = 5, add_punctuation: bool = True,
                            dev_pid: int = 1537) -> Tuple[Optional[str], Optional[str]]:
        """
        Record audio and recognize
        :param duration: Recording duration (seconds)
        :param add_punctuation: Whether to add punctuation
        :param dev_pid: Recognition model ID, default is Mandarin Chinese
        :return: (Recognition result, temporary file path)
        """
        if duration <= 0 or duration > self.MAX_DURATION:
            self.logger.error(f"Duration should be between 1-{self.MAX_DURATION} seconds")
            return None, None

        try:
            # Create temp file
            temp_path = self._create_temp_file()
            
            # Record audio
            if not self._record_audio(temp_path, duration):
                return None, None
            
            # Recognize audio
            result = self.recognize_file(temp_path, add_punctuation, dev_pid)
            
            return result, temp_path
            
        except Exception as e:
            self.logger.error(f"Recording recognition failed: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None, None

    def start_continuous_recording(self, callback, silence_detection: bool = True):
        """
        Start continuous recording recognition
        :param callback: Recognition result callback function
        :param silence_detection: Whether to enable silence detection
        """
        if self._is_recording:
            self.logger.warning("Already recording")
            return
        
        self._callback = callback
        self._is_recording = True
        
        # Start recording thread
        self._recording_thread = threading.Thread(
            target=self._continuous_recording_loop,
            args=(silence_detection,)
        )
        self._recording_thread.daemon = True
        self._recording_thread.start()
        
        self.logger.info("Starting continuous recording...")

    def stop_continuous_recording(self):
        """Stop continuous recording recognition"""
        self._is_recording = False
        if hasattr(self, '_recording_thread'):
            self._recording_thread.join(timeout=2.0)
            self.logger.info("Continuous recording stopped")

    def _continuous_recording_loop(self, silence_detection: bool):
        """Continuous recording loop"""
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            frames = []
            silence_frames = 0
            silence_threshold_frames = int(self.SILENCE_DURATION * self.RATE / self.CHUNK)
            
            while self._is_recording:
                data = stream.read(self.CHUNK)
                frames.append(data)
                
                # Silence detection
                if silence_detection:
                    rms = self._calculate_rms(data)
                    if rms < self.SILENCE_THRESHOLD:
                        silence_frames += 1
                    else:
                        silence_frames = 0
                    
                    # If enough silence detected, process current audio
                    if silence_frames >= silence_threshold_frames and len(frames) > silence_threshold_frames:
                        self._process_audio_chunk(frames[:-silence_threshold_frames])
                        frames = frames[-silence_threshold_frames:]
                        silence_frames = 0
                
                # Limit memory usage
                if len(frames) > int(self.MAX_DURATION * self.RATE / self.CHUNK):
                    self._process_audio_chunk(frames)
                    frames = []
            
            # Process remaining audio
            if frames:
                self._process_audio_chunk(frames)
                
        except Exception as e:
            self.logger.error(f"Continuous recording error: {str(e)}")
            if self._callback:
                self._callback(None, f"Recording error: {str(e)}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()

    def _calculate_rms(self, data: bytes) -> float:
        """Calculate RMS of audio data for silence detection"""
        import struct
        count = len(data) / 2
        format = "%dh" % (count)
        shorts = struct.unpack(format, data)
        
        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0 / 32768)
            sum_squares += n * n
            
        return (sum_squares / count) ** 0.5 * 1000

    def _process_audio_chunk(self, frames: list):
        """Process audio chunk"""
        if not frames:
            return
            
        temp_path = self._create_temp_file()
        
        try:
            # Save audio chunk
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Async recognition
            threading.Thread(
                target=self._async_recognize,
                args=(temp_path,)
            ).start()
            
        except Exception as e:
            self.logger.error(f"Failed to process audio chunk: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _async_recognize(self, file_path: str):
        """Asynchronously recognize audio file"""
        try:
            result = self.recognize_file(file_path)
            
            if self._callback:
                self._callback(result, None)
        except Exception as e:
            self.logger.error(f"Async recognition failed: {str(e)}")
            if self._callback:
                self._callback(None, f"Recognition error: {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def _create_temp_file(self) -> str:
        """Create temporary audio file"""
        temp_dir = tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', 
            dir=temp_dir,
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        self.logger.debug(f"Created temp file: {temp_path}")
        return temp_path

    def _record_audio(self, file_path: str, duration: int) -> bool:
        """Record audio to file"""
        try:
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self.logger.info(f"Starting recording ({duration} seconds)...")
            frames = []
            
            try:
                for i in range(0, int(self.RATE / self.CHUNK * duration)):
                    if i % (self.RATE // self.CHUNK) == 0 and i > 0:
                        self.logger.info(f"Time remaining: {duration - i // (self.RATE // self.CHUNK)} seconds")
                    data = stream.read(self.CHUNK)
                    frames.append(data)
            except KeyboardInterrupt:
                self.logger.info("Recording interrupted by user")
            
            self.logger.info("Recording finished")
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save recording
            wf = wave.open(file_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return True
            
        except OSError as e:
            self.logger.error(f"Recording device error: {str(e)}")
            return False

    def _process_result(self, result: dict) -> str:
        """Process recognition result"""
        if not result:
            raise Exception("No recognition result")
            
        if result.get('err_no') == 0:
            return result['result'][0].strip()
        else:
            error_msg = self.ERROR_MAP.get(result['err_no'], result.get('err_msg', 'Unknown error'))
            raise Exception(f"Recognition error [{result['err_no']}]: {error_msg}")

def check_dependencies():
    """Check required dependencies"""
    missing = []
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
    
    if missing:
        print("\nMissing dependencies:")
        for lib in missing:
            print(f"- {lib}")
        
        print("\nInstall commands:")
        if platform.system() == 'Windows':
            print("pip install pipwin")
            print("pipwin install pyaudio")
        else:
            print("pip install pyaudio")
        
        if 'pyaudio' in missing:
            print("\nWarning: Real-time recording will not be available")

def main():
    # Check dependencies
    check_dependencies()
    
    # Get config path from command line
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            config_path = None
    
    # Initialize recognizer
    try:
        recognizer = BaiduSpeechRecognizer(config_path)
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    while True:
        print("\n" + "="*30)
        print("Baidu Speech Recognition System")
        print("="*30)
        print("1. Recognize audio file")
        print("2. Fixed duration recording")
        print("3. Continuous recording")
        print("4. Exit")
        
        choice = input("\nSelect function (1/2/3/4): ").strip()
        
        if choice == '1':
            file_path = input("Enter audio file path: ").strip()
            if not file_path:
                continue
                
            # Optional parameters
            add_punctuation = input("Add punctuation? (y/n, default y): ").strip().lower() != 'n'
            dev_pid = int(input("Enter recognition model ID (default 1537-Mandarin): ") or 1537)
            
            start_time = time.time()
            result = recognizer.recognize_file(file_path, add_punctuation, dev_pid)
            
            if result:
                print("\nRecognition result:")
                print("-"*40)
                print(result)
                print("-"*40)
                print(f"Recognition time: {time.time()-start_time:.2f} seconds")
                
        elif choice == '2':
            try:
                duration = int(input(f"Enter recording duration (1-{recognizer.MAX_DURATION} seconds, default 5): ") or 5)
                add_punctuation = input("Add punctuation? (y/n, default y): ").strip().lower() != 'n'
                dev_pid = int(input("Enter recognition model ID (default 1537-Mandarin): ") or 1537)
                
                start_time = time.time()
                result, temp_path = recognizer.record_and_recognize(duration, add_punctuation, dev_pid)
                
                if result:
                    print("\nRecognition result:")
                    print("-"*40)
                    print(result)
                    print("-"*40)
                    print(f"Recognition time: {time.time()-start_time:.2f} seconds")
                
                # Ask to save recording
                if temp_path and os.path.exists(temp_path):
                    save = input("Save recording file? (y/n): ").lower()
                    if save == 'y':
                        new_path = input("Enter save path (empty for default name): ").strip() or f"recording_{int(time.time())}.wav"
                        os.replace(temp_path, new_path)
                        print(f"Recording saved to: {new_path}")
                    else:
                        os.unlink(temp_path)
                elif temp_path:
                    os.unlink(temp_path)
                    
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == '3':
            print("\nContinuous recording mode (Press Ctrl+C to stop)")
            print("-"*40)
            
            def callback(result, error):
                if error:
                    print(f"Error: {error}")
                elif result:
                    print(f"Recognition result: {result}")
            
            recognizer.start_continuous_recording(callback)
            
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                recognizer.stop_continuous_recording()
                
        elif choice == '4':
            print("\nExiting program")
            break
            
        else:
            print("Invalid option, please try again")
            
            

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nProgram error: {str(e)}")
        sys.exit(1)