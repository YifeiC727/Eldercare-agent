import time
from speech.baidu_speech_recognizer import BaiduSpeechRecognizer
from emotion_detection.emotion_recognizer import EmotionRecognizer
from strategy.strategy_selector import StrategySelector
from strategy.llm_generator import LLMGenerator
from dotenv import load_dotenv
load_dotenv()

def process_text(text, history, emotion_recog, selector, generator):
    history.append(text)
    print(f"Recognized text: {text}")

    # Only analyze LIWC for current recognized text
    liwc_score = emotion_recog.liwc_score(text)
    print("Current recognition LIWC result:", liwc_score)

    # Emotion recognition
    emotion = emotion_recog.analyze_emotion_deepseek(text)
    print("Emotion recognition result:", emotion)

    # Strategy selection
    strategy = selector.select_strategy(
        emotion,
        emotion.get("intensity", 0),
        history,
        liwc_score,
        text
    )
    print("Strategy selection result:", strategy)

    # LLM response generation
    reply = generator.generate_response(text, strategy)
    print("AI reply:", reply)
    print("-" * 60)

def main():
    recognizer = BaiduSpeechRecognizer()
    emotion_recog = EmotionRecognizer()
    selector = StrategySelector()
    generator = LLMGenerator()
    history = []

    while True:
        print("\n==============================")
        print("AI Companion Voice System")
        print("==============================")
        print("1. Recognize audio file")
        print("2. Fixed duration recording")
        print("3. Continuous recording (segmented recognition)")
        print("4. Exit")
        choice = input("\nPlease choose function (1/2/3/4): ").strip()

        if choice == "1":
            file_path = input("Please enter audio file path: ").strip()
            text = recognizer.recognize_file(file_path)
            if text:
                process_text(text, history, emotion_recog, selector, generator)
            else:
                print("No text recognized")

        elif choice == "2":
            try:
                duration = int(input("Please enter recording duration (seconds): ").strip())
            except ValueError:
                print("Please enter a valid number!")
                continue
            text, _ = recognizer.record_and_recognize(duration)
            if text:
                process_text(text, history, emotion_recog, selector, generator)
            else:
                print("No text recognized")

        elif choice == "3":
            print("\nContinuous recording mode (segmented recognition, automatic analysis per segment, press Enter to stop)")
            print("----------------------------------------")
            def callback(result, error=None):
                if error:
                    print("Recognition error:", error)
                    return
                if result:
                    print("Recognition result:", result)
                    process_text(result, history, emotion_recog, selector, generator)
            recognizer.start_continuous_recording(callback)
            try:
                input("Press Enter to stop real-time recognition...\n")
            except KeyboardInterrupt:
                print("\nDetected Ctrl+C, stopping recognition...")
            finally:
                recognizer.stop_continuous_recording()

        elif choice == "4":
            print("Thank you for using, goodbye!")
            break
        else:
            print("Invalid option, please re-enter.")

if __name__ == "__main__":
    main()
