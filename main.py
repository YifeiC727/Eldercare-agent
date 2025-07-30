import time
from speech.baidu_speech_recognizer import BaiduSpeechRecognizer
from emotion_detection.emotion_recognizer import EmotionRecognizer
from strategy.strategy_selector import StrategySelector
from strategy.llm_generator import LLMGenerator
from dotenv import load_dotenv
load_dotenv()

def process_text(text, history, emotion_recog, selector, generator):
    history.append(text)
    print(f"识别文本: {text}")

    # 只分析本次识别文本的 LIWC
    liwc_score = emotion_recog.liwc_score(text)
    print("本次识别 LIWC 结果:", liwc_score)

    # 情绪识别
    emotion = emotion_recog.analyze_emotion_deepseek(text)
    print("情感识别结果:", emotion)

    # 策略选择
    strategy = selector.select_strategy(
        emotion,
        emotion.get("intensity", 0),
        history,
        liwc_score,
        text
    )
    print("策略选择结果:", strategy)

    # LLM生成回复
    reply = generator.generate_response(text, strategy)
    print("AI回复:", reply)
    print("-" * 60)

def main():
    recognizer = BaiduSpeechRecognizer()
    emotion_recog = EmotionRecognizer()
    selector = StrategySelector()
    generator = LLMGenerator()
    history = []

    while True:
        print("\n==============================")
        print("AI 陪伴助手语音系统")
        print("==============================")
        print("1. 识别音频文件")
        print("2. 固定时长录音")
        print("3. 连续录音（分段识别）")
        print("4. 退出")
        choice = input("\n请选择功能 (1/2/3/4): ").strip()

        if choice == "1":
            file_path = input("请输入音频文件路径: ").strip()
            text = recognizer.recognize_file(file_path)
            if text:
                process_text(text, history, emotion_recog, selector, generator)
            else:
                print("未识别到文本")

        elif choice == "2":
            try:
                duration = int(input("请输入录音时长（秒）: ").strip())
            except ValueError:
                print("请输入有效的数字！")
                continue
            text, _ = recognizer.record_and_recognize(duration)
            if text:
                process_text(text, history, emotion_recog, selector, generator)
            else:
                print("未识别到文本")

        elif choice == "3":
            print("\n连续录音模式（分段识别，每段自动分析，按回车键停止）")
            print("----------------------------------------")
            def callback(result, error=None):
                if error:
                    print("识别出错:", error)
                    return
                if result:
                    print("Recognition result:", result)
                    process_text(result, history, emotion_recog, selector, generator)
            recognizer.start_continuous_recording(callback)
            try:
                input("按回车键停止实时识别...\n")
            except KeyboardInterrupt:
                print("\n检测到 Ctrl+C，正在停止识别...")
            finally:
                recognizer.stop_continuous_recording()

        elif choice == "4":
            print("感谢使用，再见！")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()
