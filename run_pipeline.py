
from speechChinese_baidu_ import BaiduSpeechRecognizer
from emotion_module import load_mapped_liwc, liwc_score, category_code_to_name

def main():
    print("📥 正在加载情绪词典...")
    liwc_dict = load_mapped_liwc("sc_liwc.dic", category_code_to_name)

    print("🎤 初始化百度语音识别模块...")
    recognizer = BaiduSpeechRecognizer()

    print("⏺️ 录音中，请说话（持续 10 秒）...")
    text, _ = recognizer.record_and_recognize(duration=10)

    if text:
        print("\n📝 语音识别结果:")
        print(text)

        print("\n📊 情绪识别结果:")
        scores = liwc_score(text, liwc_dict)
        print("\n📊 情绪维度词频:")
        for category, count in scores.items():
            print(f"{category}: {count}")
    else:
        print("⚠️ 识别失败，请重试。")

if __name__ == "__main__":
    main()
