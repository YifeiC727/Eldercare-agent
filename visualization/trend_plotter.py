import pandas as pd
import os

def get_emotion_trend():
    if not os.path.exists("emotion_trend.csv"):
        #返回一个空的字典，避免绘图时出错
        return {
            "timestamp": [],
            "sadness": [],
            "joy": [],
            "anger": [],
            "intensity": []
        }
    df = pd.read_csv("emotion_trend.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date  # 提取日期

    df_daily = df.groupby("date").mean().reset_index()

    return {
        "dates": df_daily["date"].astype(str).tolist(),
        "sadness": df_daily["sadness"].round(2).tolist(),
        "joy": df_daily["joy"].round(2).tolist(),
        "anger": df_daily["anger"].round(2).tolist(),
        "intensity": df_daily["intensity"].round(2).tolist()
    }