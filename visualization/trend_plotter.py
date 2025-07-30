import pandas as pd
import os
from io import StringIO

def get_emotion_trend(user_id: str = None):
    """获取情绪趋势数据，支持用户隔离"""
    csv_file = "visualization/emotion_trend.csv"
    if not os.path.exists(csv_file):
        #返回一个空的字典，避免绘图时出错
        return {
            "dates": [],
            "sadness": [],
            "joy": [],
            "anger": [],
            "intensity": []
        }
    
    try:
        # 尝试读取CSV文件，处理混合格式
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 分离不同格式的数据
        old_format_data = []
        new_format_data = []
        
        for line in lines[1:]:  # 跳过标题行
            parts = line.strip().split(',')
            if len(parts) == 5:  # 旧格式
                old_format_data.append(line)
            elif len(parts) == 6:  # 新格式
                new_format_data.append(line)
        
                # 处理旧格式数据
        old_df = None
        if old_format_data:
            old_df = pd.read_csv(StringIO('timestamp,anger,sadness,joy,intensity\n' + ''.join(old_format_data)))
        
        # 处理新格式数据
        new_df = None
        if new_format_data:
            new_df = pd.read_csv(StringIO('user_id,timestamp,anger,sadness,joy,intensity\n' + ''.join(new_format_data)))
            if user_id:
                new_df = new_df[new_df["user_id"] == user_id]
        
        # 合并数据
        if old_df is not None and new_df is not None:
            df = pd.concat([old_df, new_df], ignore_index=True)
        elif old_df is not None:
            df = old_df
        elif new_df is not None:
            df = new_df
        else:
            return {
                "dates": [],
                "sadness": [],
                "joy": [],
                "anger": [],
                "intensity": []
            }
        
        # 处理时间戳
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.date  # 提取日期

        # 按日期分组计算平均值，确保数值列正确
        numeric_columns = ["sadness", "joy", "anger", "intensity"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_daily = df.groupby("date")[numeric_columns].mean().reset_index()

        result = {
            "dates": df_daily["date"].astype(str).tolist(),
            "sadness": df_daily["sadness"].round(2).tolist(),
            "joy": df_daily["joy"].round(2).tolist(),
            "anger": df_daily["anger"].round(2).tolist(),
            "intensity": df_daily["intensity"].round(2).tolist()
        }
        
        return result
        
    except Exception as e:
        print(f"读取情绪趋势数据时出错: {e}")
        return {
            "dates": [],
            "sadness": [],
            "joy": [],
            "anger": [],
            "intensity": []
        }