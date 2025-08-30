import pandas as pd
import os
from io import StringIO

def get_emotion_trend(user_id: str = None):
    """Get emotion trend data with user isolation support"""
    csv_file = "visualization/emotion_trend.csv"
    if not os.path.exists(csv_file):
        # Return empty dict to avoid plotting errors
        return {
            "dates": [],
            "sadness": [],
            "joy": [],
            "anger": [],
            "intensity": []
        }
    
    try:
        # Try to read CSV file, handle mixed formats
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Separate different format data
        old_format_data = []
        new_format_data = []
        
        for line in lines[1:]:  # Skip header row
            parts = line.strip().split(',')
            if len(parts) == 5:  # Old format
                old_format_data.append(line)
            elif len(parts) == 6:  # New format
                new_format_data.append(line)
        
        # Process old format data
        old_df = None
        if old_format_data:
            old_df = pd.read_csv(StringIO('timestamp,anger,sadness,joy,intensity\n' + ''.join(old_format_data)))
        
        # Process new format data
        new_df = None
        if new_format_data:
            new_df = pd.read_csv(StringIO('user_id,timestamp,anger,sadness,joy,intensity\n' + ''.join(new_format_data)))
            if user_id:
                new_df = new_df[new_df["user_id"] == user_id]
        
        # Merge data
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
        
        # Process timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.date  # Extract date

        # Group by date and calculate average, ensure numeric columns are correct
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
        print(f"Error reading emotion trend data: {e}")
        return {
            "dates": [],
            "sadness": [],
            "joy": [],
            "anger": [],
            "intensity": []
        }