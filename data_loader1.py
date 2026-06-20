import os
import pandas as pd
from pathlib import Path

def load_utkface(dataset_relative_path="data/UTKFace"):
    # 1. GENERATE  PATH
    # This finds  folder where script is saved 
    base_path = Path(__file__).resolve().parent
    dataset_path = base_path / dataset_relative_path

    if not dataset_path.exists():
        print(f" CRITICAL ERROR: Folder not found at {dataset_path}")
        print("Please ensure your 'data' folder is in the same directory as this script.")
        return pd.DataFrame()

    records = []

    
    for file_path in dataset_path.iterdir():
        
        
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in [".jpg", ".jpeg"]:
            continue

        # get just the filename string for splitting
        file_name = file_path.name
        parts = file_name.split("_")
        
        
        if len(parts) < 4:
            continue

        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            
            records.append({
                
                
                "image_path": str(file_path.resolve()),
                "age": age,
                "race": race,
                "gender": gender
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    print(f"✅ Successfully loaded {len(df)} images with verified paths.")
    return df

def assign_age_group(age):
    if age < 18:
        return "child"
    elif age < 35:
        return "young_adult"
    elif age < 60:
        return "adult"
    else:
        return "older_adult"

if __name__ == "__main__":
    
    df = load_utkface()

    if not df.empty:
        df["age_group"] = df["age"].apply(assign_age_group)
        
        print("\n--- Data Sample ---")
        print(df.head())
        
        print("\n--- Summary Statistics ---")
        print(df["age"].describe())
        
        print("\n--- Demographic Distribution ---")
        print(df["age_group"].value_counts())
    else:
        print("No records found. Check if images exist in the data/UTKFace folder.")