# age_bias_analysis.py

from run_verification import results
from evaluation import evaluate_results
import os



# get age from UTK filename

def extract_age(filepath):
    """
    UTKFace filename format:
    age_gender_race_date.jpg.chip.jpg
    Example: 24_1_3_20170104220232144.jpg.chip.jpg
    """
    filename = os.path.basename(filepath)
    age = int(filename.split("_")[0])
    return age



# Assigning age group

def get_age_group(age):
    if age <= 17:
        return "child"
    elif age <= 35:
        return "young_adult"
    elif age <= 55:
        return "middle_age"
    else:
        return "senior"



# Group results by age

def group_by_age(results):
    grouped = {
        "child": [],
        "young_adult": [],
        "middle_age": [],
        "senior": []
    }

    for r in results:
        age1 = extract_age(r["img1"])
        age2 = extract_age(r["img2"])

        
        avg_age = (age1 + age2) / 2
        group = get_age_group(avg_age)

        grouped[group].append(r)

    return grouped



# Main analysis

def age_bias_analysis(results, threshold=0.70):
    print("\n=== AGE GROUP ANALYSIS ===")

    grouped = group_by_age(results)

    for group, group_results in grouped.items():
        if len(group_results) == 0:
            continue

        metrics = evaluate_results(group_results, threshold)

        print(f"\nGroup: {group}")
        print(f"Samples: {len(group_results)}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"FAR: {metrics['FAR']:.3f}")
        print(f"FRR: {metrics['FRR']:.3f}")



# Run

if __name__ == "__main__":
    age_bias_analysis(results, threshold=0.70)
