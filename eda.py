import csv
import json
import numpy as np

def run_eda():
    csv_file = "Student Social Media And Mental Health Impact.csv"
    
    # 1. Read dataset
    data = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    print(f"Loaded {len(data)} rows from {csv_file}")
    
    # 2. Extract numerical columns
    ages = []
    daily_usage = []
    unlocks = []
    study_hours = []
    physical_activity = []
    sleep_hours = []
    mental_health_scores = []
    
    # Categorical distributions
    platforms = {}
    genders = {}
    stress_levels = {}
    countries = {}
    
    # Aggregated lists for stress level screen time
    stress_screentime = {"Low": [], "Medium": [], "High": [], "Very High": []}
    
    for row in data:
        try:
            # Parse numerical fields
            age = float(row["Age"])
            usage = float(row["Avg_Daily_Usage_Hours"])
            unlk = float(row["Daily_Unlocks"])
            study = float(row["Study_Hours"])
            phys = float(row["Physical_Activity_Hours"])
            slp = float(row["Sleep_Hours_Per_Night"])
            mh_score = float(row["Mental_Health_Score"])
            
            ages.append(age)
            daily_usage.append(usage)
            unlocks.append(unlk)
            study_hours.append(study)
            physical_activity.append(phys)
            sleep_hours.append(slp)
            mental_health_scores.append(mh_score)
            
            # Categorical counts
            platform = row["Most_Used_Platform"]
            platforms[platform] = platforms.get(platform, 0) + 1
            
            gender = row["Gender"]
            genders[gender] = genders.get(gender, 0) + 1
            
            stress = row["Stress_Level"]
            stress_levels[stress] = stress_levels.get(stress, 0) + 1
            if stress in stress_screentime:
                stress_screentime[stress].append(usage)
                
            country = row["Country"]
            countries[country] = countries.get(country, 0) + 1
            
        except ValueError as e:
            # Skip rows with missing or corrupted values
            continue

    # Convert to numpy arrays for calculation
    ages_arr = np.array(ages)
    usage_arr = np.array(daily_usage)
    unlocks_arr = np.array(unlocks)
    study_arr = np.array(study_hours)
    phys_arr = np.array(physical_activity)
    sleep_arr = np.array(sleep_hours)
    mh_arr = np.array(mental_health_scores)
    
    total_valid = len(ages)
    print(f"Valid records processed: {total_valid}")
    
    # Compute Averages
    averages = {
        "age": float(np.mean(ages_arr)),
        "screen_time": float(np.mean(usage_arr)),
        "unlocks": float(np.mean(unlocks_arr)),
        "study_hours": float(np.mean(study_arr)),
        "physical_activity": float(np.mean(phys_arr)),
        "sleep": float(np.mean(sleep_arr)),
        "mental_health_score": float(np.mean(mh_arr))
    }
    
    # Compute screen time per stress level
    stress_screentime_avg = {}
    for level, vals in stress_screentime.items():
        stress_screentime_avg[level] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    # Compute correlation matrix
    # Columns to correlate: screen_time, unlocks, study, physical_activity, sleep, mental_health
    matrix_data = np.vstack([usage_arr, unlocks_arr, study_arr, phys_arr, sleep_arr, mh_arr])
    corr_matrix = np.corrcoef(matrix_data)
    
    # Prepare correlation dictionary for JSON
    corr_labels = ["Screen Time", "Daily Unlocks", "Study Hours", "Physical Activity", "Sleep Hours", "Mental Health Score"]
    corr_data = {}
    for i, label1 in enumerate(corr_labels):
        corr_data[label1] = {}
        for j, label2 in enumerate(corr_labels):
            corr_data[label1][label2] = float(corr_matrix[i, j])

    # Sample a scatter dataset for screen time vs mental health (limit to 100 points for chart size)
    np.random.seed(42)
    indices = np.random.choice(total_valid, min(total_valid, 150), replace=False)
    scatter_data = []
    for idx in indices:
        scatter_data.append({
            "x": float(usage_arr[idx]),
            "y": float(mh_arr[idx]),
            "stress": data[idx]["Stress_Level"]
        })
        
    # High risk calculation (defined as Stress Level in ['High', 'Very High'] and Screen Time > 5 hours)
    high_risk_count = sum(1 for row in data if row.get("Stress_Level") in ["High", "Very High"] and float(row.get("Avg_Daily_Usage_Hours", 0)) > 5.0)
    high_risk_ratio = float(high_risk_count) / len(data) if len(data) > 0 else 0.0

    # Save to export dictionary
    export_data = {
        "total_records": len(data),
        "valid_records": total_valid,
        "high_risk_ratio": high_risk_ratio,
        "averages": averages,
        "platform_distribution": platforms,
        "gender_distribution": genders,
        "stress_level_distribution": stress_levels,
        "stress_screentime_avg": stress_screentime_avg,
        "correlation_matrix": corr_data,
        "screentime_vs_mentalhealth": scatter_data
    }
    
    output_file = "data_export.json"
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(export_data, out_f, indent=2)
        
    print(f"EDA successfully completed. Statistics written to {output_file}")

if __name__ == "__main__":
    run_eda()
