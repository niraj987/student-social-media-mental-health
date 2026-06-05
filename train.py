import csv
import json
import numpy as np

# Import custom models from the models directory
from models.decision_tree import DecisionTreeClassifier
from models.linear_regression import LinearRegression

def load_and_preprocess_data():
    csv_file = "Student Social Media And Mental Health Impact.csv"
    
    # 1. Read raw rows
    raw_data = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append(row)
            
    # 2. Extract unique values for categorical mapping
    categories = {
        "Gender": set(),
        "Academic_Level": set(),
        "Most_Used_Platform": set(),
        "Purpose_Of_Use": set()
    }
    
    for row in raw_data:
        for cat in categories:
            if row[cat]:
                categories[cat].add(row[cat])
                
    # Create encoding maps
    encoding_maps = {}
    for cat, vals in categories.items():
        # Sort values to ensure deterministic encoding
        encoding_maps[cat] = {val: i for i, val in enumerate(sorted(list(vals)))}
        
    stress_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Very High": 3
    }
    
    inverse_stress_mapping = {v: k for k, v in stress_mapping.items()}

    # 3. Parse features and targets
    feature_names = [
        "Age", 
        "Avg_Daily_Usage_Hours", 
        "Daily_Unlocks", 
        "Study_Hours", 
        "Physical_Activity_Hours", 
        "Sleep_Hours_Per_Night",
        "Gender",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use"
    ]
    
    X = []
    y_stress = []
    y_mental_health = []
    
    for row in raw_data:
        try:
            # Numerical parsing
            age = float(row["Age"])
            usage = float(row["Avg_Daily_Usage_Hours"])
            unlocks = float(row["Daily_Unlocks"])
            study = float(row["Study_Hours"])
            phys = float(row["Physical_Activity_Hours"])
            sleep = float(row["Sleep_Hours_Per_Night"])
            mh_score = float(row["Mental_Health_Score"])
            
            # Categorical mapping
            gender_code = encoding_maps["Gender"][row["Gender"]]
            academic_code = encoding_maps["Academic_Level"][row["Academic_Level"]]
            platform_code = encoding_maps["Most_Used_Platform"][row["Most_Used_Platform"]]
            purpose_code = encoding_maps["Purpose_Of_Use"][row["Purpose_Of_Use"]]
            
            stress_code = stress_mapping[row["Stress_Level"]]
            
            feature_vector = [
                age, 
                usage, 
                unlocks, 
                study, 
                phys, 
                sleep, 
                gender_code, 
                academic_code, 
                platform_code, 
                purpose_code
            ]
            
            X.append(feature_vector)
            y_stress.append(stress_code)
            y_mental_health.append(mh_score)
            
        except (ValueError, KeyError) as e:
            continue
            
    return np.array(X), np.array(y_stress), np.array(y_mental_health), feature_names, encoding_maps, stress_mapping, inverse_stress_mapping

def train_test_split(X, y1, y2, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y1[train_idx], y1[test_idx], y2[train_idx], y2[test_idx]

def main():
    print("Loading and preprocessing data...")
    X, y_stress, y_mh, feature_names, encoding_maps, stress_mapping, inverse_stress_mapping = load_and_preprocess_data()
    
    # Split data into train & test
    X_train, X_test, y_stress_train, y_stress_test, y_mh_train, y_mh_test = train_test_split(X, y_stress, y_mh, test_size=0.2)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # ------------------ Train Linear Regression (predict Mental Health Score) ------------------
    print("\n--- Training Linear Regression Model (predict Mental Health Score) ---")
    lr = LinearRegression()
    lr.fit(X_train, y_mh_train)
    
    # Evaluate Regression
    y_mh_pred = lr.predict(X_test)
    rmse = np.sqrt(np.mean((y_mh_test - y_mh_pred)**2))
    ss_res = np.sum((y_mh_test - y_mh_pred)**2)
    ss_tot = np.sum((y_mh_test - np.mean(y_mh_test))**2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    print(f"Linear Regression Results:")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R2) Score: {r2:.4f}")
    print(f"  Model Intercept: {lr.intercept:.4f}")
    for fname, coef in zip(feature_names, lr.coefficients):
        print(f"  Coefficient [{fname}]: {coef:.4f}")

    # ------------------ Train Decision Tree Classifier (predict Stress Level) ------------------
    print("\n--- Training Decision Tree Classifier (predict Stress Level) ---")
    # max_depth=6 splits well and keeps model file clean and light for browser JSON
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=5)
    dt.fit(X_train, y_stress_train)
    
    # Evaluate Classifier
    y_stress_pred = dt.predict(X_test)
    accuracy = np.mean(y_stress_test == y_stress_pred)
    print(f"Decision Tree Classifier Results:")
    print(f"  Classification Accuracy: {accuracy * 100:.2f}%")
    
    # Simple Confusion Matrix representation
    unique_classes = sorted(stress_mapping.values())
    print("\nConfusion Matrix:")
    print("True \\ Pred | " + " | ".join(f"{c:^9}" for c in unique_classes))
    for true_c in unique_classes:
        row_str = f"{true_c:^11} | "
        pred_counts = []
        for pred_c in unique_classes:
            count = np.sum((y_stress_test == true_c) & (y_stress_pred == pred_c))
            pred_counts.append(f"{count:^9}")
        print(row_str + " | ".join(pred_counts))

    # ------------------ Export Models to JSON ------------------
    print("\nExporting trained models and metadata to trained_model.json...")
    
    model_export = {
        "features": feature_names,
        "categorical_mappings": {k: {val: int(i) for val, i in v.items()} for k, v in encoding_maps.items()},
        "stress_level_mapping": {k: int(v) for k, v in stress_mapping.items()},
        "stress_level_inverse_mapping": {str(k): v for k, v in inverse_stress_mapping.items()},
        "linear_regression": lr.to_dict(),
        "decision_tree": dt.to_dict()
    }
    
    output_file = "trained_model.json"
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(model_export, out_f, indent=2)
        
    print(f"Training successfully completed. Trained models exported to {output_file}")

if __name__ == "__main__":
    main()
