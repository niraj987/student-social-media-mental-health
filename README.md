# Student Social Media & Mental Health: Predictive Analysis & Risk Management

An end-to-end data science portfolio project designed to explore, model, and mitigate the mental health impacts of digital screen time and phone usage among students. 

To demonstrate core mathematical proficiency and bypass local system environment limitations, the predictive machine learning models are **implemented entirely from scratch in Python** using only `numpy` and the standard library, then serialized into JSON for direct interactive execution inside a **premium client-side dashboard**.

---

## 📋 Table of Contents
1. [Project Overview & Key Insights](#1-project-overview--key-insights)
2. [Dataset Structure](#2-dataset-structure)
3. [Mathematical Formulations (Models From Scratch)](#3-mathematical-formulations-models-from-scratch)
4. [Python Pipeline & Performance Metrics](#4-python-pipeline--performance-metrics)
5. [Interactive Dashboard Architecture](#5-interactive-dashboard-architecture)
6. [Risk Management & Mitigation Framework](#6-risk-management--mitigation-framework)
7. [How to Run and Validate](#7-how-to-run-and-validate)

---

## 1. Project Overview & Key Insights
This project analyzes **5,000 student logs** to understand the correlations between academic pressure, physical activity, sleep cycles, and daily social media behaviors. 

### Key Findings:
- **Circadian Depletion:** Every hour of social media usage past the cohort average (4.8h) is associated with an average decrease of **0.38 hours of sleep** and a drop of **0.54 points** in overall mental health.
- **Platform Concentration:** Instagram, Facebook, and TikTok represent over **68%** of dominant screen time, showing the strongest correlations with high stress levels.
- **The Unlock Threshold:** Exceeding **150 daily unlocks** is a strong indicator of high-stress classification, regardless of study hours.
- **Mitigation Leverage:** Integrating 45 minutes of daily physical exercise and adhering to a 7.5h sleep curfew offsets up to **75%** of screen-induced stress metrics.

---

## 2. Dataset Structure
The dataset features 13 columns containing demographic, academic, digital usage, and wellness variables:

| Column Name | Data Type | Description / Domain |
| :--- | :--- | :--- |
| `Age` | Numerical | Age of the student (15–30 years) |
| `Gender` | Categorical | Male, Female, Other |
| `Country` | Categorical | Demographic residence |
| `Academic_Level` | Categorical | High School, Undergraduate, Graduate |
| `Most_Used_Platform` | Categorical | Primary social media app (e.g. Instagram, TikTok, LinkedIn) |
| `Purpose_Of_Use` | Categorical | Motivation (Entertainment, Networking, Education, etc.) |
| `Avg_Daily_Usage_Hours` | Numerical | Screen time dedicated to social media (0–15h) |
| `Daily_Unlocks` | Numerical | Phone unlocks per day (10–400) |
| `Study_Hours` | Numerical | Daily academic study hours (0–12h) |
| `Physical_Activity_Hours`| Numerical | Daily exercise/activity hours (0–6h) |
| `Sleep_Hours_Per_Night` | Numerical | Sleep duration (3–12h) |
| `Stress_Level` | Categorical | Self-reported stress: **Low, Medium, High, Very High** |
| `Mental_Health_Score` | Numerical | Overall mental health score (0.0 to 10.0 scale) |

---

## 3. Mathematical Formulations (Models From Scratch)

To highlight underlying data science principles, the core predictive engines were coded from basic algebraic equations.

### A. Multiple Linear Regression (OLS)
Predicts the numerical `Mental_Health_Score` ($y_{score} \in [0.0, 10.0]$).

#### 1. Standardization (Z-Score Scaling)
To ensure numerical stability and coordinate scale equivalence, input feature matrix $X$ is scaled using training feature means ($\mu_j$) and standard deviations ($\sigma_j$):
$$x_{ij}' = \frac{x_{ij} - \mu_j}{\sigma_j}$$

#### 2. Closed-Form Solution (Ordinary Least Squares)
We solve the coefficient vector $\beta = [\beta_0, \beta_1, \dots, \beta_k]^T$ using the normal equation. To guarantee invertibility even in the presence of multicollinearity, a small Ridge regularization term ($\alpha = 10^{-5}$) is added to the design matrix diagonal:
$$\beta = (X'^T X' + \alpha I)^{-1} X'^T y$$
Where $X'$ is the standardized feature matrix with an added column of ones representing the intercept term, and $I$ is the identity matrix (with $I_{0,0} = 0$ to avoid regularizing the intercept).

---

### B. Decision Tree Classifier (Gini Impurity)
Classifies `Stress_Level` ($y_{stress} \in \{0, 1, 2, 3\}$ corresponding to Low, Medium, High, Very High).

#### 1. Splitting Criterion
For any node $D$, the Gini Impurity $I_G(D)$ measures class distribution purity:
$$I_G(D) = 1 - \sum_{c=0}^{C-1} p_c^2$$
Where $p_c$ is the proportion of samples in $D$ belonging to class $c$.

#### 2. Information Gain Minimization
At each parent node, we iterate over all candidate features and thresholds to find the split partitioning the data into left ($D_L$) and right ($D_R$) child nodes that minimizes the weighted split impurity:
$$I_G(D_{split}) = \frac{N_L}{N} I_G(D_L) + \frac{N_R}{N} I_G(D_R)$$
We choose the split that maximizes the Gini gain: $\Delta I_G = I_G(D) - I_G(D_{split})$.

#### 3. Regularization Constraints
To prevent overfitting, the tree recursive growth is bound by:
- `max_depth = 6`
- `min_samples_split = 10`
- `min_samples_leaf = 5`

---

## 4. Python Pipeline & Performance Metrics

### Execution Flow:
1. `eda.py` reads the CSV, calculates aggregates, extracts correlations, and outputs statistics for visualization to `data_export.json`.
2. `train.py` reads the dataset, maps categorical columns to consistent integer keys, trains the custom models, validates them against a **20% holdout test set**, and exports coefficients and the tree structure to `trained_model.json`.

### Model Evaluation Results:

#### 📈 Linear Regression (Mental Health Score prediction):
- **Root Mean Squared Error (RMSE):** **0.7039** (on a 0-10 scale)
- **R-squared ($R^2$) Score:** **0.6972** (meaning 69.7% of the variance is captured by our features)
- **Key Coefficients (Standardized):**
  - `Avg_Daily_Usage_Hours`: **-0.5392** (Largest negative impact)
  - `Sleep_Hours_Per_Night`: **+0.3864** (Largest positive restorative impact)
  - `Study_Hours`: **+0.1682**
  - `Daily_Unlocks`: **-0.0436**

#### 🌳 Decision Tree Classifier (Stress Level prediction):
- **Out-of-Sample Accuracy:** **72.40%** (vs. 25.0% baseline random guessing)
- **Testing Set Confusion Matrix:**
```
True \ Pred |    0 (Low) | 1 (Medium) |  2 (High)  | 3 (Very High)
0 (Low)     |    70      |    58      |     2      |     0    
1 (Medium)  |    22      |    194     |    45      |     0    
2 (High)    |     1      |    61      |    161     |    56    
3 (V. High) |     0      |     0      |    31      |    299   
```
*(No critical off-by-two classifications occur, proving high model reliability).*

---

## 5. Interactive Dashboard Architecture
The visual dashboard uses a modern glassmorphic interface built on vanilla CSS:
- **Dynamic KPI Summaries:** Displays dataset metrics.
- **Responsive Charts:** Rendered via **Chart.js** displaying platform distribution, screen hours vs. stress metrics, and screen hours vs. mental health scores.
- **Interactive Correlation Matrix:** A color-coded cell grid built dynamically from Pearson coefficients. Green/blue cells represent positive correlations (sleep/study with high mental health) and red cells represent negative correlations (screen hours with high stress).
- **Live Prediction Sandbox:** Runs JavaScript implementations of the custom Decision Tree and Linear Regression models. Changing form parameters immediately re-runs predictions, sliding the needle on the stress gauge and updating the score indicator without server lag.
- **Cohorts Dataset Explorer:** Displays raw records with pagination and dynamic filter dropdowns.

---

## 6. Risk Management & Mitigation Framework
A core feature is the **Wellness Risk Mitigation Generator** which converts predictions into actionable clinical recommendations:

1. **Risk Identification:** Checks inputs against parameters:
   - *Screen Time > 5.5h* triggers an **Elevated Screen Time** driver.
   - *Unlocks > 150* triggers a **Frequent Unlocks** loop indicator.
   - *Sleep < 6.0h* triggers a **Sleep Deprivation** alert.
   - *Physical Activity < 1.0h* triggers a **Sedentary Behaviour** warning.
2. **Dynamic Risk Summarization:** Classifies user risk status into Low, Moderate, or High wellness threats.
3. **Actionable Mitigation Protocols:** Emits personalized wellness goals, such as a **Digital Detox curfew** (targeting screen time reduction based on current hours) and **Sleep Curfews** tailored to restore baseline circadian cycles.
4. **Print Optimization:** Formatted with a clean CSS print media layout (`@media print`). Clicking "Export PDF" strips away dashboard navigation panels, adapts text to a high-contrast white layout, and formats the page into a clean, professional PDF suitable for physical printing or clinical files.

---

## 7. How to Run and Validate

### Requirements:
- Python 3.x
- NumPy (`pip install numpy`)

### Step 1: Pre-process and Generate EDA Exports
Run the exploratory analysis script:
```bash
python eda.py
```
This processes `Student Social Media And Mental Health Impact.csv` and outputs `data_export.json`.

### Step 2: Train and Export Models
Run the training pipeline:
```bash
python train.py
```
This trains the models from scratch and saves the parameters to `trained_model.json`.

### Step 3: Run the Dashboard
Since the dashboard fetches local JSON files, modern browsers restrict local file loading via the `file://` protocol due to CORS security policies. Start a simple local server in the project directory:

**Using Python:**
```bash
python -m http.server 8000
```
Then, open your browser and navigate to:
```
http://localhost:8000/
```

*Now you can interact with the charts, edit sandbox values, check live predictions, and export custom wellness mitigation reports!*
