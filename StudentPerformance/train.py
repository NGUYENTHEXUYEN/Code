import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load data from the CSV file
df = pd.read_csv('student-por.csv')

# Step 2: Preprocess the data (e.g., encoding categorical features)
label_encoder = LabelEncoder()
categorical_columns = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Step 3: Split data into features and target (assume G3 is the target)
X = df.drop(columns=["G3"])
y = df["G3"]

# Step 4: Train a model (RandomForestClassifier for this example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the model using joblib
joblib.dump(model, 'student_performance_model.pkl')

print("Model saved as student_performance_model.pkl")
