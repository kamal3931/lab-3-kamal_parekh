import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import os

# Load penguins dataset
df = sns.load_dataset('penguins')

# Drop rows with missing values
df = df.dropna()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['sex', 'island'], drop_first=True)

# Label encode the target
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])  # E.g., Adelie=0, Chinstrap=1, Gentoo=2

# Features and target
X = df.drop(columns=['species'])
y = df['species']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Instantiate XGBoost classifier
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=3,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluation
print("Training F1 Score:", f1_score(y_train, y_pred_train, average='weighted'))
print("Test F1 Score:", f1_score(y_test, y_pred_test, average='weighted'))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Save the model as JSON
os.makedirs('app/data', exist_ok=True)
model.save_model('app/data/model.json')
