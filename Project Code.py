import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except:
    print("\n⚠ LIME is not fully available (scikit-image issue). Using fallback...\n")
    LIME_AVAILABLE = False


file_path = "healthcare_event_anomaly_dataset.csv"
df = pd.read_csv(file_path)

# Handle missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("Dataset Sample:\n", df.head())
print("\nClass Distribution:\n", df["Label"].value_counts())

plt.figure(figsize=(6, 6))
df["Label"].value_counts().plot.pie(
    autopct="%1.1f%%", labels=["Normal", "Event", "Anomaly"]
)
plt.title("Class Distribution")
plt.show()

# Split data
X = df.drop(columns=["Label"])
y = df["Label"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning
param_grid = {
    'max_depth': range(3, 7),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 5)
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", grid_search.best_params_)

# Testing accuracy
y_pred = best_model.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Decision Tree Plot
plt.figure(figsize=(12, 6))
plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=[str(c) for c in y.unique()],
    filled=True,
    fontsize=8
)
plt.title("Decision Tree")
plt.show()

# ---- SHAP ----
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Convert to DataFrame for SHAP
X_test_df = pd.DataFrame(X_test, columns=X.columns)

shap.summary_plot(shap_values[0], X_test_df, feature_names=X.columns)

# ---- LIME ----
if LIME_AVAILABLE:
    explainer_lime = LimeTabularExplainer(
        X_train,
        feature_names=X.columns,
        class_names=["Normal", "Event", "Anomaly"],
        mode="classification"
    )

    instance = X_test[0]
    lime_exp = explainer_lime.explain_instance(instance, best_model.predict_proba)

    plt.figure(figsize=(8, 5))
    lime_exp.as_pyplot_figure()
    plt.title("LIME Explanation for Instance 0")
    plt.show()
else:
    print("\n⚠ LIME is skipped because scikit-image cannot be installed in Python 3.14\n")
