import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(
    os.path.abspath(__file__)))

# Load data
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Shape: {df.shape}")
print(f"\nFraud rate: "
      f"{df['Class'].mean():.4%}")
print(f"Fraud cases   : "
      f"{df['Class'].sum():,}")
print(f"Normal cases  : "
      f"{(df['Class']==0).sum():,}")

# Features
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount and Time
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(
    X[['Amount']])
X['Time']   = scaler.fit_transform(
    X[['Time']])

# Split BEFORE SMOTE
X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

print(f"\nTraining set: {X_train.shape}")
print(f"Test set     : {X_test.shape}")
print(f"Train fraud  : {y_train.sum()}")

# Apply SMOTE only on training data
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(
    X_train, y_train)
print(f"After SMOTE  : {X_train_sm.shape}")
print(f"Fraud after  : {y_train_sm.sum():,}")

# Train models
models = {
    'Logistic Regression':
        LogisticRegression(
            max_iter=1000,
            random_state=42),
    'Random Forest':
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1)
}

results = {}
print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    preds = model.predict(X_test)
    proba = model.predict_proba(
        X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(
        y_test, proba)

    results[name] = {
        'model': model,
        'preds': preds,
        'proba': proba,
        'auc':   round(auc, 4),
        'ap':    round(ap, 4)
    }
    print(f"\n{name}:")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(classification_report(
        y_test, preds,
        target_names=['Normal', 'Fraud']))

# Best model
best_name = max(
    results, key=lambda x: results[x]['auc'])
best      = results[best_name]
print(f"\nBest model: {best_name}")

# Plot 1 — Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Normal', 'Fraud'],
            [y.value_counts()[0],
             y.value_counts()[1]],
            color=['#2ecc71', '#e74c3c'],
            edgecolor='black')
axes[0].set_title(
    'Original Class Distribution',
    fontsize=12)
axes[0].set_ylabel('Count')
for i, v in enumerate(
    [y.value_counts()[0],
     y.value_counts()[1]]
):
    axes[0].text(i, v + 100,
                 f'{v:,}',
                 ha='center',
                 fontweight='bold')

axes[1].bar(['Normal', 'Fraud'],
            [y_train_sm.value_counts()[0],
             y_train_sm.value_counts()[1]],
            color=['#3498db', '#f39c12'],
            edgecolor='black')
axes[1].set_title(
    'After SMOTE Balancing',
    fontsize=12)
axes[1].set_ylabel('Count')
plt.suptitle(
    'Class Imbalance — Before vs After SMOTE',
    fontsize=14)
plt.tight_layout()
plt.savefig('class_distribution.png')
print("\nClass distribution chart saved!")

# Plot 2 — Confusion Matrix
cm = confusion_matrix(y_test, best['preds'])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title(
    f'Confusion Matrix — {best_name}',
    fontsize=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved!")

# Plot 3 — Precision-Recall Curve
plt.figure(figsize=(8, 5))
colors = ['#3498db', '#e74c3c']
for (name, res), color in zip(
    results.items(), colors
):
    precision, recall, _ = \
        precision_recall_curve(
            y_test, res['proba'])
    ap = res['ap']
    plt.plot(recall, precision,
             color=color, linewidth=2,
             label=f'{name} (AP={ap:.3f})')
plt.title(
    'Precision-Recall Curve — Fraud Detection',
    fontsize=13)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall.png')
print("Precision-recall curve saved!")

# Plot 4 — Feature importance
if hasattr(best['model'],
           'feature_importances_'):
    feat_imp = pd.Series(
        best['model'].feature_importances_,
        index=X.columns
    ).sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 7))
    colors_fi = plt.cm.RdYlGn(
        np.linspace(0.3, 1.0, len(feat_imp)))
    plt.barh(feat_imp.index,
             feat_imp.values,
             color=colors_fi,
             edgecolor='black')
    plt.title('Top 15 Features — '
              'Fraud Detection',
              fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance saved!")

# Save model
with open('model.pkl',  'wb') as f:
    pickle.dump(best['model'], f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nAll files saved!")
print(f"Best model: {best_name}")
print(f"AUC-ROC   : {best['auc']}")
print("Run app.py next!")