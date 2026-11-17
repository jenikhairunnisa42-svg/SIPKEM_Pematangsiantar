# ============================================================
# TAHAP 2: PREPROCESSING & TRAINING RANDOM FOREST
# Dataset: Kemiskinan Ekstrem Kota Pematangsiantar (Sintetis)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Set style visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("TAHAP 2: TRAINING MODEL RANDOM FOREST")
print("=" * 60)

# 1. LOAD DATASET
df = pd.read_csv('dataset_kemiskinan_siantar_sintetis.csv')
print(f"\n✅ Dataset berhasil dimuat: {df.shape[0]} baris × {df.shape[1]} kolom")
print(f"\n📊 Distribusi Target (Status Kemiskinan):")
print(df['Status_Kemiskinan_Ekstrem'].value_counts(normalize=True) * 100)

# 2. PREPROCESSING - Encode variabel kategorikal
le_dict = {}
kategorikal_cols = ['Pendidikan_Tertinggi', 'Status_Ketenagakerjaan', 
                    'Status_Kepemilikan_Rumah', 'Jenis_Lantai', 
                    'Jenis_Dinding', 'Sumber_Air_Minum', 
                    'Fasilitas_Jamban', 'Tangki_Septik_IPAL']

for col in kategorikal_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    le_dict[col] = le

# 3. PERSIAPAN FITUR (X) DAN TARGET (y)
feature_cols = [col + '_encoded' for col in kategorikal_cols] + \
               ['Luas_Lantai_perkapita_m2', 'Pengeluaran_perkapita_Rp',
                'Penerima_BPNT', 'Penerima_PKH', 
                'Penerima_Jaminan_Sosial', 'Penggunaan_Jaminan_Kesehatan']

X = df[feature_cols]
y = df['Status_Kemiskinan_Ekstrem']

print(f"\n🔢 Jumlah fitur: {X.shape[1]}")
print(f"📈 Jumlah sampel: {X.shape[0]}")

# 4. SPLIT DATA (80% Train, 20% Test) dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️ Data Training: {X_train.shape[0]} sampel")
print(f"✂️ Data Testing: {X_test.shape[0]} sampel")

# 5. TRAINING RANDOM FOREST dengan class_weight='balanced'
print("\n" + "=" * 60)
print("MULAI TRAINING MODEL RANDOM FOREST...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Penting untuk data tidak seimbang
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✅ Model berhasil dilatih!")

# 6. PREDIKSI & EVALUASI
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"🎯 Akurasi (Accuracy)  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Presisi (Precision) : {precision:.4f} ({precision*100:.2f}%)")
print(f"🎯 Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
print(f"🎯 F1-Score            : {f1:.4f} ({f1*100:.2f}%)")
print(f"🎯 AUC-ROC             : {auc_roc:.4f} ({auc_roc*100:.2f}%)")

# 7. CROSS-VALIDATION (5-Fold Stratified)
print("\n" + "=" * 60)
print("VALIDASI SILANG (5-Fold Stratified Cross-Validation)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='f1')
print(f"📊 F1-Score per fold: {cv_scores}")
print(f"📊 Rata-rata F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 8. FEATURE IMPORTANCE (Variabel Paling Berpengaruh)
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Variabel Paling Mempengaruhi Kemiskinan)")
print("=" * 60)

feature_names_clean = [col.replace('_encoded', '') for col in feature_cols]
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Fitur': feature_names_clean,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_imp_df.to_string(index=False))

# 9. VISUALISASI 1: Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_imp_df, x='Importance', y='Fitur', palette='viridis')
plt.title('Feature Importance - Random Forest\nKlasifikasi Kemiskinan Ekstrem Pematangsiantar', 
          fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik Feature Importance disimpan sebagai 'feature_importance.png'")

# 10. VISUALISASI 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tidak Miskin (0)', 'Miskin (1)'],
            yticklabels=['Tidak Miskin (0)', 'Miskin (1)'])
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Confusion Matrix disimpan sebagai 'confusion_matrix.png'")

# 11. VISUALISASI 3: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_roc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 ROC Curve disimpan sebagai 'roc_curve.png'")

# 12. CLASSIFICATION REPORT LENGKAP
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT LENGKAP")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Tidak Miskin (0)', 'Miskin (1)']))

print("\n" + "=" * 60)
print("✅ TAHAP 2 SELESAI!")
print("=" * 60)