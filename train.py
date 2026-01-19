import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUTPUT_DIR = r'D:\\sculpture_analysis\\analysis_results'
LABELING_CSV = os.path.join(OUTPUT_DIR, 'manual_labeling.csv')

df = pd.read_csv(LABELING_CSV)
df.set_index('filename', inplace=True)

# Train damage classifier
labeled_damage = df[df['true_damage'].notna() & (df['true_damage'] != '')]
if not labeled_damage.empty:
    damage_features = ['edge_density', 'contour_count', 'entropy']
    X_damage = labeled_damage[damage_features]
    y_damage = labeled_damage['true_damage']
    damage_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    damage_clf.fit(X_damage, y_damage)
    df['predicted_damage'] = damage_clf.predict(df[damage_features])
else:
    df['predicted_damage'] = 'Unknown'

# Train material classifier
labeled_material = df[df['true_material'].notna() & (df['true_material'] != '')]
if not labeled_material.empty:
    material_features = ['contrast', 'homogeneity', 'energy', 'brightness', 'mean_R', 'mean_G', 'mean_B']
    X_material = labeled_material[material_features]
    y_material = labeled_material['true_material']
    material_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    material_clf.fit(X_material, y_material)
    df['predicted_material'] = material_clf.predict(df[material_features])
else:
    df['predicted_material'] = 'Unknown'

# Save final results
df.to_csv(os.path.join(OUTPUT_DIR, 'full_analysis_supervised.csv'))
df[['true_damage', 'predicted_damage']].to_csv(os.path.join(OUTPUT_DIR, 'damage_supervised.csv'))
df[['true_material', 'predicted_material']].to_csv(os.path.join(OUTPUT_DIR, 'material_texture_supervised.csv'))

print("Classification complete!")
print(df['predicted_damage'].value_counts())
print(df['predicted_material'].value_counts())

