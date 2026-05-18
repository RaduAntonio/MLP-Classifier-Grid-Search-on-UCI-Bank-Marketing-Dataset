"""
MLP Classification - Bank Marketing Dataset (bank-additional)
=============================================================
Variatie concomitenta a tuturor combinatiilor:
  - Numar straturi ascunse: 1 sau 2
  - Neuroni pe strat: egal cu stratul anterior sau jumatate
  - Learning rate: 0.1 sau 0.01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import itertools
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. INCARCARE SI PREPROCESARE DATE
# ──────────────────────────────────────────────
print("=" * 70)
print("  MLP Classification - Bank Marketing Dataset (bank-additional)")
print("=" * 70)

df = pd.read_csv('bank-additional.csv', sep=';')
print(f"\n[1] Dataset shape: {df.shape}")
print(f"    Distributie target:\n{df['y'].value_counts().to_string()}")
print(f"    Procent 'yes': {df['y'].value_counts(normalize=True)['yes']*100:.1f}%")

# Encoding
df_enc = df.copy()
label_encoders = {}
cat_cols = df_enc.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'y']

for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

# Target
df_enc['y'] = (df_enc['y'] == 'yes').astype(int)

X = df_enc.drop('y', axis=1).values
y = df_enc['y'].values
feature_names = df_enc.drop('y', axis=1).columns.tolist()

print(f"    Features: {X.shape[1]}")

# Train/test split stratificat
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Scalare
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

n_features = X.shape[1]

# ──────────────────────────────────────────────
# 2. DEFINIRE COMBINATII HIPERPARAMETRI
# ──────────────────────────────────────────────
# Regula pentru neuroni:
#   - Strat 1: egal cu n_features (20) sau jumatate (10)
#   - Strat 2 (daca exista): egal cu strat1 sau jumatate din strat1

learning_rates = [0.1, 0.01]

def get_hidden_layer_configs(n_features):
    """
    Returneaza toate configuratiile posibile de straturi ascunse.
    Regula: fiecare strat = strat anterior SAU jumatate din strat anterior.
    Primul strat porneste de la n_features.
    """
    configs = []
    # 1 strat ascuns
    layer1_options = [n_features, n_features // 2]
    for l1 in layer1_options:
        configs.append((l1,))
    # 2 straturi ascunse
    for l1 in layer1_options:
        layer2_options = [l1, l1 // 2]
        for l2 in layer2_options:
            configs.append((l1, l2))
    return configs

hidden_configs = get_hidden_layer_configs(n_features)

print(f"\n[2] Configuratii hiperparametri:")
print(f"    Learning rates: {learning_rates}")
print(f"    Configuratii straturi ascunse ({len(hidden_configs)}):")
for cfg in hidden_configs:
    label = f"{len(cfg)} strat{'uri' if len(cfg)>1 else ''} ascuns{'e' if len(cfg)>1 else ''}"
    print(f"      {cfg} -> {label}")

all_combos = list(itertools.product(hidden_configs, learning_rates))
print(f"\n    Total combinatii: {len(all_combos)}")

# ──────────────────────────────────────────────
# 3. ANTRENARE SI EVALUARE TOATE COMBINATIILE
# ──────────────────────────────────────────────
print(f"\n[3] Antrenare modele...")
print("-" * 70)
fmt = "{:<5} {:<20} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8}"
print(fmt.format("#", "Straturi ascunse", "LR", "Acc%", "Prec%", "Recall%", "F1%", "AUC%"))
print("-" * 70)

results = []
models = {}

for idx, (hidden_layers, lr) in enumerate(all_combos, 1):
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=lr,
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

    t0 = time.time()
    clf.fit(X_train_sc, y_train)
    t1 = time.time()

    y_pred  = clf.predict(X_test_sc)
    y_prob  = clf.predict_proba(X_test_sc)[:, 1]

    acc    = accuracy_score(y_test, y_pred) * 100
    prec   = precision_score(y_test, y_pred, zero_division=0) * 100
    rec    = recall_score(y_test, y_pred, zero_division=0) * 100
    f1     = f1_score(y_test, y_pred, zero_division=0) * 100
    auc    = roc_auc_score(y_test, y_prob) * 100
    n_iter = clf.n_iter_

    result = {
        'id': idx,
        'hidden_layers': hidden_layers,
        'n_hidden': len(hidden_layers),
        'lr': lr,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'n_iter': n_iter,
        'train_time': t1 - t0,
        'label': f"{hidden_layers} lr={lr}"
    }
    results.append(result)
    models[idx] = clf

    print(fmt.format(
        idx,
        str(hidden_layers),
        str(lr),
        f"{acc:.2f}",
        f"{prec:.2f}",
        f"{rec:.2f}",
        f"{f1:.2f}",
        f"{auc:.2f}"
    ))

print("-" * 70)

df_results = pd.DataFrame(results)

# ──────────────────────────────────────────────
# 4. IDENTIFICARE MODEL OPTIM
# ──────────────────────────────────────────────
best_idx_f1  = df_results['f1'].idxmax()
best_idx_auc = df_results['auc'].idxmax()
best_idx_acc = df_results['accuracy'].idxmax()

best_f1  = df_results.loc[best_idx_f1]
best_auc = df_results.loc[best_idx_auc]
best_acc = df_results.loc[best_idx_acc]

print(f"\n[4] Modele optime:")
print(f"    Cel mai bun F1  -> #{int(best_f1['id'])}: straturi={best_f1['hidden_layers']}, LR={best_f1['lr']}, F1={best_f1['f1']:.2f}%")
print(f"    Cel mai bun AUC -> #{int(best_auc['id'])}: straturi={best_auc['hidden_layers']}, LR={best_auc['lr']}, AUC={best_auc['auc']:.2f}%")
print(f"    Cel mai bun Acc -> #{int(best_acc['id'])}: straturi={best_acc['hidden_layers']}, LR={best_acc['lr']}, Acc={best_acc['accuracy']:.2f}%")

# Cel mai slab
worst_idx = df_results['f1'].idxmin()
worst = df_results.loc[worst_idx]
print(f"    Cel mai slab F1 -> #{int(worst['id'])}: straturi={worst['hidden_layers']}, LR={worst['lr']}, F1={worst['f1']:.2f}%")

# ──────────────────────────────────────────────
# 5. RAPORT DETALIAT - MODELUL OPTIM (F1)
# ──────────────────────────────────────────────
best_model_id = int(best_f1['id'])
best_clf = models[best_model_id]
y_pred_best = best_clf.predict(X_test_sc)
y_prob_best = best_clf.predict_proba(X_test_sc)[:, 1]

print(f"\n[5] Raport detaliat - Modelul #{best_model_id} (F1 maxim):")
print(f"    Straturi: {best_f1['hidden_layers']}, LR: {best_f1['lr']}")
print(classification_report(y_test, y_pred_best, target_names=['no (0)', 'yes (1)']))

# ──────────────────────────────────────────────
# 6. VIZUALIZARI
# ──────────────────────────────────────────────
print("\n[6] Generare grafice...")

palette_dark = '#0d1117'
palette_blue = '#58a6ff'
palette_green = '#3fb950'
palette_orange = '#f78166'
palette_purple = '#bc8cff'
palette_yellow = '#e3b341'
palette_gray = '#8b949e'

fig = plt.figure(figsize=(20, 22), facecolor=palette_dark)
fig.suptitle('MLP Classification — Bank Marketing (bank-additional)\nAnaliza completa a tuturor combinatiilor de hiperparametri',
             fontsize=16, color='white', fontweight='bold', y=0.98)

gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                      left=0.06, right=0.97, top=0.94, bottom=0.04)

def style_ax(ax, title):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=palette_gray, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.xaxis.label.set_color(palette_gray)
    ax.yaxis.label.set_color(palette_gray)
    return ax

# ── 6.1 Tabel comparativ metrici
ax1 = fig.add_subplot(gs[0, :2])
ax1 = style_ax(ax1, '1. Comparatie metrici — toate combinatiile')
x_pos = np.arange(len(df_results))
w = 0.18
metrics = [('accuracy', palette_blue, 'Accuracy'),
           ('precision', palette_green, 'Precision'),
           ('recall', palette_orange, 'Recall'),
           ('f1', palette_purple, 'F1-Score'),
           ('auc', palette_yellow, 'AUC-ROC')]
for i, (m, c, label) in enumerate(metrics):
    ax1.bar(x_pos + i*w, df_results[m], width=w, color=c, alpha=0.85, label=label)
ax1.set_xticks(x_pos + 2*w)
ax1.set_xticklabels([f"#{r['id']}" for _, r in df_results.iterrows()], fontsize=7)
ax1.set_ylabel('Scor (%)')
ax1.set_ylim(0, 115)
ax1.legend(fontsize=7, loc='upper right',
           facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
# Marcare best F1
ax1.axvline(x=best_model_id - 1 + 2*w, color=palette_purple, linestyle='--', alpha=0.5, linewidth=1)
ax1.text(best_model_id - 1 + 2*w + 0.05, 112, f'Best F1\n#{best_model_id}',
         color=palette_purple, fontsize=7, va='top')

# ── 6.2 Heatmap F1 score
ax2 = fig.add_subplot(gs[0, 2])
ax2 = style_ax(ax2, '2. F1 Score — Straturi vs LR')
pivot = df_results.pivot_table(values='f1', index='hidden_layers', columns='lr', aggfunc='mean')
pivot.index = [str(i) for i in pivot.index]
sns.heatmap(pivot, ax=ax2, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=0.5,
            linecolor='#30363d', cbar_kws={'label': 'F1 (%)'})
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Straturi ascunse')
ax2.tick_params(axis='x', colors=palette_gray)
ax2.tick_params(axis='y', colors=palette_gray)

# ── 6.3 Matrice confuzie - model optim
ax3 = fig.add_subplot(gs[1, 0])
ax3 = style_ax(ax3, f'3. Matrice confuzie — Model #{best_model_id} (Best F1)')
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, ax=ax3, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
            linewidths=1, linecolor='#30363d')
ax3.set_xlabel('Prezis')
ax3.set_ylabel('Real')

# ── 6.4 Curba ROC - toate modelele
ax4 = fig.add_subplot(gs[1, 1])
ax4 = style_ax(ax4, '4. Curbe ROC — Toate modelele')
colors_roc = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
for i, r in enumerate(results):
    m = models[r['id']]
    yp = m.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, yp)
    lw = 2.5 if r['id'] == best_model_id else 0.8
    alpha = 1.0 if r['id'] == best_model_id else 0.4
    ax4.plot(fpr, tpr, color=colors_roc[i], linewidth=lw, alpha=alpha,
             label=f"#{r['id']} AUC={r['auc']:.1f}%" if r['id'] == best_model_id else None)
ax4.plot([0,1],[0,1],'--', color=palette_gray, linewidth=0.7, alpha=0.5)
ax4.set_xlabel('FPR')
ax4.set_ylabel('TPR')
ax4.set_xlim(0,1); ax4.set_ylim(0,1.02)
ax4.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# ── 6.5 Curba de invatare - model optim
ax5 = fig.add_subplot(gs[1, 2])
ax5 = style_ax(ax5, f'5. Curba de invatare — Model #{best_model_id}')
if hasattr(best_clf, 'loss_curve_'):
    ax5.plot(best_clf.loss_curve_, color=palette_blue, linewidth=1.5, label='Train loss')
    if hasattr(best_clf, 'validation_scores_'):
        ax5_r = ax5.twinx()
        ax5_r.plot(best_clf.validation_scores_, color=palette_orange, linewidth=1.5, label='Val score')
        ax5_r.tick_params(colors=palette_gray, labelsize=8)
        ax5_r.set_ylabel('Validation score', color=palette_orange, fontsize=8)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white', loc='upper left')

# ── 6.6 AUC per configuratie de straturi
ax6 = fig.add_subplot(gs[2, 0])
ax6 = style_ax(ax6, '6. AUC-ROC per arhitectura')
df_results['arch_label'] = df_results['hidden_layers'].apply(str)
arch_auc = df_results.groupby('arch_label')['auc'].mean().sort_values(ascending=True)
bars = ax6.barh(arch_auc.index, arch_auc.values, color=palette_purple, alpha=0.8)
ax6.set_xlabel('AUC-ROC medie (%)')
for bar, val in zip(bars, arch_auc.values):
    ax6.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', color='white', fontsize=7)

# ── 6.7 F1 vs LR
ax7 = fig.add_subplot(gs[2, 1])
ax7 = style_ax(ax7, '7. F1-Score vs Learning Rate')
colors_lr = {0.1: palette_orange, 0.01: palette_blue}
for lr_val in learning_rates:
    sub = df_results[df_results['lr'] == lr_val]
    ax7.scatter(sub['accuracy'], sub['f1'],
                color=colors_lr[lr_val], s=80, alpha=0.85,
                label=f'LR={lr_val}', zorder=3)
ax7.set_xlabel('Accuracy (%)')
ax7.set_ylabel('F1-Score (%)')
ax7.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# ── 6.8 Numar iteratii pana la convergenta
ax8 = fig.add_subplot(gs[2, 2])
ax8 = style_ax(ax8, '8. Iteratii pana la convergenta')
bar_colors = [palette_orange if r['lr'] == 0.1 else palette_blue for _, r in df_results.iterrows()]
ax8.bar([f"#{r['id']}" for _, r in df_results.iterrows()],
        df_results['n_iter'], color=bar_colors, alpha=0.85)
ax8.set_xlabel('Model #')
ax8.set_ylabel('Nr. iteratii')
legend_elems = [mpatches.Patch(facecolor=palette_orange, label='LR=0.1'),
                mpatches.Patch(facecolor=palette_blue, label='LR=0.01')]
ax8.legend(handles=legend_elems, fontsize=8, facecolor='#161b22',
           edgecolor='#30363d', labelcolor='white')

# ── 6.9 Tabel sumar final (row 3, full width)
ax9 = fig.add_subplot(gs[3, :])
ax9.set_facecolor('#161b22')
ax9.axis('off')
ax9.set_title('9. Tabel sumar — toate configuratiile', color='white',
               fontsize=10, fontweight='bold', pad=8)

col_labels = ['#', 'Straturi ascunse', 'Nr straturi', 'LR', 'Accuracy %', 'Precision %', 'Recall %', 'F1 %', 'AUC %', 'Iteratii']
table_data = []
for _, r in df_results.iterrows():
    table_data.append([
        int(r['id']),
        str(r['hidden_layers']),
        int(r['n_hidden']),
        r['lr'],
        f"{r['accuracy']:.2f}",
        f"{r['precision']:.2f}",
        f"{r['recall']:.2f}",
        f"{r['f1']:.2f}",
        f"{r['auc']:.2f}",
        int(r['n_iter'])
    ])

tbl = ax9.table(cellText=table_data, colLabels=col_labels,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#21262d')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        r_data = df_results.iloc[row - 1]
        is_best = int(r_data['id']) == best_model_id
        cell.set_facecolor('#1c6b3a' if is_best else '#0d1117')
        cell.set_text_props(color='white' if is_best else palette_gray)
    cell.set_edgecolor('#30363d')

plt.savefig('/mnt/user-data/outputs/mlp_bank_results.png',
            dpi=150, bbox_inches='tight', facecolor=palette_dark)
print("    Salvat: mlp_bank_results.png")

# ──────────────────────────────────────────────
# 7. CROSS-VALIDATION PE MODELUL OPTIM
# ──────────────────────────────────────────────
print(f"\n[7] Cross-validation 5-fold — Model #{best_model_id}:")
cv_clf = MLPClassifier(
    hidden_layer_sizes=best_clf.hidden_layer_sizes,
    learning_rate_init=best_clf.learning_rate_init,
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20
)
pipe = Pipeline([('scaler', StandardScaler()), ('mlp', cv_clf)])
cv_scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring='roc_auc')
print(f"    AUC per fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"    AUC medie: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\n{'='*70}")
print(f"  CONCLUZIE FINALA")
print(f"{'='*70}")
print(f"  Best model (F1): #{best_model_id} | Straturi: {best_f1['hidden_layers']} | LR: {best_f1['lr']}")
print(f"  Accuracy:  {best_f1['accuracy']:.2f}%")
print(f"  Precision: {best_f1['precision']:.2f}%")
print(f"  Recall:    {best_f1['recall']:.2f}%")
print(f"  F1-Score:  {best_f1['f1']:.2f}%")
print(f"  AUC-ROC:   {best_f1['auc']:.2f}%")
print(f"  CV AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*70}")
