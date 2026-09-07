import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['axes.edgecolor'] = '#000000'
mpl.rcParams['axes.linewidth'] = 0.6
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = '#d9d9d9'
mpl.rcParams['grid.linewidth'] = 0.4
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['text.color'] = '#000000'
mpl.rcParams['axes.labelcolor'] = '#000000'
mpl.rcParams['xtick.color'] = '#000000'
mpl.rcParams['ytick.color'] = '#000000'

TRAIN_COLOR = '#000000'
VAL_COLOR = '#555555'

df = pd.read_csv('training_log.csv')

# --- Figure 1: three-panel training curves ---
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))

ax = axes[0]
ax.plot(df['epoch'], df['loss'], color=TRAIN_COLOR, lw=1.3, linestyle='-', label='train')
ax.plot(df['epoch'], df['val_loss'], color=VAL_COLOR, lw=1.3, linestyle='--', label='val')
ax.set_title('Loss', fontsize=10, fontweight='bold')
ax.set_xlabel('epoch', fontsize=9)
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
ax.plot(df['epoch'], df['mean_iou'], color=TRAIN_COLOR, lw=1.3, linestyle='-', label='train')
ax.plot(df['epoch'], df['val_mean_iou'], color=VAL_COLOR, lw=1.3, linestyle='--', label='val')
best_epoch = df['val_mean_iou'].idxmax()
ax.scatter([df['epoch'][best_epoch]], [df['val_mean_iou'][best_epoch]],
           color='#000000', zorder=5, s=22, edgecolor='white', linewidth=0.6)
ax.annotate(f"best (ep. {int(df['epoch'][best_epoch])+1})",
            (df['epoch'][best_epoch], df['val_mean_iou'][best_epoch]),
            textcoords="offset points", xytext=(6, -18), fontsize=7.5, color='#000000')
ax.set_title('Mean IoU', fontsize=10, fontweight='bold')
ax.set_xlabel('epoch', fontsize=9)
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
ax.plot(df['epoch'], df['accuracy'], color=TRAIN_COLOR, lw=1.3, linestyle='-', label='train')
ax.plot(df['epoch'], df['val_accuracy'], color=VAL_COLOR, lw=1.3, linestyle='--', label='val')
ax.set_title('Pixel Accuracy', fontsize=10, fontweight='bold')
ax.set_xlabel('epoch', fontsize=9)
ax.legend(frameon=False, fontsize=8)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('chart_training_curves.png', dpi=220, transparent=True)
plt.close()

# --- Figure 2: learning rate schedule (small) ---
fig, ax = plt.subplots(figsize=(11, 1.2))
ax.step(df['epoch'], df['learning_rate'], color='#000000', lw=1.2, where='post')
ax.set_yscale('log')
ax.set_title('Learning Rate Schedule', fontsize=9, fontweight='bold', loc='left')
ax.set_xlabel('epoch', fontsize=8)
ax.tick_params(labelsize=7.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('chart_lr.png', dpi=220, transparent=True)
plt.close()

# --- Figure 3: per-class IoU / Dice bar chart ---
with open('metrics.json') as f:
    metrics = json.load(f)

classes = list(metrics['per_class_metrics'].keys())
ious = [metrics['per_class_metrics'][c]['iou'] for c in classes]
dices = [metrics['per_class_metrics'][c]['dice'] for c in classes]

order = sorted(range(len(classes)), key=lambda i: ious[i], reverse=True)
classes = [classes[i] for i in order]
ious = [ious[i] for i in order]
dices = [dices[i] for i in order]

x = np.arange(len(classes))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 3.6))
bars1 = ax.bar(x - width/2, ious, width, label='IoU', color='#000000')
bars2 = ax.bar(x + width/2, dices, width, label='Dice', color='#a6a6a6')

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=8.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel('score', fontsize=9)
ax.set_title('Per-Class IoU and Dice (test set)', fontsize=11, fontweight='bold')
ax.legend(frameon=False, fontsize=8.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)

for b in bars1:
    h = b.get_height()
    ax.annotate(f'{h:.2f}', (b.get_x() + b.get_width()/2, h), textcoords="offset points",
                xytext=(0, 2), ha='center', fontsize=7, color='#000000')

plt.tight_layout()
plt.savefig('chart_per_class.png', dpi=220, transparent=True)
plt.close()

print("charts done")
print("best epoch:", int(df['epoch'][best_epoch]), "val_mean_iou:", df['val_mean_iou'][best_epoch])
