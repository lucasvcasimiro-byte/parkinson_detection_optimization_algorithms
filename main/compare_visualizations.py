import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cmp_df = pd.read_csv('results/csv/ga_vs_gwo.csv')

### Boxplot — F1-Score distribution across 30 runs
plt.figure(figsize=(6, 5))
sns.boxplot(data=cmp_df, x='algorithm', y='f1_score', palette={'GA': '#4c72b0', 'GWO': '#dd8452'})
plt.title('GA vs GWO — F1-Score Distribution (30 runs)')
plt.xlabel('Algorithm')
plt.ylabel('F1-Score (Parkinson\'s Class)')
plt.tight_layout()
plt.savefig('results/visualizations/ga_vs_gwo_boxplot.png')
plt.close()

### Bar chart — mean of all metrics side by side
metrics = ['f1_score', 'accuracy', 'precision', 'recall']
means = cmp_df.groupby('algorithm')[metrics].mean()

x = range(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([i - width/2 for i in x], means.loc['GA'],  width, label='GA',  color='#4c72b0')
ax.bar([i + width/2 for i in x], means.loc['GWO'], width, label='GWO', color='#dd8452')
ax.set_xticks(list(x))
ax.set_xticklabels(['F1-Score', 'Accuracy', 'Precision', 'Recall'])
ax.set_ylabel('Mean Score (30 runs)')
ax.set_title('GA vs GWO — Average Metrics')
ax.set_ylim(means.values.min() * 0.95, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig('results/visualizations/ga_vs_gwo_metrics.png')
plt.close()

print("Comparison visualizations saved to results/visualizations/")
