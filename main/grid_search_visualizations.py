import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GA grid search visualizations
results_df = pd.read_csv('results/csv/ga_grid_search.csv')

# Clean label for configurations
results_df['config'] = (
        'Arch ' + results_df['architecture'].astype(str) + ' + ' +
        results_df['init_method'] + ' + ' +
        results_df['selection'].str.replace('_selection', '', regex=False) + ' + ' +
        results_df['crossover'].str.replace('_crossover', '', regex=False) + ' + ' +
        results_df['mutation'].str.replace('_mutation', '', regex=False)
    )

### Top 10 configurations by F1-Score
top_10 = results_df.sort_values('f1_score', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_10, x='f1_score', y='config', palette='viridis', hue='config', legend=False)
plt.title('Top 10 GA Configurations by F1-Score')
plt.xlabel('F1-Score (Parkinson\'s Class)')
plt.ylabel('Configuration (Arch + Init + Selection + Crossover + Mutation)')

# Zoom in on differences
min_f1 = top_10['f1_score'].min()
max_f1 = top_10['f1_score'].max()
plt.xlim(max(0, min_f1 - 0.02), min(1.0, max_f1 + 0.01))
plt.tight_layout()
plt.savefig('results/visualizations/grid_search_top10.png')
plt.close()

### Average performance per operator type
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(data=results_df, x='selection', y='f1_score', ax=axes[0], palette='Blues', hue='selection', legend=False)
axes[0].set_title('Avg F1 by Selection Method')
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(data=results_df, x='crossover', y='f1_score', ax=axes[1], palette='Greens', hue='crossover', legend=False)
axes[1].set_title('Avg F1 by Crossover Method')
axes[1].tick_params(axis='x', rotation=15)

sns.barplot(data=results_df, x='mutation', y='f1_score', ax=axes[2], palette='Oranges', hue='mutation', legend=False)
axes[2].set_title('Avg F1 by Mutation Method')
axes[2].tick_params(axis='x', rotation=15)

# Standardize Y-axis to make comparison fair
y_min = results_df['f1_score'].min() * 0.95
y_max = results_df['f1_score'].max() * 1.02
for ax in axes:
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Avg F1-Score')
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig('results/visualizations/grid_search_operators_avg.png')
plt.close()

print("Grid search visualizations saved to results/visualizations/")
