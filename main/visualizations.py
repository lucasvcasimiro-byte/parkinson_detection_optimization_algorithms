import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/parkinsons_preprocessed.csv')
counts = df['status'].value_counts().sort_index()

# Simple bar chart to visualize dataset balance
counts.plot(kind='bar')
plt.xticks([0, 1], ['Healthy', "Parkinson's"], rotation=0)
plt.ylabel('Count')
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig('results/visualizations/class_distribution.png')
plt.close() 


# Ga operators grid search visualizations
results_df = pd.read_csv('results/ga_grid_search.csv')
    
# Clean label for configurations
results_df['config'] = (
        'Arch ' + results_df['architecture'].astype(str) + ' + ' +
        results_df['init_method'] + ' + ' +
        results_df['selection'].str.replace('_selection', '') + ' + ' + 
        results_df['crossover'].str.replace('_crossover', '') + ' + ' + 
        results_df['mutation'].str.replace('_mutation', '')
    )

### Top 10 configurations by F1-Score
top_10 = results_df.sort_values('f1_score', ascending=False).head(10)
    
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10, x='f1_score', y='config', palette='viridis', hue='config', legend=False)
plt.title('Top 10 GA Configurations by F1-Score')
plt.xlabel('F1-Score (Parkinson\'s Class)')
plt.ylabel('Configuration (Selection + Crossover + Mutation)')
    
# Zoom in on differences
min_f1 = top_10['f1_score'].min()
max_f1 = top_10['f1_score'].max()
plt.xlim(max(0, min_f1 - 0.02), min(1.0, max_f1 + 0.01))    
plt.tight_layout()
plt.savefig('results/visualizations/grid_search_top10.png')
plt.close()
    
### Average operators performance
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


# GA vs GWO comparison visualizations 
compare_path = 'results/csv/ga_vs_gwo.csv'

cmp_df = pd.read_csv('results/csv/ga_vs_gwo.csv')

### Boxplot - F1-Score distribution across 30 runs
plt.figure(figsize=(6, 5))
sns.boxplot(data=cmp_df, x='algorithm', y='f1_score', palette={'GA': '#4c72b0', 'GWO': '#dd8452'})
plt.title('GA vs GWO — F1-Score Distribution (30 runs)')
plt.xlabel('Algorithm')
plt.ylabel('F1-Score (Parkinson\'s Class)')
plt.tight_layout()
plt.savefig('results/visualizations/ga_vs_gwo_boxplot.png')
plt.close()


### Bar chart - mean of all metrics side by side
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
