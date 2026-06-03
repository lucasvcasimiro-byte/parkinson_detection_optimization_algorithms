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
plt.savefig('results/class_distribution.png')
plt.close() # Closed instead of show() so it doesn't block the script
print("Saved: results/class_distribution.png")




#### Podem ser melhorados


# Ga operators grid search visualizations
csv_path = 'results/ga_grid_search.csv'
if os.path.exists(csv_path):
    print(f"\nFound {csv_path}, generating grid search visualizations...")
    results_df = pd.read_csv(csv_path)
    
    # Create a clean label for configurations
    results_df['config'] = (
        results_df['selection'].str.replace('_selection', '') + ' + ' + 
        results_df['crossover'].str.replace('_crossover', '') + ' + ' + 
        results_df['mutation'].str.replace('_mutation', '')
    )

    # 1. Top 10 configurations by F1-Score
    top_10 = results_df.sort_values('f1_score', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_10, x='f1_score', y='config', palette='viridis', hue='config', legend=False)
    plt.title('Top 10 GA Configurations by F1-Score')
    plt.xlabel('F1-Score (Parkinson\'s Class)')
    plt.ylabel('Configuration (Selection + Crossover + Mutation)')
    
    # Zoom in on the differences
    min_f1 = top_10['f1_score'].min()
    max_f1 = top_10['f1_score'].max()
    plt.xlim(max(0, min_f1 - 0.02), min(1.0, max_f1 + 0.01))
    
    plt.tight_layout()
    plt.savefig('results/grid_search_top10.png')
    plt.close()
    print("Saved: results/grid_search_top10.png")
    
    # 2. Average performance by Operator type
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
    plt.savefig('results/grid_search_operators_avg.png')
    plt.close()
    print("Saved: results/grid_search_operators_avg.png")
else:
    print(f"\n{csv_path} not found. Run main.py first to generate the grid search data.")
