import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset class distribution
df = pd.read_csv('data/parkinsons_preprocessed.csv')
counts = df['status'].value_counts().sort_index()

counts.plot(kind='bar')
plt.xticks([0, 1], ['Healthy', "Parkinson's"], rotation=0)
plt.ylabel('Count')
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig('results/visualizations/class_distribution.png')
plt.close()