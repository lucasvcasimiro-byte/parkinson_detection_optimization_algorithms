import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/parkinsons_preprocessed.csv')
counts = df['status'].value_counts().sort_index()

counts.plot(kind='bar')
plt.xticks([0, 1], ['Healthy', "Parkinson's"], rotation=0)
plt.ylabel('Count')
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig('results/class_distribution.png')
plt.show()
