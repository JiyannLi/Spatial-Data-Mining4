from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 参数组合
n_estimators_range = [10, 50, 100, 200]
max_depth_range = [3, 5, 10, None]

# 结果记录
results = []

for n in n_estimators_range:
    for d in max_depth_range:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        scores = cross_val_score(rf, X, y, cv=5)
        results.append({
            'n_estimators': n,
            'max_depth': d if d is not None else 'None',
            'mean_accuracy': np.mean(scores)
        })

# 打印结果表格
df_results = pd.DataFrame(results)
print(df_results)
df_results.sort_values(by='mean_accuracy', ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'max_depth' is string type for hue grouping
df_results['max_depth'] = df_results['max_depth'].astype(str)

# Set visualization style
sns.set(style="whitegrid")

# Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_results,
    x='n_estimators',
    y='mean_accuracy',
    hue='max_depth',
    palette='viridis'
)

# Titles and axis labels
plt.title('Cross-validated Accuracy of Random Forest under Different Parameters', fontsize=14)
plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Mean Accuracy (5-Fold CV)', fontsize=12)
plt.ylim(0.9, 1.01)
plt.legend(title='Max Tree Depth')
plt.tight_layout()
plt.show()

