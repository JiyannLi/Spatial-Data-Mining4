import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#  创建并训练随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,  # 森林里树的数量
    max_depth=5,       # 每棵树最大深度
    random_state=42
)
rf_model.fit(X_train, y_train)

#  预测测试集
y_pred = rf_model.predict(X_test)

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"测试集准确率: {accuracy:.2f}")
print("混淆矩阵:")
print(conf_mat)
print("\n分类报告:")
print(report)

# 特征重要性可视化
feature_names = iris.feature_names
importances = rf_model.feature_importances_

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("feature score")
plt.ylabel("feature")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Confusing matrix\naccuraccy: {accuracy: 0.2%}")
plt.xlabel("prediction label")
plt.ylabel("true label")
plt.show()