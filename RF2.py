import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#  加载
iris = load_iris()
X = iris.data
y = iris.target

# 选择两个特征（方便二维可视化）
X = X[:, [2, 3]]  # 花瓣长度和宽度

#  划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#  随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 4. 预测
y_pred = rf_model.predict(X_test)

# 5. 评估
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"测试集准确率: {accuracy:.2%}")
print("混淆矩阵:")
print(conf_mat)
print("\n分类报告:")
print(report)

# -----------------------


# -----------------------
# 8. 决策边界可视化
# 创建网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# 网格预测
Z = rf_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
sns.scatterplot(
    x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
    palette=["red", "green", "blue"], s=50
)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("RF decision boundary")
plt.legend(title="classification")
plt.show()
