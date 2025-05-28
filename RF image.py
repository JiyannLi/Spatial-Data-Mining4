import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 设置数据路径
data_dir = 'D:\pycharm\KJ_Zuoye4\cifar-10-images\cifar-10-images\cifar-10-batches-py'

# CIFAR-10 类别标签（可用于混淆矩阵）
label_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 读取单个 batch 的函数
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']              # 图像数据 (N, 3072)
        labels = batch[b'labels']          # 标签
        return data, labels

# 读取训练数据（5个batch）
X_train = []
y_train = []
for i in range(1, 6):
    batch_path = os.path.join(data_dir, f'data_batch_{i}')
    data, labels = load_cifar_batch(batch_path)
    X_train.append(data)
    y_train += labels

X_train = np.concatenate(X_train)
y_train = np.array(y_train)

# 读取测试数据
X_test, y_test = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
X_test = np.array(X_test)
y_test = np.array(y_test)

# 归一化到 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# 降低训练量（可选，加速测试）
X_train_sample = X_train[:10000]
y_train_sample = y_train[:10000]
X_test_sample = X_test[:2000]
y_test_sample = y_test[:2000]

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
rf.fit(X_train_sample, y_train_sample)

# 预测并评估
y_pred = rf.predict(X_test_sample)
acc = accuracy_score(y_test_sample, y_pred)

print(f"准确率: {acc:.2%}")
print("\n分类报告:")
print(classification_report(y_test_sample, y_pred, target_names=label_names))

# 混淆矩阵可视化
conf_mat = confusion_matrix(y_test_sample, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_names, yticklabels=label_names)
plt.title(f"CIFAR-10 RF result\naccurary: {acc:.2%}")
plt.xlabel("predict label")
plt.ylabel("true label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
