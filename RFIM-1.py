from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import os
import pickle
import numpy as np

def load_cifar_batch(filename):
    """读取单个batch的图像和标签"""
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X = data_dict[b'data']
        y = data_dict[b'labels']
        X = X.reshape(len(X), 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为RGB图像格式
        return X, np.array(y)

def load_cifar10_data(data_dir, num_train=3000, num_test=1000):
    """读取前num_train张训练图像和num_test张测试图像"""
    X_train, y_train = [], []
    for i in range(1, 6):
        X_batch, y_batch = load_cifar_batch(os.path.join(data_dir, f"data_batch_{i}"))
        X_train.append(X_batch)
        y_train.append(y_batch)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar_batch(os.path.join(data_dir, "test_batch"))

    return X_train[:num_train], y_train[:num_train], X_test[:num_test], y_test[:num_test]

# ✅ 修改为你本地的数据路径
data_dir = r'D:\pycharm\KJ_Zuoye4\cifar-10-images\cifar-10-images\cifar-10-batches-py'  # ←改成你解压后的路径
X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. 方法一：HOG 特征提取函数
def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # 转灰度
        features = hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        hog_features.append(features)
    return np.array(hog_features)

print("🔍 Extracting HOG features...")
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# 4. HOG + RF 训练与评估
rf_hog = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
rf_hog.fit(X_train_hog, y_train)
y_pred_hog = rf_hog.predict(X_test_hog)
acc_hog = accuracy_score(y_test, y_pred_hog)
print(f"HOG + RF Accuracy: {acc_hog:.2%}")

# 5. 方法二：PCA 降维
print("🔍 Applying PCA on flattened images...")
X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
X_test_flat = X_test.reshape(len(X_test), -1) / 255.0

pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# 6. PCA + RF 训练与评估
rf_pca = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"PCA + RF Accuracy: {acc_pca:.2%}")

