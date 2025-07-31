# =====================
# 深度伪造图像检测 - 深度学习版本 (从头训练)
# =====================

# =====================
# 1. 导入依赖
# =====================
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# =====================
# 2. 设置参数 (深度学习配置)
# =====================
BASE_PATH = '/kaggle/input/deepfake-and-real-images/Dataset/Train'
IMG_SIZE = 256
LEARNING_RATE = 1e-3  # 从头训练使用更高的学习率
EPOCHS = 60  # 从头训练需要更多轮数
WEIGHT_DECAY = 1e-4  # 添加权重衰减防止过拟合

# 多GPU设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
print(f"🚀 深度学习模式 - 从头训练")
print(f"可用GPU数量: {NUM_GPUS}")
if NUM_GPUS > 1:
    print(f"使用多GPU训练: {[torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]}")
    BATCH_SIZE = 64  # 双GPU时增加batch size
    NUM_WORKERS = 8  # 多GPU时增加数据加载线程
else:
    BATCH_SIZE = 32
    NUM_WORKERS = 4

print(f"Batch Size: {BATCH_SIZE}")
print(f"Num Workers: {NUM_WORKERS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")

# =====================
# 3. 构建标签 DataFrame
# =====================
classes = ['Real', 'Fake']
filepaths, labels = [], []

for label_idx, cls in enumerate(classes):
    folder = os.path.join(BASE_PATH, cls)
    for img_name in os.listdir(folder):
        filepaths.append(os.path.join(folder, img_name))
        labels.append(label_idx)

labels_df = pd.DataFrame({'filepath': filepaths, 'label': labels})
print(f"总图片数: {len(labels_df)}")
print(labels_df.head())

# =====================
# 4. 数据划分
# =====================
train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label'], random_state=42)

# =====================
# 5. 数据增强 & 预处理 (深度学习需要更强的数据增强)
# =====================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # 添加垂直翻转
    transforms.RandomRotation(degrees=15),  # 添加旋转
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # 随机灰度化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1)  # 随机擦除
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================
# 6. 自定义 Dataset
# =====================
class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        label = self.df.iloc[idx]['label']
        return img, label

train_dataset = DeepfakeDataset(train_df, transform=train_transform)
val_dataset = DeepfakeDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=NUM_WORKERS, pin_memory=True)

# =====================
# 7. 构建模型 (EfficientNet-B0 从头训练)
# =====================
print("🔥 创建EfficientNet-B0模型 (无预训练权重)")
model = efficientnet_b0(weights=None)  # 不使用预训练权重，从头训练

# 修改分类器
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)

# 权重初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)
model = model.to(DEVICE)

# 多GPU支持
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"✅ 模型已配置为多GPU训练，使用 {NUM_GPUS} 个GPU")

# =====================
# 8. 损失函数和优化器 (深度学习配置)
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# =====================
# 9. GPU使用监控函数
# =====================
def print_gpu_usage():
    if torch.cuda.is_available():
        for i in range(NUM_GPUS):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - "
                  f"Memory: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB / "
                  f"{torch.cuda.memory_reserved(i)/1024**3:.2f}GB")

# =====================
# 10. 训练 & 验证循环 + Early Stopping
# =====================
best_val_loss = float('inf')
patience = 10  # 从头训练需要更多耐心
trigger_times = 0
train_losses, val_losses, val_accuracies = [], [], []

print("🚀 开始深度学习训练...")
print_gpu_usage()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    # 在第一个epoch显示详细的GPU使用情况
    if epoch == 0:
        print("第一个batch的GPU使用情况:")
    
    for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        # 在第一个epoch的第一个batch显示GPU分布情况
        if epoch == 0 and batch_idx == 0:
            print(f"输入数据形状: {imgs.shape}")
            if NUM_GPUS > 1:
                print(f"数据将分布到 {NUM_GPUS} 个GPU上")
                print_gpu_usage()
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
    
    # 每10个epoch显示一次GPU使用情况
    if (epoch + 1) % 10 == 0:
        print_gpu_usage()
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        # 保存模型时处理多GPU情况
        if NUM_GPUS > 1:
            torch.save(model.module.state_dict(), "best_model_deep_learning.pth")
        else:
            torch.save(model.state_dict(), "best_model_deep_learning.pth")
        print("✅ Best model saved.")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("⛔ Early stopping triggered.")
            break

# =====================
# 11. 训练曲线可视化
# =====================
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss Curve (Deep Learning)")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,3,2)
plt.plot(val_accuracies, label='Val Acc', color='green')
plt.legend()
plt.title("Accuracy Curve (Deep Learning)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1,3,3)
epochs_range = range(len(train_losses))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.fill_between(epochs_range, train_losses, alpha=0.3)
plt.fill_between(epochs_range, val_losses, alpha=0.3)
plt.legend()
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

# =====================
# 12. 混淆矩阵 & 报告
# =====================
plt.figure(figsize=(8,6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Deep Learning)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

print("\n" + "="*50)
print("深度学习训练完成！")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=classes))

# =====================
# 13. 最终统计
# =====================
print(f"\n📊 训练统计:")
print(f"总训练轮数: {len(train_losses)}")
print(f"最佳验证损失: {best_val_loss:.4f}")
print(f"最终验证准确率: {val_accuracies[-1]:.4f}")
print(f"最高验证准确率: {max(val_accuracies):.4f}")
print(f"模型保存路径: best_model_deep_learning.pth")