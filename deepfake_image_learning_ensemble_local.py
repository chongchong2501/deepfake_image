# =====================
# 深度伪造图像检测 - 多模型集成投票版本 (本地优化版)
# 针对 8GB RTX 4070 Laptop 优化
# =====================

# Cell 1: 导入依赖和环境设置
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import gc
import time
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 避免重复初始化输出
if not hasattr(torch, '_deepfake_initialized'):
    print("🚀 本地多模型集成深度伪造检测 (8GB RTX 4070 Laptop 优化版)")
    print(f"PyTorch版本: {torch.__version__}")
    torch._deepfake_initialized = True

# Cell 2: 参数配置 (本地优化)
# 本地数据路径 - 请根据实际情况修改
BASE_PATH = r'E:\program\deepfake_image\Dataset'  # 修改为你的数据集路径
TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
VAL_PATH = os.path.join(BASE_PATH, 'Validation')

# 训练参数 (针对8GB显存优化 - 提升GPU利用率)
# 高GPU利用率模式 - 如果显存不足可以降低这些参数
HIGH_GPU_UTILIZATION = True  # 设置为False可降低显存使用


IMG_SIZE = 256 
BATCH_SIZE = 28  
LEARNING_RATE = 1e-4
EPOCHS = 15
WEIGHT_DECAY = 1e-4
PATIENCE = 4  # 使用原版早停耐心
NUM_WORKERS = 4

# GPU设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只在第一次初始化时输出GPU信息
if not hasattr(torch, '_deepfake_gpu_info_shown'):
    print(f"使用设备: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        # 设置显存优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # 启用混合精度训练
        USE_AMP = True
        print("✅ 启用混合精度训练以节省显存")
    else:
        USE_AMP = False
        print("使用CPU训练")
    
    torch._deepfake_gpu_info_shown = True
else:
    # 静默设置GPU参数
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        USE_AMP = True
    else:
        USE_AMP = False



# Cell 3: 数据加载函数
classes = ['Real', 'Fake']

def create_dataframe(data_path, dataset_type):
    """创建数据集DataFrame"""
    filepaths, labels = [], []
    
    for label_idx, cls in enumerate(classes):
        folder = os.path.join(data_path, cls)
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(folder, img_name))
                    labels.append(label_idx)
    
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    print(f"{dataset_type}集图片数: {len(df)}")
    if len(df) > 0:
        print(f"{dataset_type}集类别分布:")
        for idx, cls in enumerate(classes):
            count = len(df[df['label'] == idx])
            print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
    return df

# Cell 4: 数据预处理和增强 (轻量化)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),  # 减少旋转角度
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 减少增强强度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        img = cv2.imread(img_path)
        if img is None:
            # 如果图片读取失败，返回一个默认图片
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        label = self.df.iloc[idx]['label']
        return img, label

# Cell 5: 模型定义 (使用原版模型)
def create_efficientnet_b0():
    """创建EfficientNet-B0模型"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(1280, 2)
    return model

def create_resnet18():
    """创建ResNet18模型"""
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 2)
    return model

def create_convnext_tiny():
    """创建ConvNeXt-Tiny模型"""
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(768, 2)
    return model

# 模型配置字典 (使用原版模型)
MODEL_CONFIGS = {
    'efficientnet_b0': {
        'create_fn': create_efficientnet_b0,
        'name': 'EfficientNet-B0'
    },
    'resnet18': {
        'create_fn': create_resnet18,
        'name': 'ResNet18'
    },
    'convnext_tiny': {
        'create_fn': create_convnext_tiny,
        'name': 'ConvNeXt-Tiny'
    }
}

# Cell 6: 优化的单模型训练函数
def train_single_model(model_key, train_loader, val_loader, save_path):
    """训练单个模型 (显存优化版)"""
    print(f"\n🔥 开始训练 {MODEL_CONFIGS[model_key]['name']}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 创建模型
    model = MODEL_CONFIGS[model_key]['create_fn']()
    model = model.to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 混合精度训练 - 优化设置
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) if USE_AMP else None
    
    # 训练记录
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []
    train_accs, learning_rates = [], []
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            # 清理显存
            del imgs, labels, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 计算训练准确率
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                else:
                    outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                del imgs, labels, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        train_acc = train_correct / train_total
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # 清理显存
                del imgs, labels, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 最佳模型已保存，验证准确率: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⛔ Early stopping triggered")
                break
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    
    return {
        'best_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accuracies,
        'learning_rates': learning_rates,
        'training_time': training_time
    }

# Cell 7: 集成预测函数
def load_trained_models(model_paths):
    """加载训练好的模型"""
    models_dict = {}
    for model_key, path in model_paths.items():
        if os.path.exists(path):
            model = MODEL_CONFIGS[model_key]['create_fn']()
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            models_dict[model_key] = model
            print(f"✅ 已加载 {MODEL_CONFIGS[model_key]['name']}")
        else:
            print(f"❌ 模型文件不存在: {path}")
    return models_dict

def ensemble_predict(models_dict, data_loader, voting_type='soft', weights=None):
    """集成预测 (显存优化版)"""
    all_predictions = []
    all_labels = []
    model_outputs = {key: [] for key in models_dict.keys()}
    
    # 如果是加权投票但没有提供权重，则使用等权重
    if voting_type == 'weighted' and weights is None:
        weights = {key: 1.0 for key in models_dict.keys()}
    
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="集成预测"):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            # 收集每个模型的预测
            batch_predictions = []
            for model_key, model in models_dict.items():
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                else:
                    outputs = model(imgs)
                
                if voting_type in ['soft', 'weighted']:
                    probs = torch.softmax(outputs, dim=1)
                    batch_predictions.append(probs.cpu().numpy())
                else:  # hard voting
                    _, predicted = torch.max(outputs, 1)
                    batch_predictions.append(predicted.cpu().numpy())
                
                model_outputs[model_key].extend(outputs.cpu().numpy())
                del outputs
            
            # 集成预测
            if voting_type == 'soft':
                ensemble_probs = np.mean(batch_predictions, axis=0)
                ensemble_pred = np.argmax(ensemble_probs, axis=1)
            elif voting_type == 'weighted':
                weighted_probs = np.zeros_like(batch_predictions[0])
                total_weight = 0
                for i, (model_key, probs) in enumerate(zip(models_dict.keys(), batch_predictions)):
                    weight = weights[model_key]
                    weighted_probs += probs * weight
                    total_weight += weight
                ensemble_probs = weighted_probs / total_weight
                ensemble_pred = np.argmax(ensemble_probs, axis=1)
            else:
                batch_predictions = np.array(batch_predictions)
                ensemble_pred = []
                for i in range(batch_predictions.shape[1]):
                    votes = batch_predictions[:, i]
                    ensemble_pred.append(np.bincount(votes).argmax())
                ensemble_pred = np.array(ensemble_pred)
            
            all_predictions.extend(ensemble_pred)
            all_labels.extend(labels.cpu().numpy())
            
            # 清理显存
            del imgs, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return np.array(all_predictions), np.array(all_labels), model_outputs

def calculate_model_weights(model_results, weight_method='accuracy'):
    """计算模型权重"""
    weights = {}
    
    if weight_method == 'accuracy':
        # 基于验证准确率计算权重
        accuracies = {key: results['best_acc'] for key, results in model_results.items()}
        total_acc = sum(accuracies.values())
        
        for key, acc in accuracies.items():
            weights[key] = acc / total_acc
            
    elif weight_method == 'softmax':
        # 使用softmax归一化准确率作为权重
        accuracies = np.array([results['best_acc'] for results in model_results.values()])
        softmax_weights = np.exp(accuracies * 10) / np.sum(np.exp(accuracies * 10))  # 乘以10增强差异
        
        for i, key in enumerate(model_results.keys()):
            weights[key] = softmax_weights[i]
            
    elif weight_method == 'rank':
        # 基于排名的权重分配
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
        n_models = len(sorted_models)
        
        for i, (key, _) in enumerate(sorted_models):
            weights[key] = (n_models - i) / sum(range(1, n_models + 1))
    
    return weights

def plot_training_history(model_results, save_dir='./works/plots'):
    """绘制训练历史"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 为每个模型绘制训练曲线
    for model_name, results in model_results.items():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(results['train_losses']) + 1)
        
        # 训练和验证损失
        ax1.plot(epochs, results['train_losses'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, results['val_losses'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title(f'{model_name} - 损失曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 训练和验证准确率
        ax2.plot(epochs, results['train_accs'], 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, results['val_accs'], 'r-', label='验证准确率', linewidth=2)
        ax2.set_title(f'{model_name} - 准确率曲线', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 学习率变化
        ax3.plot(epochs, results['learning_rates'], 'g-', linewidth=2)
        ax3.set_title(f'{model_name} - 学习率变化', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 验证准确率分布
        ax4.hist(results['val_accs'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(results['best_acc'], color='red', linestyle='--', linewidth=2, 
                   label=f'最佳准确率: {results["best_acc"]:.4f}')
        ax4.set_title(f'{model_name} - 验证准确率分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Validation Accuracy')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制所有模型的比较图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 验证损失比较
    for i, (model_name, results) in enumerate(model_results.items()):
        epochs = range(1, len(results['val_losses']) + 1)
        ax1.plot(epochs, results['val_losses'], color=colors[i % len(colors)], 
                label=model_name, linewidth=2)
    ax1.set_title('所有模型验证损失比较', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证准确率比较
    for i, (model_name, results) in enumerate(model_results.items()):
        epochs = range(1, len(results['val_accs']) + 1)
        ax2.plot(epochs, results['val_accs'], color=colors[i % len(colors)], 
                label=model_name, linewidth=2)
    ax2.set_title('所有模型验证准确率比较', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 最佳准确率柱状图
    model_names = list(model_results.keys())
    best_accs = [results['best_acc'] for results in model_results.values()]
    bars = ax3.bar(model_names, best_accs, color=colors[:len(model_names)], alpha=0.7)
    ax3.set_title('各模型最佳验证准确率', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Best Validation Accuracy')
    ax3.set_ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 训练时间比较（如果有的话）
    if 'training_time' in list(model_results.values())[0]:
        training_times = [results['training_time'] for results in model_results.values()]
        bars = ax4.bar(model_names, training_times, color=colors[:len(model_names)], alpha=0.7)
        ax4.set_title('各模型训练时间', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Training Time (seconds)')
        
        for bar, time in zip(bars, training_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.01,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, '训练时间数据不可用', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('训练时间', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练历史图表已保存到 {save_dir}")

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_ensemble_analysis(model_outputs, y_true, y_pred, class_names, save_dir='./works/plots'):
    """绘制集成分析图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 模型预测一致性分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 各模型预测概率分布
    ax1 = axes[0, 0]
    for model_name, outputs in model_outputs.items():
        outputs_array = np.array(outputs)
        probs = torch.softmax(torch.tensor(outputs_array), dim=1).numpy()
        max_probs = np.max(probs, axis=1)
        ax1.hist(max_probs, alpha=0.6, label=model_name, bins=30)
    
    ax1.set_title('各模型最大预测概率分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('最大预测概率')
    ax1.set_ylabel('频次')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测一致性热图
    ax2 = axes[0, 1]
    model_names = list(model_outputs.keys())
    n_models = len(model_names)
    consistency_matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                outputs1 = np.array(model_outputs[model1])
                outputs2 = np.array(model_outputs[model2])
                pred1 = np.argmax(outputs1, axis=1)
                pred2 = np.argmax(outputs2, axis=1)
                consistency = np.mean(pred1 == pred2)
                consistency_matrix[i, j] = consistency
            else:
                consistency_matrix[i, j] = 1.0
    
    im = ax2.imshow(consistency_matrix, cmap='RdYlBu', vmin=0, vmax=1)
    ax2.set_xticks(range(n_models))
    ax2.set_yticks(range(n_models))
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.set_yticklabels(model_names)
    ax2.set_title('模型间预测一致性', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i in range(n_models):
        for j in range(n_models):
            text = ax2.text(j, i, f'{consistency_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # 3. 集成预测置信度分析
    ax3 = axes[1, 0]
    
    # 计算集成预测的置信度
    ensemble_outputs = np.mean([np.array(outputs) for outputs in model_outputs.values()], axis=0)
    ensemble_probs = torch.softmax(torch.tensor(ensemble_outputs), dim=1).numpy()
    ensemble_confidence = np.max(ensemble_probs, axis=1)
    
    # 按正确/错误预测分组
    correct_mask = (y_pred == y_true)
    correct_confidence = ensemble_confidence[correct_mask]
    wrong_confidence = ensemble_confidence[~correct_mask]
    
    ax3.hist(correct_confidence, alpha=0.7, label=f'正确预测 ({len(correct_confidence)})', 
             bins=30, color='green')
    ax3.hist(wrong_confidence, alpha=0.7, label=f'错误预测 ({len(wrong_confidence)})', 
             bins=30, color='red')
    
    ax3.set_title('集成预测置信度分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('预测置信度')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 类别预测准确率
    ax4 = axes[1, 1]
    from sklearn.metrics import classification_report
    
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    bars = ax4.bar(class_names, class_accuracies, color=['skyblue', 'lightcoral'])
    ax4.set_title('各类别预测准确率', fontsize=14, fontweight='bold')
    ax4.set_ylabel('准确率')
    ax4.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"集成分析图表已保存到 {save_dir}")

# Cell 8: 主训练流程
def main():
    """主训练流程"""
    print("📂 加载数据集...")
    
    # 检查数据路径
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print("❌ 数据路径不存在，请修改 BASE_PATH 变量")
        print(f"当前训练路径: {TRAIN_PATH}")
        print(f"当前验证路径: {VAL_PATH}")
        return
    
    train_df = create_dataframe(TRAIN_PATH, "训练")
    val_df = create_dataframe(VAL_PATH, "验证")
    
    if len(train_df) == 0 or len(val_df) == 0:
        print("❌ 数据集为空，请检查数据路径和文件格式")
        return
    
    # 限制验证集大小以节省显存
    MAX_VAL_SAMPLES = 3200
    if len(val_df) > MAX_VAL_SAMPLES:
        print(f"⚠️ 验证集过大 ({len(val_df)} 张)，随机采样 {MAX_VAL_SAMPLES} 张图片")
        val_df = val_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_VAL_SAMPLES//2), random_state=42)
        ).reset_index(drop=True)
        print(f"✅ 验证集采样完成，当前大小: {len(val_df)}")
    
    # 创建数据集和数据加载器 (优化GPU利用率)
    train_dataset = DeepfakeDataset(train_df, transform=train_transform)
    val_dataset = DeepfakeDataset(val_df, transform=val_transform)
    
    # 优化数据加载器设置以更好利用GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                             persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)
    
    print(f"\n📊 数据集总览:")
    print(f"训练集总数: {len(train_df)}")
    print(f"验证集总数: {len(val_df)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 只在第一次显示GPU相关参数
    if not hasattr(torch, '_deepfake_dataset_info_shown'):
        print(f"图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
        print(f"批次大小: {BATCH_SIZE}")
        print(f"数据加载线程: {NUM_WORKERS}")
        torch._deepfake_dataset_info_shown = True
    
    # 训练所有模型
    print("\n🚀 开始训练多个模型...")
    
    # 创建输出目录
    output_dir = './works'
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择要训练的模型 (轻量化组合)
    selected_models = ['efficientnet_b0', 'resnet18','convnext_tiny']
    model_paths = {}
    model_results = {}
    
    for model_key in selected_models:
        save_path = os.path.join(output_dir, f"best_{model_key}_model_local.pth")
        model_paths[model_key] = save_path
        
        print(f"\n{'='*50}")
        print(f"训练模型: {MODEL_CONFIGS[model_key]['name']}")
        print(f"{'='*50}")
        
        result = train_single_model(
            model_key, train_loader, val_loader, save_path
        )
        
        model_results[model_key] = result
        
        print(f"✅ {MODEL_CONFIGS[model_key]['name']} 训练完成，最佳验证准确率: {result['best_acc']:.4f}")
        
        # 强制清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 集成预测和评估
    print("\n🔮 开始集成预测...")
    
    # 加载训练好的模型
    trained_models = load_trained_models(model_paths)
    
    if len(trained_models) == 0:
        print("❌ 没有成功加载任何模型")
        return
    
    # 计算模型权重
    model_weights = calculate_model_weights(model_results, weight_method='accuracy')
    print("\n⚖️ 模型权重分配:")
    for model_key, weight in model_weights.items():
        print(f"  {MODEL_CONFIGS[model_key]['name']:20}: {weight:.4f}")
    
    # 软投票预测
    print("\n📊 软投票集成预测:")
    soft_predictions, true_labels, val_model_outputs = ensemble_predict(trained_models, val_loader, voting_type='soft')
    soft_accuracy = accuracy_score(true_labels, soft_predictions)
    print(f"软投票准确率: {soft_accuracy:.4f}")
    
    # 硬投票预测
    print("\n📊 硬投票集成预测:")
    hard_predictions, _, _ = ensemble_predict(trained_models, val_loader, voting_type='hard')
    hard_accuracy = accuracy_score(true_labels, hard_predictions)
    print(f"硬投票准确率: {hard_accuracy:.4f}")
    
    # 加权投票预测
    print("\n📊 加权投票集成预测:")
    weighted_predictions, _, _ = ensemble_predict(trained_models, val_loader, voting_type='weighted', weights=model_weights)
    weighted_accuracy = accuracy_score(true_labels, weighted_predictions)
    print(f"加权投票准确率: {weighted_accuracy:.4f}")
    
    # 可视化
    print("\n" + "="*50)
    print("生成可视化图表")
    print("="*50)
    
    # 创建可视化输出目录
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制训练历史
    plot_training_history(model_results, save_dir=plots_dir)
    
    # 绘制混淆矩阵
    class_names = ['Real', 'Fake']
    plot_confusion_matrix(true_labels, soft_predictions, class_names, 
                         '集成模型混淆矩阵', 
                         save_path=os.path.join(plots_dir, 'ensemble_confusion_matrix.png'))
    
    # 绘制集成分析
    plot_ensemble_analysis(val_model_outputs, true_labels, soft_predictions, 
                          class_names, save_dir=plots_dir)
    
    # 结果总结
    print("\n" + "="*60)
    print("🎉 多模型集成训练完成！")
    print("="*60)
    print(f"训练的模型数量: {len(selected_models)}")
    
    for model_key in selected_models:
        best_acc = model_results[model_key]['best_acc']
        print(f"{MODEL_CONFIGS[model_key]['name']:20}: {best_acc:.4f}")
    
    print(f"{'软投票集成':20}: {soft_accuracy:.4f}")
    print(f"{'硬投票集成':20}: {hard_accuracy:.4f}")
    print(f"{'加权投票集成':20}: {weighted_accuracy:.4f}")
    
    # 找出最佳方法
    best_single = max([results['best_acc'] for results in model_results.values()])
    ensemble_results = {
        '软投票': soft_accuracy,
        '硬投票': hard_accuracy,
        '加权投票': weighted_accuracy
    }
    best_ensemble = max(ensemble_results, key=ensemble_results.get)
    
    print(f"\n🏆 最佳单模型准确率: {best_single:.4f}")
    print(f"🏆 最佳集成方法: {best_ensemble} (准确率: {ensemble_results[best_ensemble]:.4f})")
    
    improvement = (ensemble_results[best_ensemble] - best_single) * 100
    print(f"🚀 集成提升: {improvement:+.2f}%")
    
    print(f"\n💾 保存的模型文件:")
    for model_key, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path}")
    
    # 保存结果
    import json
    results = {
        'model_results': {k: {key: val for key, val in v.items() if key != 'model'} for k, v in model_results.items()},
        'ensemble_results': {
            'soft_voting': float(soft_accuracy),
            'hard_voting': float(hard_accuracy),
            'weighted_voting': float(weighted_accuracy)
        },
        'model_weights': model_weights
    }
    
    results_path = os.path.join(output_dir, 'ensemble_results_local.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ 训练完成！结果已保存到 {results_path}")
    print(f"📊 可视化图表已保存到 {plots_dir} 目录")

if __name__ == "__main__":
    main()