"""
单阶段Teacher训练脚本 - 使用Blur增强数据 (Mixed: Motion + CCMBA)
- DinoV3 Adapter (Backbone Frozen)
- 全面autocast半精度训练
- 仅训练模式 (No Test/Eval)
- 支持多机多卡 (Multi-Node Distributed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from PIL import Image
import random
import os
import json
import time
import argparse
import logging
import traceback
import sys
from transformers import AutoModel
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

corrupted_images = []

def add_corrupted_image(image_path, error_msg):
    """线程安全地添加损坏图片记录"""
    global corrupted_images
    corrupted_images.append({
        'path': str(image_path),
        'error': str(error_msg)
    })

def save_corrupted_images_report(output_dir, rank=0):
    """保存损坏图片报告"""
    if rank == 0 and corrupted_images:
        report_path = os.path.join(output_dir, 'corrupted_images_report.json')
        with open(report_path, 'w') as f:
            json.dump({
                'total_corrupted': len(corrupted_images),
                'corrupted_images': corrupted_images
            }, f, indent=2)
        
        print(f"{'='*60}")
        print(f"CORRUPTED IMAGES REPORT")
        print(f"{'='*60}")
        print(f"Total corrupted images found: {len(corrupted_images)}")
        print(f"Report saved to: {report_path}")

# 分布式配置函数
def setup_distributed(rank, world_size, backend=None):
    """初始化分布式训练环境"""
    if backend is None:
        if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
            backend = 'nccl'
        else:
            backend = 'gloo'
    
    if backend == 'nccl':
        os.environ['NCCL_TIMEOUT'] = '1800'
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'
    
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    if backend == 'nccl':
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available")
        # 注意：这里不能单纯 set_device(rank)，因为在多机环境下 rank 会超过单机 GPU 数量
        # 这里仅做 backend 初始化，device 设置移至 main 函数
        pass 
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    if dist.is_initialized():
        print(f"[Rank {rank}] Distributed training initialized successfully")

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(rank):
    """设置日志记录"""
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)


# ===============================
# 配置参数
# ===============================
# 模型配置
DINOV3_MODEL_ID = "/home/work/xueyunqi/dinov3-vit7b16-pretrain-lvd1689m"

# 数据配置
TRAIN_ROOT_FOLDER = "/home/work/xueyunqi/11ar_datasets/extracted"
# 移除了 TEST 相关路径配置

# 训练配置
LEARNING_RATE = 1e-4
PROJECTION_DIM = 512
NUM_WORKERS = 4

# Motion blur配置 (保持混合模式)
BLUR_PROB = 0.2
BLUR_STRENGTH_RANGE = (0.1, 0.3)
BLUR_MODE = "mixed"
MIXED_MODE_RATIO = 0.5
CCMBA_DATA_DIR = "/home/work/xueyunqi/11ar_datasets/progan_ccmba_train"

# 数据集名称
DATASET_NAME = "teacher_mixed_blur_training_notest"

# ===============================
# Motion Blur工具函数
# ===============================
def apply_motion_blur(image_pil, strength):
    """应用运动模糊"""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    kernel_size = int(5 + (strength - 0.05) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, min(kernel_size, 31))
    
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / np.sum(kernel)
    
    blurred_cv = cv2.filter2D(image_cv, -1, kernel)
    blurred_pil = Image.fromarray(cv2.cvtColor(blurred_cv, cv2.COLOR_BGR2RGB))
    return blurred_pil

def apply_motion_blur_batch_efficient(images, strength):
    """高效的批量运动模糊处理"""
    if not isinstance(images, torch.Tensor):
        return apply_motion_blur_batch(images, strength)
    
    device = images.device
    batch_size, channels, height, width = images.shape
    
    kernel_size = int(5 + (strength - 0.05) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, min(kernel_size, 31))
    
    kernel = torch.zeros(kernel_size, kernel_size, device=device)
    kernel[kernel_size//2, :] = 1.0
    kernel = kernel / kernel.sum()
    
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
    
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
    
    return blurred

def apply_mixed_blur_enhanced(original_tensor, ccmba_loader, image_name, category, is_real, blur_strength_range, mixed_ratio=0.5):
    """应用混合模糊：结合全局模糊和CCMBA部分模糊"""
    device = original_tensor.device
    blur_info = {'mode': 'mixed', 'ccmba_used': False, 'global_used': False}
    
    if random.random() < mixed_ratio:
        ccmba_data_start = time.time()
        blurred_pil, blur_mask, metadata = ccmba_loader.load_ccmba_blur_data(image_name, category, is_real)
        ccmba_data_time = time.time() - ccmba_data_start
        
        if blurred_pil is not None:
            ccmba_transform_start = time.time()
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomCrop((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            blurred_tensor = transform(blurred_pil).to(device)
            ccmba_transform_time = time.time() - ccmba_transform_start
            
            blur_info.update({
                'ccmba_used': True,
                'ccmba_data_load_time': ccmba_data_time,
                'ccmba_transform_time': ccmba_transform_time,
                'blur_mask_ratio': np.sum(blur_mask) / blur_mask.size if blur_mask is not None else 0,
                'metadata': metadata
            })
            
            return blurred_tensor, blur_info
    
    global_blur_start = time.time()
    blur_strength = random.uniform(*blur_strength_range)
    blurred_tensor = apply_motion_blur_batch_efficient(original_tensor.unsqueeze(0), blur_strength).squeeze(0)
    global_blur_time = time.time() - global_blur_start
    
    blur_info.update({
        'global_used': True,
        'global_blur_time': global_blur_time,
        'blur_strength': blur_strength
    })
    
    return blurred_tensor, blur_info

def apply_motion_blur_batch(images, strength):
    """批量应用运动模糊"""
    if isinstance(images, torch.Tensor):
        return apply_motion_blur_batch_efficient(images, strength)
    elif isinstance(images, list):
        return [apply_motion_blur(img, strength) for img in images]
    else:
        return apply_motion_blur(images, strength)

# ===============================
# 数据增强
# ===============================
def get_train_transforms():
    """训练数据增强"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# 移除了 get_test_transforms

# ===============================
# L2Norm层
# ===============================
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

# ===============================
# DinoV3 Adapter模型
# ===============================
class ImprovedDinoV3Adapter(nn.Module):
    """改进的DinoV3模型适配器"""
    
    def __init__(self, model_path, num_classes=2, projection_dim=512, 
                 adapter_layers=2, dropout_rate=0.1, device="cuda"):
        super().__init__()
        
        self.device = device
        self.projection_dim = projection_dim
        
        print(f"Loading DinoV3 model: {model_path}")
        
        self.backbone = self._load_dinov3_backbone(model_path, device)
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 4096)
        
        print(f"DinoV3 feature dimension: {self.hidden_size}")
        
        # 特征投影器
        projection_layers = []
        current_dim = self.hidden_size
        
        for i in range(adapter_layers):
            if i == adapter_layers - 1:
                projection_layers.extend([
                    nn.Linear(current_dim, projection_dim),
                    nn.LayerNorm(projection_dim)
                ])
            else:
                next_dim = max(projection_dim, current_dim // 2)
                projection_layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.LayerNorm(next_dim)
                ])
                current_dim = next_dim
        
        self.projection = nn.Sequential(*projection_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim // 2, num_classes)
        )
        
        self.projection = self.projection.to(device)
        self.classifier = self.classifier.to(device)
        
        print(f"✓ DinoV3 Adapter initialized:")
        print(f"  - Backbone: {self.hidden_size}D")
        print(f"  - Projection: {projection_dim}D ({adapter_layers} layers)")
        print(f"  - Classes: {num_classes}")
        
        self._initialize_weights()
    
    def _load_dinov3_backbone(self, model_path, device):
        """加载DinoV3 backbone"""
        try:
            backbone = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            
            backbone = backbone.to(device).eval()
            
            # [关键] 这里设置了 backbone 不进行梯度更新 (Frozen)
            for param in backbone.parameters():
                param.requires_grad = False
            
            return backbone
            
        except Exception as e:
            print(f"Error loading DinoV3 backbone: {e}")
            raise e
    
    def _initialize_weights(self):
        """初始化投影和分类层权重"""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.projection.apply(init_layer)
        self.classifier.apply(init_layer)
    
    def extract_features(self, pixel_values):
        """提取DinoV3特征"""
        with torch.no_grad():
            with autocast():
                outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
                
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0]
                
                return features.float()
    
    def forward(self, pixel_values, return_features=False):
        """前向传播"""
        raw_features = self.extract_features(pixel_values)
        
        with autocast():
            projected_features = self.projection(raw_features)
            logits = self.classifier(projected_features)
        
        if return_features:
            return {
                'logits': logits,
                'projected_features': projected_features,
                'raw_features': raw_features
            }
        else:
            return logits
    
    def get_projected_features(self, pixel_values):
        """获取投影特征"""
        raw_features = self.extract_features(pixel_values)
        with autocast():
            projected_features = self.projection(raw_features)
        return projected_features

# ===============================
# CCMBA数据加载工具
# ===============================
class CCMBADataLoader:
    """CCMBA预处理数据加载器"""
    
    def __init__(self, ccmba_root_dir, categories=None):
        self.ccmba_root_dir = Path(ccmba_root_dir)
        self.categories = categories
        
        self.real_mapping = {}
        self.fake_mapping = {}
        
        self._build_all_mappings()
        
        print(f"CCMBA Data Loader initialized:")
        if self.categories:
            print(f"  - Specified categories: {len(self.categories)}")
        else:
            print(f"  - Auto-discovered categories: {len(self.categories)}")
        print(f"  - Real images: {len(self.real_mapping)} entries")
        print(f"  - Fake images: {len(self.fake_mapping)} entries")
    
    def _build_all_mappings(self):
        """构建所有类别的映射"""
        if self.categories is None:
            self.categories = [d.name for d in self.ccmba_root_dir.iterdir() 
                              if d.is_dir() and ((d / "nature").exists() or (d / "ai").exists())]
            print(f"Auto-discovered CCMBA categories: {self.categories}")
        
        for category in self.categories:
            category_dir = self.ccmba_root_dir / category
            if not category_dir.exists():
                print(f"Warning: Category directory not found: {category_dir}")
                continue
            
            real_dir = category_dir / "nature"
            fake_dir = category_dir / "ai"
            
            if real_dir.exists():
                category_real_mapping = self._build_mapping(real_dir)
                for key, value in category_real_mapping.items():
                    self.real_mapping[f"{category}_{key}"] = value
                print(f"  - {category}/nature (real): {len(category_real_mapping)} entries")
            
            if fake_dir.exists():
                category_fake_mapping = self._build_mapping(fake_dir)
                for key, value in category_fake_mapping.items():
                    self.fake_mapping[f"{category}_{key}"] = value
                print(f"  - {category}/ai (fake): {len(category_fake_mapping)} entries")
    
    def _build_mapping(self, class_dir):
        """构建原图文件名到CCMBA数据的映射"""
        mapping = {}
        
        if not class_dir.exists():
            return mapping
        
        blurred_images_dir = class_dir / "blurred_images"
        blur_masks_dir = class_dir / "blur_masks"
        metadata_dir = class_dir / "metadata"
        
        if not all([blurred_images_dir.exists(), blur_masks_dir.exists(), metadata_dir.exists()]):
            print(f"Warning: Incomplete CCMBA structure in {class_dir}")
            return mapping
        
        for blur_img_path in blurred_images_dir.glob("*.jpg"):
            base_name = blur_img_path.stem
            
            blur_mask_path = blur_masks_dir / f"{base_name}.png"
            metadata_path = metadata_dir / f"{base_name}.json"
            
            if blur_mask_path.exists() and metadata_path.exists():
                mapping[base_name] = {
                    'blurred_image': blur_img_path,
                    'blur_mask': blur_mask_path,
                    'metadata': metadata_path
                }
        
        return mapping
    
    def get_ccmba_data(self, image_name, category, is_real=True):
        """获取指定图像的CCMBA数据"""
        mapping = self.real_mapping if is_real else self.fake_mapping
        key = f"{category}_{image_name}"
        
        if key in mapping:
            return mapping[key]
        else:
            return None
    
    def load_ccmba_blur_data(self, image_name, category, is_real=True):
        """加载CCMBA模糊数据"""
        ccmba_data = self.get_ccmba_data(image_name, category, is_real)
        
        if ccmba_data is None:
            return None, None, None
        
        try:
            blurred_image = load_image_safely(ccmba_data['blurred_image'])
            if blurred_image is None:
                return None, None, None
            
            try:
                blur_mask = np.array(Image.open(ccmba_data['blur_mask'])) / 255.0
            except Exception as e:
                print(f"Error loading blur mask for {category}_{image_name}: {e}")
                blur_mask = None
            
            try:
                with open(ccmba_data['metadata'], 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata for {category}_{image_name}: {e}")
                metadata = {}
            
            return blurred_image, blur_mask, metadata
            
        except Exception as e:
            print(f"Error loading CCMBA data for {category}_{image_name}: {e}")
            return None, None, None

def load_image_safely(image_path, max_retries=3):
    """安全加载图像"""
    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert('RGB')
            _ = image.size
            return image
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to load image after {max_retries} attempts: {image_path}")
                print(f"Error: {str(e)}")
                return None
            else:
                print(f"Attempt {attempt + 1} failed for {image_path}: {str(e)}")
    return None

# ===============================
# 损失函数
# ===============================
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        with autocast():
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

# ===============================
# 辅助函数
# ===============================
def all_reduce_tensor(tensor, world_size):
    """对张量进行all-reduce操作"""
    if dist.is_initialized():
        original_dtype = tensor.dtype
        
        if tensor.dtype in [torch.int, torch.long, torch.int32, torch.int64]:
            tensor = tensor.float()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
                tensor /= world_size
                break
            except Exception as e:
                print(f"Warning: all_reduce failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    print("Error: all_reduce failed after all retries, using local tensor")
                else:
                    time.sleep(0.1 * (attempt + 1))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    return tensor

def calculate_class_weights(train_dataset):
    """计算类别权重"""
    labels = [item[1] for item in train_dataset.data]
    
    real_count = labels.count(0)
    fake_count = labels.count(1)
    total = len(labels)
    
    real_weight = total / (2 * real_count)
    fake_weight = total / (2 * fake_count)
    
    print(f"Dataset statistics:")
    print(f"  Real images: {real_count} ({real_count/total*100:.1f}%)")
    print(f"  Fake images: {fake_count} ({fake_count/total*100:.1f}%)")
    print(f"  Class weights: Real={real_weight:.3f}, Fake={fake_weight:.3f}")
    
    return [real_weight, fake_weight]

def get_memory_usage():
    """获取内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {gpu_memory:.2f}GB allocated, {gpu_reserved:.2f}GB reserved"
    else:
        import psutil
        cpu_memory = psutil.virtual_memory().used / 1024**3
        return f"CPU: {cpu_memory:.2f}GB used"

def clear_memory():
    """清理内存"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ===============================
# 数据集类
# ===============================
class AdapterDataset(Dataset):
    """训练数据集 - 支持Blur增强"""
    
    def __init__(self, root_folder=None, real_folder=None, fake_folder=None, transform=None, max_samples_per_class=None,
                 blur_prob=0.0, blur_strength_range=(0.1, 0.5), blur_mode="global", 
                 mixed_mode_ratio=0.5, ccmba_data_dir=None, enable_strong_aug=False):
        self.transform = transform
        self.blur_prob = blur_prob
        self.blur_strength_range = blur_strength_range
        self.blur_mode = blur_mode
        self.mixed_mode_ratio = mixed_mode_ratio
        self.enable_strong_aug = enable_strong_aug
        
        self.ccmba_loader = None
        if blur_mode in ["ccmba", "mixed"] and ccmba_data_dir:
            self.ccmba_loader = CCMBADataLoader(ccmba_data_dir)
        
        self.data = []
        
        if root_folder and os.path.exists(root_folder):
            extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            root_path = Path(root_folder)
            real_count = 0
            fake_count = 0
            
            for category_folder in root_path.iterdir():
                if category_folder.is_dir():
                    category_name = category_folder.name
                    
                    real_subfolder = category_folder / "0_real"
                    fake_subfolder = category_folder / "1_fake"
                    
                    if real_subfolder.exists():
                        for img_file in real_subfolder.rglob('*'):
                            if img_file.is_file() and img_file.suffix.lower() in extensions:
                                self.data.append((img_file, 0, category_name))
                                real_count += 1
                    
                    if fake_subfolder.exists():
                        for img_file in fake_subfolder.rglob('*'):
                            if img_file.is_file() and img_file.suffix.lower() in extensions:
                                self.data.append((img_file, 1, category_name))
                                fake_count += 1
            
            print(f"Dataset loaded from root {root_folder}: {real_count} real, {fake_count} fake images")
        
        elif real_folder and fake_folder:
            self.real_images = self._get_image_files(real_folder)
            self.fake_images = self._get_image_files(fake_folder)
            for img_path in self.real_images:
                self.data.append((img_path, 0, 'test'))
            for img_path in self.fake_images:
                self.data.append((img_path, 1, 'test'))
            print(f"Dataset loaded: {len(self.real_images)} real, {len(self.fake_images)} fake images")
        
        else:
            raise ValueError("Must provide either root_folder or both real_folder and fake_folder")
        
        if max_samples_per_class:
            self.data = self.data[:max_samples_per_class * 2]
        
        print(f"Blur mode: {self.blur_mode}, Blur prob: {self.blur_prob}")
        if self.ccmba_loader:
            print("✓ CCMBA data loader initialized")
    
    def _get_image_files(self, folder_path):
        """获取图像文件列表"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Warning: Folder not found at {folder_path}")
            return []
        
        image_files = []
        for img_file in folder.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in extensions:
                image_files.append(img_file)
        
        return sorted(image_files)
    
    def apply_blur_augmentation(self, tensor_image, image_name, category, is_real):
        """应用模糊增强"""
        original_device = tensor_image.device
        
        if random.random() > self.blur_prob:
            return tensor_image, {'mode': 'no_blur', 'total_time': 0}
        
        blur_start_time = time.time()
        
        if self.blur_mode == "global":
            blur_strength = random.uniform(*self.blur_strength_range)
            if tensor_image.device != original_device:
                tensor_image = tensor_image.to(original_device)
            blurred_tensor = apply_motion_blur_batch_efficient(tensor_image.unsqueeze(0), blur_strength).squeeze(0)
            blurred_tensor = blurred_tensor.to(original_device)
            blur_info = {
                'mode': 'global',
                'blur_strength': blur_strength,
                'total_time': time.time() - blur_start_time
            }
        elif self.blur_mode == "ccmba":
            if self.ccmba_loader is None:
                blur_strength = random.uniform(*self.blur_strength_range)
                blurred_tensor = apply_motion_blur_batch_efficient(tensor_image.unsqueeze(0), blur_strength).squeeze(0)
                blurred_tensor = blurred_tensor.to(original_device)
                blur_info = {
                    'mode': 'ccmba_fallback_global',
                    'blur_strength': blur_strength,
                    'total_time': time.time() - blur_start_time
                }
            else:
                ccmba_data_start = time.time()
                blurred_pil, blur_mask, metadata = self.ccmba_loader.load_ccmba_blur_data(image_name, category, is_real)
                ccmba_data_time = time.time() - ccmba_data_start
                
                if blurred_pil is not None:
                    ccmba_transform_start = time.time()
                    blurred_tensor = self.transform(blurred_pil)
                    blurred_tensor = blurred_tensor.to(original_device)
                    ccmba_transform_time = time.time() - ccmba_transform_start
                    
                    blur_info = {
                        'mode': 'ccmba',
                        'ccmba_data_load_time': ccmba_data_time,
                        'ccmba_transform_time': ccmba_transform_time,
                        'blur_mask_ratio': np.sum(blur_mask) / blur_mask.size if blur_mask is not None else 0,
                        'total_time': time.time() - blur_start_time
                    }
                else:
                    blur_strength = random.uniform(*self.blur_strength_range)
                    blurred_tensor = apply_motion_blur_batch_efficient(tensor_image.unsqueeze(0), blur_strength).squeeze(0)
                    blurred_tensor = blurred_tensor.to(original_device)
                    blur_info = {
                        'mode': 'ccmba_fallback_global',
                        'blur_strength': blur_strength,
                        'ccmba_data_load_time': ccmba_data_time,
                        'total_time': time.time() - blur_start_time
                    }
        elif self.blur_mode == "mixed":
            blurred_tensor, blur_info = apply_mixed_blur_enhanced(
                tensor_image, self.ccmba_loader, image_name, category, is_real,
                self.blur_strength_range, self.mixed_mode_ratio
            )
            blurred_tensor = blurred_tensor.to(original_device)
            blur_info['total_time'] = time.time() - blur_start_time
        else:
            blurred_tensor = tensor_image
            blur_info = {'mode': 'unknown', 'total_time': time.time() - blur_start_time}
        
        return blurred_tensor, blur_info
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                current_idx = (idx + retry_count) % len(self.data)
                img_path, label, category = self.data[current_idx]
                
                image = Image.open(img_path).convert('RGB')
                width, height = image.size
                if width == 0 or height == 0:
                    raise ValueError(f"Invalid image dimensions: {width}x{height}")
                
                np.array(image)
                
                tensor_image = self.transform(image)
                image_name = img_path.stem
                
                return tensor_image, label, image_name, category
                
            except Exception as e:
                add_corrupted_image(self.data[current_idx][0], str(e))
                print(f"Warning: Skipping corrupted image - {self.data[current_idx][0]}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    raise RuntimeError(f"Too many consecutive corrupted images starting from index {idx}")
        
        raise RuntimeError("Unexpected error in __getitem__")

# ===============================
# 训练函数
# ===============================
def train_teacher(model, train_loader, optimizer, scheduler, scaler, rank, world_size, device, args):
    """Teacher训练 - 使用Blur增强数据"""
    if rank == 0:
        print(f"Starting Teacher training with blur augmentation...")
    
    # 计算类别权重
    if rank == 0:
        class_weights = calculate_class_weights(train_loader.dataset)
    else:
        class_weights = None
    
    # 广播类别权重
    if dist.is_initialized() and world_size > 1:
        if rank == 0:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            class_weights_tensor = torch.zeros(2, dtype=torch.float32, device=device)
        
        dist.broadcast(class_weights_tensor, src=0)
        class_weights = class_weights_tensor.cpu().tolist()
    
    # 创建损失函数
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    history = {'train_loss': [], 'train_acc': []}
    best_acc = 0.0
    
    # 获取实际模型
    actual_model = model.module if hasattr(model, 'module') else model
    
    model_dir = f"checkpoints/teacher/{DATASET_NAME}"
    
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"{'='*60}")
            print(f"TEACHER TRAINING - EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 处理batch数据
            if len(batch_data) == 4:
                images, labels, image_names, categories = batch_data
            else:
                images, labels = batch_data[:2]
                image_names = [f"img_{i}" for i in range(len(images))]
                categories = ["train"] * len(images)
            
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 应用模糊增强
            blurred_imgs_list = []
            for i, (img_tensor, img_name, label, category) in enumerate(zip(images, image_names, labels, categories)):
                is_real = (label.item() == 0)
                
                if img_tensor.device != device:
                    img_tensor = img_tensor.to(device, non_blocking=True)
                
                # 应用模糊
                if hasattr(train_loader.dataset, 'apply_blur_augmentation'):
                    blurred_tensor, _ = train_loader.dataset.apply_blur_augmentation(
                        img_tensor, img_name, category, is_real
                    )
                else:
                    blurred_tensor = img_tensor
                
                if blurred_tensor.device != device:
                    blurred_tensor = blurred_tensor.to(device, non_blocking=True)
                
                blurred_imgs_list.append(blurred_tensor)
            
            blurred_imgs_list = [t.to(device, non_blocking=True) if t.device != device else t 
                               for t in blurred_imgs_list]
            
            blurred_imgs = torch.stack(blurred_imgs_list)
            
            if blurred_imgs.device != device:
                blurred_imgs = blurred_imgs.to(device, non_blocking=True)
            
            # 前向传播
            with autocast():
                logits = actual_model(blurred_imgs)
                loss = criterion(logits, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if rank == 0 and (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * train_correct / train_total
                avg_loss = train_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / len(train_loader) * 100
                
                print(f"[Epoch {epoch+1}] Batch {batch_idx+1:4d}/{len(train_loader)} "
                      f"({progress:5.1f}%) | Loss: {loss.item():.4f} "
                      f"| Avg Loss: {avg_loss:.4f} | Acc: {current_acc:.2f}%")
        
        # 收集统计信息
        if world_size > 1:
            train_loss_tensor = torch.tensor(train_loss, device=device, dtype=torch.float32)
            train_correct_tensor = torch.tensor(train_correct, device=device, dtype=torch.float32)
            train_total_tensor = torch.tensor(train_total, device=device, dtype=torch.float32)
            
            all_reduce_tensor(train_loss_tensor, world_size)
            all_reduce_tensor(train_correct_tensor, world_size)
            all_reduce_tensor(train_total_tensor, world_size)
            
            train_loss = train_loss_tensor.item() / len(train_loader)
            train_acc = 100 * train_correct_tensor.item() / train_total_tensor.item()
        else:
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        if rank == 0:
            print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        scheduler.step()
        
        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            
            if rank == 0:
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, '0.5blur_best_teacher_blur_model.pth')
                
                model_state_dict = actual_model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_acc,
                    'history': history,
                    'config': {
                        'projection_dim': PROJECTION_DIM,
                        'world_size': world_size,
                        'blur_mode': BLUR_MODE,
                        'blur_prob': BLUR_PROB
                    }
                }, model_path)
                
                print(f"✓ New best model saved! Acc: {train_acc:.2f}%")
        
        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    
    return history

# 移除了 evaluate_model 函数

# ===============================
# 主训练函数
# ===============================
def main_distributed(rank, local_rank, world_size, args):
    """分布式训练主函数"""
    start_time = time.time()
    
    try:
        print(f"[Rank {rank}] Initializing distributed training...")
        setup_distributed(rank, world_size)
        setup_logging(rank)
        
        # [修改点] 使用 local_rank 设置设备，支持多机多卡
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.empty_cache()
            
            if rank == 0:
                print(f"Using CUDA devices for distributed training on {world_size} GPUs")
        else:
            device = torch.device('cpu')
            if rank == 0:
                print(f"Using CPU for distributed training")
        
        print(f"[Rank {rank}] Device set to: {device}")
        
        # 验证模型路径
        if rank == 0:
            print(f"{'='*60}")
            print("MODEL VERIFICATION")
            print(f"{'='*60}")
            print(f"DinoV3 model path: {DINOV3_MODEL_ID}")
            
            if not os.path.exists(DINOV3_MODEL_ID):
                raise FileNotFoundError(f"DinoV3 model directory not found: {DINOV3_MODEL_ID}")
            
            print("✓ DinoV3 model path verified")
        
        if rank == 0:
            print(f"{'='*60}")
            print("TEACHER TRAINING WITH BLUR AUGMENTATION")
            print("- Using DinoV3 backbone (Frozen)")
            print("- Blur/CCMBA data augmentation")
            print("- Full autocast FP16 training")
            print("- No Test/Evaluation Phase")
            print(f"{'='*60}")
        
        # 创建输出目录
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 创建数据变换
        train_transform = get_train_transforms()
        
        if rank == 0:
            print("Creating datasets...")
        
        # 训练数据集（带模糊增强）
        train_dataset = AdapterDataset(
            root_folder=TRAIN_ROOT_FOLDER,
            transform=train_transform,
            max_samples_per_class=None,
            blur_prob=BLUR_PROB,
            blur_strength_range=BLUR_STRENGTH_RANGE,
            blur_mode=BLUR_MODE,
            mixed_mode_ratio=MIXED_MODE_RATIO,
            ccmba_data_dir=CCMBA_DATA_DIR,
            enable_strong_aug=False
        )
        
        # [修改点] 移除了 Test Dataset 的初始化
        
        if rank == 0:
            print(f"Dataset sizes - Train: {len(train_dataset)}")
        
        # 创建模型
        if rank == 0:
            print("Creating model...")
            print(f"Memory before model loading: {get_memory_usage()}")
        
        try:
            model = ImprovedDinoV3Adapter(
                model_path=DINOV3_MODEL_ID,
                num_classes=2,
                projection_dim=PROJECTION_DIM,
                adapter_layers=3,
                dropout_rate=0.1,
                device=device
            )
            
            if rank == 0:
                print(f"Memory after model loading: {get_memory_usage()}")
                print("✓ Model created successfully")
            
            clear_memory()
            
        except Exception as e:
            print(f"[Rank {rank}] Error creating model: {e}")
            cleanup_distributed()
            return
        
        # 包装为DDP模型
        model = DDP(
            model, 
            device_ids=[local_rank] if device.type == 'cuda' else None, # 使用 local_rank
            output_device=local_rank if device.type == 'cuda' else None, # 使用 local_rank
            find_unused_parameters=True,
        )
        
        if rank == 0:
            print("Starting training process...")
        
        if world_size > 1:
            dist.barrier()
        
        # 创建GradScaler
        scaler = GradScaler()
        
        # 训练数据加载器
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, 
            shuffle=True, drop_last=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            sampler=train_sampler, num_workers=NUM_WORKERS, 
            pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=2
        )
        
        # 优化器
        params = list(model.module.projection.parameters()) + \
                 list(model.module.classifier.parameters())
        
        optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 训练
        history = train_teacher(
            model, train_loader,
            optimizer, scheduler, scaler, rank, world_size, device, args
        )
        
        # [修改点] 移除了 Evaluation 阶段
        
        # 保存训练历史
        if rank == 0:
            training_info = {
                'history': history,
                'model_config': {
                    'dinov3_model_id': DINOV3_MODEL_ID,
                    'projection_dim': PROJECTION_DIM
                },
                'training_config': {
                    'world_size': world_size,
                    'batch_size': args.batch_size * world_size,
                    'epochs': args.epochs,
                    'blur_mode': BLUR_MODE,
                    'blur_prob': BLUR_PROB,
                    'blur_strength_range': BLUR_STRENGTH_RANGE
                }
            }
            
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print("Training completed!")
            print(f"Results saved to: {args.output_dir}")
            
            # 保存最终模型
            print("Saving final model checkpoint...")
            final_model_path = os.path.join(args.output_dir, "0.5blur_final_teacher_blur_model.pth")
            torch.save(model.module.state_dict(), final_model_path)
            print(f"Final model saved to: {final_model_path}")
            
            save_corrupted_images_report(args.output_dir, rank)
        
    except Exception as e:
        print(f"[Rank {rank}] Critical error in main_distributed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        return
    
    except KeyboardInterrupt:
        print(f"[Rank {rank}] Training interrupted by user")
        cleanup_distributed()
        return
    
    finally:
        cleanup_distributed()

def run_distributed_training(args):
    """使用torch.distributed.launch启动分布式训练"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    # [修改点] 传入 local_rank
    main_distributed(rank, local_rank, world_size, args)

# ===============================
# 命令行接口
# ===============================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Teacher Training with Blur Augmentation')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--output-dir', type=str, default="teacher_blur_training",
                        help='Output directory for models and logs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        if 'RANK' in os.environ:
            run_distributed_training(args)
        else:
            print("Running in single process mode")
            # [修改点] 单机模式 local_rank=0
            main_distributed(0, 0, 1, args)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        if dist.is_initialized():
            cleanup_distributed()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            cleanup_distributed()