"""
分布式Teacher-Student训练脚本 
- 更好的DinoV3 Adapter
- 全面autocast半精度训练
- 优化的训练流程
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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

# [保持分布式配置函数不变]
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
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
    
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
#TRAIN_REAL_FOLDER = "/home/work/xueyunqi/11ar/fake"
#TRAIN_FAKE_FOLDER = "/home/work/xueyunqi/11ar/fake"
TEST_REAL_FOLDER = "/home/work/xueyunqi/11ar/fake"
TEST_FAKE_FOLDER = "/home/work/xueyunqi/11ar/fake"

# 训练配置
TEACHER_LEARNING_RATE = 1e-4
STUDENT_LEARNING_RATE = 5e-5
PROJECTION_DIM = 512
TEMPERATURE = 0.07
NUM_WORKERS = 4

# Loss权重
ALPHA_DISTILL = 1.0
ALPHA_CLS = 1.0
ALPHA_FEATURE = 0.5

# 数据集名称
DATASET_NAME = "0917_dinov3_teacher_student"

# Motion blur配置
BLUR_PROB = 0.2
BLUR_STRENGTH_RANGE = (0.1, 0.3)
BLUR_MODE = "mixed"
MIXED_MODE_RATIO = 0.5
CCMBA_DATA_DIR = "/home/work/xueyunqi/11ar_datasets/progan_ccmba_train"


# ===============================
# Motion Blur工具函数
# ===============================
def apply_motion_blur(image_pil, strength):
    """应用运动模糊"""
    # 转换为OpenCV格式
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 计算kernel_size
    kernel_size = int(5 + (strength - 0.05) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, min(kernel_size, 31))
    
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / np.sum(kernel)
    
    # 应用模糊
    blurred_cv = cv2.filter2D(image_cv, -1, kernel)
    
    # 转换回PIL格式
    blurred_pil = Image.fromarray(cv2.cvtColor(blurred_cv, cv2.COLOR_BGR2RGB))
    return blurred_pil

def apply_motion_blur_batch_efficient(images, strength):
    """
    高效的批量运动模糊处理，直接在tensor上操作避免PIL转换
    输入: torch.Tensor (B, C, H, W)
    输出: torch.Tensor (B, C, H, W)
    """
    if not isinstance(images, torch.Tensor):
        return apply_motion_blur_batch(images, strength)  # fallback
    
    device = images.device
    batch_size, channels, height, width = images.shape
    
    # 计算kernel_size (与原函数一致)
    kernel_size = int(5 + (strength - 0.05) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, min(kernel_size, 31))
    
    # 创建运动模糊核 (水平方向)
    kernel = torch.zeros(kernel_size, kernel_size, device=device)
    kernel[kernel_size//2, :] = 1.0
    kernel = kernel / kernel.sum()
    
    # 扩展kernel为4D: (out_channels, in_channels, h, w)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
    
    # 使用depthwise convolution应用模糊 (每个通道独立)
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
    
    return blurred

def apply_mixed_blur_enhanced(original_tensor, ccmba_loader, image_name, category, is_real, blur_strength_range, mixed_ratio=0.5):
    """
    应用混合模糊：结合全局模糊和CCMBA部分模糊（增强版，兼容tstrain数据变换）
    
    Args:
        original_tensor: 原始图像tensor (C, H, W)
        ccmba_loader: CCMBA数据加载器
        image_name: 图像文件名
        category: 图像类别
        is_real: 是否为真实图像
        blur_strength_range: 全局模糊强度范围
        mixed_ratio: CCMBA使用比例
    
    Returns:
        blurred_tensor: 模糊后的图像tensor
        blur_info: 模糊信息字典
    """
    device = original_tensor.device
    blur_info = {'mode': 'mixed', 'ccmba_used': False, 'global_used': False}
    
    # 根据比例决定使用哪种模糊方式
    if random.random() < mixed_ratio:
        # 尝试使用CCMBA
        ccmba_data_start = time.time()
        blurred_pil, blur_mask, metadata = ccmba_loader.load_ccmba_blur_data(image_name, category, is_real)
        ccmba_data_time = time.time() - ccmba_data_start
        
        if blurred_pil is not None:
            # 成功加载CCMBA数据，使用与tstrain一致的变换
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
    
    # 回退到全局模糊
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
    """
    批量应用运动模糊。
    支持输入为 torch.Tensor (B, C, H, W) 或 PIL.Image 列表。
    返回同样类型的 motion blur 处理结果。
    """
    if isinstance(images, torch.Tensor):
        # 使用高效的tensor版本
        return apply_motion_blur_batch_efficient(images, strength)
    elif isinstance(images, list):
        # PIL列表输入
        return [apply_motion_blur(img, strength) for img in images]
    else:
        # 单张图片
        return apply_motion_blur(images, strength)



# ===============================
# 数据增强
# ===============================
def get_train_transforms():
    """Teacher网络的数据增强"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        
        transforms.RandomCrop((448, 448)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def get_test_transforms():
    """测试时的数据变换"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        
        # transforms.CenterCrop(224, 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# ===============================
# L2Norm层, nn.Module
# ===============================
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class ImprovedDinoV3Adapter(nn.Module):
    """改进的DinoV3模型适配器 - 集成特征提取、投影和分类"""
    
    def __init__(self, model_path, num_classes=2, projection_dim=512, 
                 adapter_layers=2, dropout_rate=0.1, device="cuda"):
        super().__init__()
        
        self.device = device
        self.projection_dim = projection_dim
        
        print(f"Loading DinoV3 model: {model_path}")
        
        # 加载DinoV3 backbone
        self.backbone = self._load_dinov3_backbone(model_path, device)
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 4096)
        
        print(f"DinoV3 feature dimension: {self.hidden_size}")
        
        # 特征投影器 - 多层设计
        projection_layers = []
        current_dim = self.hidden_size
        
        for i in range(adapter_layers):
            if i == adapter_layers - 1:  # 最后一层
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
        
        # 移动到指定设备
        self.projection = self.projection.to(device)
        self.classifier = self.classifier.to(device)
        
        print(f"✓ DinoV3 Adapter initialized:")
        print(f"  - Backbone: {self.hidden_size}D")
        print(f"  - Projection: {projection_dim}D ({adapter_layers} layers)")
        print(f"  - Classes: {num_classes}")
        
        # 初始化权重
        self._initialize_weights()
    
    def _load_dinov3_backbone(self, model_path, device):
        """加载DinoV3 backbone"""
        try:
            # 加载到CPU后再转移到目标设备
            backbone = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            
            # 移动到目标设备并设置为eval模式
            backbone = backbone.to(device).eval()
            
            # 冻结backbone参数
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
        """提取DinoV3特征 - 支持autocast"""
        with torch.no_grad():
            with autocast():
                outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
                
                # 提取CLS token特征
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0]  # CLS token
                
                return features.float()  # 确保输出为float32
    
    def forward(self, pixel_values, return_features=False):
        """前向传播 - 支持autocast"""
        # 特征提取（frozen backbone）
        raw_features = self.extract_features(pixel_values)
        
        # 投影和分类（trainable layers）
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
    
    def freeze_backbone(self):
        """冻结backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_projection(self):
        """解冻投影层"""
        for param in self.projection.parameters():
            param.requires_grad = True
    
    def unfreeze_classifier(self):
        """解冻分类器"""
        for param in self.classifier.parameters():
            param.requires_grad = True


# CCMBA数据加载工具
# ===============================
class CCMBADataLoader:
    """CCMBA预处理数据加载器 - 支持多类别（自动发现 category 如 airplane, bird 等）"""
    
    def __init__(self, ccmba_root_dir, categories=None):
        self.ccmba_root_dir = Path(ccmba_root_dir)
        self.categories = categories
        
        # 构建映射字典：原图文件名 -> CCMBA数据路径（key: f"{category}_{image_name}"）
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
        """构建所有类别的映射（自动发现或使用指定 categories）"""
        if self.categories is None:
            # 自动发现所有子目录（airplane, bird 等）
            self.categories = [d.name for d in self.ccmba_root_dir.iterdir() 
                              if d.is_dir() and (d / "nature").exists() or (d / "ai").exists()]
            print(f"Auto-discovered CCMBA categories: {self.categories}")
        
        for category in self.categories:
            category_dir = self.ccmba_root_dir / category
            if not category_dir.exists():
                print(f"Warning: Category directory not found: {category_dir}")
                continue
            
            real_dir = category_dir / "nature"  # real
            fake_dir = category_dir / "ai"      # fake
            
            if real_dir.exists():
                category_real_mapping = self._build_mapping(real_dir)
                # 添加类别前缀避免文件名冲突：key = f"{category}_{base_name}"
                for key, value in category_real_mapping.items():
                    self.real_mapping[f"{category}_{key}"] = value
                print(f"  - {category}/nature (real): {len(category_real_mapping)} entries")
            
            if fake_dir.exists():
                category_fake_mapping = self._build_mapping(fake_dir)
                for key, value in category_fake_mapping.items():
                    self.fake_mapping[f"{category}_{key}"] = value
                print(f"  - {category}/ai (fake): {len(category_fake_mapping)} entries")
    
    def _build_mapping(self, class_dir):
        """构建原图文件名到CCMBA数据的映射（从 blurred_images/*.jpg）"""
        mapping = {}
        
        if not class_dir.exists():
            return mapping
        
        blurred_images_dir = class_dir / "blurred_images"
        blur_masks_dir = class_dir / "blur_masks"
        metadata_dir = class_dir / "metadata"
        
        if not all([blurred_images_dir.exists(), blur_masks_dir.exists(), metadata_dir.exists()]):
            print(f"Warning: Incomplete CCMBA structure in {class_dir}")
            return mapping
        
        # 获取所有模糊图像文件
        for blur_img_path in blurred_images_dir.glob("*.jpg"):
            base_name = blur_img_path.stem  # 文件名 stem，用于 key
            
            # 构建对应的文件路径
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
        """
        获取指定图像的CCMBA数据
        
        Args:
            image_name: 图像文件名（不含扩展名）
            category: 类别名称
            is_real: 是否为真实图像
            
        Returns:
            dict: 包含模糊图像、掩码和元数据的字典，如果未找到返回None
        """
        mapping = self.real_mapping if is_real else self.fake_mapping
        key = f"{category}_{image_name}"
        
        if key in mapping:
            return mapping[key]
        else:
            return None
    
    def load_ccmba_blur_data(self, image_name, category, is_real=True):
        """加载CCMBA模糊数据，带错误处理"""
        ccmba_data = self.get_ccmba_data(image_name, category, is_real)
        
        if ccmba_data is None:
            return None, None, None
        
        try:
            # 安全加载模糊图像
            blurred_image = load_image_safely(ccmba_data['blurred_image'])
            if blurred_image is None:
                return None, None, None
            
            # 加载模糊掩码
            try:
                blur_mask = np.array(Image.open(ccmba_data['blur_mask'])) / 255.0
            except Exception as e:
                print(f"Error loading blur mask for {category}_{image_name}: {e}")
                blur_mask = None
            
            # 加载元数据
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
    """安全加载图像，带重试机制"""
    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert('RGB')
            # 尝试获取图像基本信息以确保能正常读取
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
# Teacher-Student 网络架构
# ===============================
class TeacherStudentNetwork(nn.Module):
    """Teacher-Student网络 - 使用DinoV3 Adapter"""
    
    def __init__(self, dinov3_model_path, num_classes=2, projection_dim=512, device="cuda"):
        super().__init__()
        
        self.device = device
        
        print(f"{'='*60}")
        print("INITIALIZING TEACHER-STUDENT NETWORK")
        print(f"{'='*60}")
        
        # Teacher模型 - DinoV3 Adapter
        print("Creating Teacher model (DinoV3 + Adapter)...")
        self.teacher = ImprovedDinoV3Adapter(
            model_path=dinov3_model_path,
            num_classes=num_classes,
            projection_dim=projection_dim,
            adapter_layers=3,
            dropout_rate=0.1,
            device=device
        )
        
        # Student模型 - 共享backbone但独立的投影和分类层
        print("Creating Student model (shared backbone + independent head)...")
        self.student_projection = nn.Sequential(
            nn.Linear(self.teacher.hidden_size, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)
        
        self.student_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, num_classes)
        ).to(device)
        
        # 初始化Student权重
        self._initialize_student_weights()
        
        print(f"✓ Teacher-Student network initialized")
        print(f"  - Shared DinoV3 backbone: {self.teacher.hidden_size}D")
        print(f"  - Projection dimension: {projection_dim}D")
        print(f"  - Number of classes: {num_classes}")
    
    def _initialize_student_weights(self):
        """初始化Student层权重"""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.student_projection.apply(init_layer)
        self.student_classifier.apply(init_layer)
    
    def forward_teacher(self, pixel_values):
        """Teacher前向传播"""
        outputs = self.teacher(pixel_values, return_features=True)
        return outputs['projected_features'], outputs['logits']
    
    def forward_student(self, pixel_values):
        """Student前向传播 - 使用Teacher的backbone"""
        # 使用Teacher的backbone提取特征
        raw_features = self.teacher.extract_features(pixel_values)
        
        # Student独有的投影和分类
        with autocast():
            projected_features = self.student_projection(raw_features)
            logits = self.student_classifier(projected_features)
        
        return projected_features, logits
    
    def freeze_teacher(self):
        """冻结Teacher的所有参数"""
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def unfreeze_teacher_head(self):
        """解冻Teacher的投影和分类层"""
        self.teacher.unfreeze_projection()
        self.teacher.unfreeze_classifier()
    
    def unfreeze_student(self):
        """解冻Student的投影和分类层"""
        for param in self.student_projection.parameters():
            param.requires_grad = True
        for param in self.student_classifier.parameters():
            param.requires_grad = True

# ===============================
# 损失函数 - 支持autocast
# ===============================
class ImprovedTeacherStudentLoss(nn.Module):
    """改进的Teacher-Student损失函数 - 支持autocast"""
    
    def __init__(self, temperature=0.07, alpha_distill=1.0, alpha_cls=1.0, 
                 alpha_feature=0.5, alpha_simclr=0.3):  # 移除focal loss参数
        super().__init__()
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_cls = alpha_cls
        self.alpha_feature = alpha_feature
        self.alpha_simclr = alpha_simclr
        
        # 使用标准CrossEntropyLoss替代FocalLoss
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.simclr_loss = SimCLRLoss(temperature=temperature)
    
    def distillation_loss(self, student_logits, teacher_logits):
        """知识蒸馏损失"""
        with autocast():
            # Softmax with temperature
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
            
            # KL散度损失
            kl_loss = self.kl_loss(student_log_probs, teacher_probs)
            return kl_loss * (self.temperature ** 2)
    
    def feature_alignment_loss(self, student_features, teacher_features):
        """特征对齐损失"""
        with autocast():
            # L2归一化
            student_norm = F.normalize(student_features, p=2, dim=1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=1)
            
            # 余弦相似度损失
            cosine_sim = (student_norm * teacher_norm).sum(dim=1)
            loss = 1.0 - cosine_sim.mean()
            
            return loss
    
    def forward(self, student_features, student_logits, teacher_features=None, 
                teacher_logits=None, labels=None, mode="student", student_features_aug=None):
        """计算总损失"""
        losses = {}
        
        with autocast():
            if mode == "teacher":
                cls_loss = self.ce_loss(student_logits, labels)
                
                losses.update({
                    'total_loss': cls_loss,
                    'cls_loss': cls_loss,
                    'distill_loss': torch.tensor(0.0, device=cls_loss.device),
                    'feature_loss': torch.tensor(0.0, device=cls_loss.device)
                })
            
            else:  # student mode
                # 分类损失 - 使用CrossEntropyLoss
                cls_loss = self.ce_loss(student_logits, labels)
                
                # 蒸馏损失
                distill_loss = self.distillation_loss(student_logits, teacher_logits)
                
                # 特征对齐损失
                feature_loss = self.feature_alignment_loss(student_features, teacher_features)
                
                # SimCLR对比学习损失
                simclr_loss = torch.tensor(0.0, device=cls_loss.device)
                if student_features_aug is not None:
                    combined_features = torch.cat([student_features, student_features_aug], dim=0)
                    simclr_loss = self.simclr_loss(combined_features)
                
                # 总损失
                total_loss = (self.alpha_cls * cls_loss + 
                            self.alpha_distill * distill_loss + 
                            self.alpha_feature * feature_loss +
                            self.alpha_simclr * simclr_loss)
                
                losses.update({
                    'total_loss': total_loss,
                    'cls_loss': cls_loss,
                    'distill_loss': distill_loss,
                    'feature_loss': feature_loss,
                    'simclr_loss': simclr_loss
                })
        
        return losses

def train_teacher_phase(model, train_loader, optimizer, scheduler, scaler, rank, world_size, device, args):
    """Teacher训练阶段 - 使用DinoV3 backbone + 数据增强"""
    if rank == 0:
        print(f"Starting Teacher training phase...")
    
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
    criterion = ImprovedTeacherStudentLoss(
        temperature=TEMPERATURE,
        alpha_distill=ALPHA_DISTILL,
        alpha_cls=ALPHA_CLS,
        alpha_feature=ALPHA_FEATURE
    )
    
    history = {'train_loss': [], 'train_acc': []}
    best_acc = 0.0
    
    # 修复：正确访问DDP包装的模型方法
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 设置训练参数
    actual_model.freeze_teacher()  # 冻结Teacher
    actual_model.unfreeze_teacher_head()  # 解冻Teacher的分类头
    
    teacher_model_dir = f"checkpoints/teacher/{DATASET_NAME}"
    
    for epoch in range(args.teacher_epochs):
        if rank == 0:
            print(f"{'='*60}")
            print(f"TEACHER TRAINING - EPOCH {epoch+1}/{args.teacher_epochs}")
            print(f"{'='*60}")
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 处理不同长度的batch数据
            if len(batch_data) == 4:
                images, labels, _, _ = batch_data
            else:
                images, labels = batch_data[:2]
            
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Teacher前向传播 - 使用autocast
            with autocast():
                teacher_features, teacher_logits = actual_model.forward_teacher(images)
                
                # 计算损失
                losses = criterion(
                    student_features=teacher_features,  # 传入teacher特征
                    student_logits=teacher_logits,      # 传入teacher logits
                    labels=labels, 
                    mode="teacher"
                )
            
            # 反向传播
            scaler.scale(losses['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_loss += losses['total_loss'].item()
            _, predicted = torch.max(teacher_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if rank == 0 and (batch_idx + 1) % 1 == 0:
                current_acc = 100.0 * train_correct / train_total
                avg_loss = train_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / len(train_loader) * 100
                
                print(f"[Teacher Epoch {epoch+1}] Batch {batch_idx+1:4d}/{len(train_loader)} "
                      f"({progress:5.1f}%) | Loss: {losses['total_loss'].item():.4f} "
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
                os.makedirs(teacher_model_dir, exist_ok=True)
                teacher_model_path = os.path.join(teacher_model_dir, 'nofocal_best_teacher_model.pth')
                
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
                        'temperature': TEMPERATURE,
                        'world_size': world_size
                    }
                }, teacher_model_path)
                
                print(f"✓ New best Teacher model saved! Acc: {train_acc:.2f}%")
        
        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        print(f"Teacher training completed! Best accuracy: {best_acc:.2f}%")
    
    return history

def train_student_phase(model, train_loader, optimizer, scheduler, scaler, rank, world_size, device, args):
    """Student训练阶段 - 使用模糊增强数据"""
    if rank == 0:
        print(f"Starting Student training phase with blur augmentation...")
    
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
    criterion = ImprovedTeacherStudentLoss(
        temperature=TEMPERATURE,
        alpha_distill=ALPHA_DISTILL,
        alpha_cls=ALPHA_CLS,
        alpha_feature=ALPHA_FEATURE,
        alpha_simclr=0.3
    )
    
    history = {
        'train_total_loss': [], 'train_cls_loss': [], 
        'train_distill_loss': [], 'train_feature_loss': [], 'train_acc': []
    }
    best_acc = 0.0
    
    # 修复：正确访问DDP包装的模型方法
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 设置训练参数
    actual_model.freeze_teacher()  # 冻结Teacher（作为固定的特征提取器）
    actual_model.unfreeze_student()  # 解冻Student
    
    student_model_dir = f"checkpoints/student/{DATASET_NAME}"
    
    for epoch in range(args.student_epochs):
        if rank == 0:
            print(f"{'='*60}")
            print(f"STUDENT TRAINING - EPOCH {epoch+1}/{args.student_epochs}")
            print(f"{'='*60}")
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_losses = {'total': 0, 'cls': 0, 'distill': 0, 'feature': 0}
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
            
            # Teacher前向传播（frozen，生成target）
            with torch.no_grad():
                with autocast():
                    teacher_features, teacher_logits = actual_model.forward_teacher(images)
            
            # Student数据流：应用模糊增强 - 修复设备一致性问题
            student_imgs_list = []
            for i, (img_tensor, img_name, label, category) in enumerate(zip(images, image_names, labels, categories)):
                is_real = (label.item() == 0)
                
                # 确保输入tensor在正确的设备上
                if img_tensor.device != device:
                    img_tensor = img_tensor.to(device, non_blocking=True)
                
                # 应用模糊增强（如果数据集支持）
                if hasattr(train_loader.dataset, 'apply_blur_augmentation'):
                    blurred_tensor, _ = train_loader.dataset.apply_blur_augmentation(
                        img_tensor, img_name, category, is_real
                    )
                else:
                    blurred_tensor = img_tensor
                
                # 关键修复：确保模糊后的tensor在正确的设备上
                if blurred_tensor.device != device:
                    blurred_tensor = blurred_tensor.to(device, non_blocking=True)
                
                student_imgs_list.append(blurred_tensor)
            
            # 修复：在stack之前再次确保所有tensor都在同一设备上
            student_imgs_list = [t.to(device, non_blocking=True) if t.device != device else t 
                               for t in student_imgs_list]
            
            student_imgs = torch.stack(student_imgs_list)
            
            # 最终确保：如果stack后的tensor不在正确设备，移动它
            if student_imgs.device != device:
                student_imgs = student_imgs.to(device, non_blocking=True)
            
            # Student前向传播
            with autocast():
                student_features, student_logits = actual_model.forward_student(student_imgs)
                
                student_imgs_aug_list = []
                for img_tensor in student_imgs:
                    # 应用额外的模糊增强
                    blur_strength = random.uniform(0.1, 0.3)
                    aug_tensor = apply_motion_blur_batch_efficient(
                        img_tensor.unsqueeze(0), blur_strength
                    ).squeeze(0)
                    student_imgs_aug_list.append(aug_tensor)
                
                student_imgs_aug = torch.stack(student_imgs_aug_list).to(device, non_blocking=True)
                student_features_aug, _ = actual_model.forward_student(student_imgs_aug)
                # 计算损失
                losses = criterion(
                    student_features=student_features,
                    student_logits=student_logits,
                    teacher_features=teacher_features,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    mode="student",
                    student_features_aug=student_features_aug
                )
            
            # 反向传播
            scaler.scale(losses['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_losses['total'] += losses['total_loss'].item()
            train_losses['cls'] += losses['cls_loss'].item()
            train_losses['distill'] += losses['distill_loss'].item()
            train_losses['feature'] += losses['feature_loss'].item()
            train_losses['simclr'] = train_losses.get('simclr', 0) + losses['simclr_loss'].item() 
            
            _, predicted = torch.max(student_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if rank == 0 and (batch_idx + 1) % 1 == 0:
                current_acc = 100.0 * train_correct / train_total
                avg_total_loss = train_losses['total'] / (batch_idx + 1)
                progress = (batch_idx + 1) / len(train_loader) * 100
                
                print(f"[Student Epoch {epoch+1}] Batch {batch_idx+1:4d}/{len(train_loader)} "
                      f"({progress:5.1f}%) | Total: {losses['total_loss'].item():.4f} "
                      f"| Cls: {losses['cls_loss'].item():.4f} "
                      f"| Distill: {losses['distill_loss'].item():.4f} "
                      f"| Feature: {losses['feature_loss'].item():.4f} | Acc: {current_acc:.2f}%"
                      f"| SimCLR: {losses['simclr_loss'].item():.4f} | Acc: {current_acc:.2f}%") 
                      
        
        # 收集统计信息
        if world_size > 1:
            for key in train_losses:
                train_losses[key] = torch.tensor(train_losses[key], device=device, dtype=torch.float32)
                all_reduce_tensor(train_losses[key], world_size)
                train_losses[key] = train_losses[key].item() / len(train_loader)
            
            train_correct_tensor = torch.tensor(train_correct, device=device, dtype=torch.float32)
            train_total_tensor = torch.tensor(train_total, device=device, dtype=torch.float32)
            
            all_reduce_tensor(train_correct_tensor, world_size)
            all_reduce_tensor(train_total_tensor, world_size)
            
            train_acc = 100 * train_correct_tensor.item() / train_total_tensor.item()
        else:
            for key in train_losses:
                train_losses[key] = train_losses[key] / len(train_loader)
            train_acc = 100 * train_correct / train_total
        
        # 记录历史
        history['train_total_loss'].append(train_losses['total'])
        history['train_cls_loss'].append(train_losses['cls'])
        history['train_distill_loss'].append(train_losses['distill'])
        history['train_feature_loss'].append(train_losses['feature'])
        history['train_acc'].append(train_acc)
        
        if rank == 0:
            print(f"Epoch {epoch+1} - Total: {train_losses['total']:.4f}, "
                  f"Cls: {train_losses['cls']:.4f}, Distill: {train_losses['distill']:.4f}, "
                  f"Feature: {train_losses['feature']:.4f}, Acc: {train_acc:.2f}%")
        
        scheduler.step()
        
        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            
            if rank == 0:
                os.makedirs(student_model_dir, exist_ok=True)
                student_model_path = os.path.join(student_model_dir, 'nofocal_best_student_model.pth')
                
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
                        'temperature': TEMPERATURE,
                        'alpha_distill': ALPHA_DISTILL,
                        'alpha_cls': ALPHA_CLS,
                        'alpha_feature': ALPHA_FEATURE,
                        'world_size': world_size
                    }
                }, student_model_path)
                
                print(f"✓ New best Student model saved! Acc: {train_acc:.2f}%")
        
        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        print(f"Student training completed! Best accuracy: {best_acc:.2f}%")
    
    return history

def evaluate_model_distributed(model, test_loader, output_dir, rank, world_size, device):
    """分布式模型评估 - 支持autocast，修复DDP访问问题"""
    if rank == 0:
        print("Starting model evaluation...")
    
    model.eval()
    
    # 修复：正确访问DDP包装的模型方法
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Teacher评估
    teacher_correct = 0
    teacher_total = 0
    teacher_predictions = []
    teacher_true_labels = []
    
    # Student评估  
    student_correct = 0
    student_total = 0
    student_predictions = []
    student_true_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # 处理不同长度的batch数据
            if len(batch_data) >= 2:
                images, labels = batch_data[:2]
            else:
                continue
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Teacher前向传播
            with autocast():
                teacher_features, teacher_logits = actual_model.forward_teacher(images)
                
                # Student前向传播
                student_features, student_logits = actual_model.forward_student(images)
            
            # Teacher统计
            _, teacher_pred = torch.max(teacher_logits.data, 1)
            teacher_total += labels.size(0)
            teacher_correct += (teacher_pred == labels).sum().item()
            
            teacher_predictions.extend(teacher_pred.cpu().numpy())
            teacher_true_labels.extend(labels.cpu().numpy())
            
            # Student统计
            _, student_pred = torch.max(student_logits.data, 1)
            student_total += labels.size(0)
            student_correct += (student_pred == labels).sum().item()
            
            student_predictions.extend(student_pred.cpu().numpy())
            student_true_labels.extend(labels.cpu().numpy())
    
    # 收集所有GPU的结果
    if world_size > 1:
        teacher_correct_tensor = torch.tensor(teacher_correct, device=device, dtype=torch.float32)
        teacher_total_tensor = torch.tensor(teacher_total, device=device, dtype=torch.float32)
        student_correct_tensor = torch.tensor(student_correct, device=device, dtype=torch.float32)
        student_total_tensor = torch.tensor(student_total, device=device, dtype=torch.float32)
        
        all_reduce_tensor(teacher_correct_tensor, world_size)
        all_reduce_tensor(teacher_total_tensor, world_size)
        all_reduce_tensor(student_correct_tensor, world_size)
        all_reduce_tensor(student_total_tensor, world_size)
        
        teacher_acc = teacher_correct_tensor.item() / teacher_total_tensor.item()
        student_acc = student_correct_tensor.item() / student_total_tensor.item()
    else:
        teacher_acc = teacher_correct / teacher_total
        student_acc = student_correct / student_total
    
    if rank == 0:
        print(f"Evaluation Results:")
        print(f"Teacher (DinoV3) Accuracy: {teacher_acc:.4f}")
        print(f"Student (DinoV3+Distill) Accuracy: {student_acc:.4f}")
        
        # 保存详细结果
        results = {
            'teacher': {
                'accuracy': teacher_acc,
                'predictions': teacher_predictions,
                'true_labels': teacher_true_labels
            },
            'student': {
                'accuracy': student_acc,
                'predictions': student_predictions,
                'true_labels': student_true_labels
            }
        }
        
        # 计算更详细的指标
        from sklearn.metrics import classification_report
        
        teacher_report = classification_report(teacher_true_labels, teacher_predictions, 
                                             target_names=['Real', 'Fake'], output_dict=True)
        student_report = classification_report(student_true_labels, student_predictions,
                                             target_names=['Real', 'Fake'], output_dict=True)
        
        results['teacher']['classification_report'] = teacher_report
        results['student']['classification_report'] = student_report
        
        # 保存结果
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Evaluation results saved!")
        return results
    
    return None



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

class SimCLRLoss(nn.Module):
    """SimCLR对比学习损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features):
        """
        Args:
            features: (2*batch_size, feature_dim) - 包含原始和增强样本的特征
        """
        with autocast():
            # L2归一化
            features = F.normalize(features, p=2, dim=1)
            batch_size = features.shape[0] // 2            
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            
            # 创建mask
            mask = torch.eye(batch_size * 2, dtype=torch.bool, device=features.device)
            # 正样本mask
            positive_mask = torch.zeros_like(mask)
            for i in range(batch_size):
                positive_mask[i, i + batch_size] = True
                positive_mask[i + batch_size, i] = True
            
            # 负样本
            negative_mask = ~(mask | positive_mask)
            
            losses = []
            for i in range(batch_size * 2):
                # 正样本相似度
                pos_sim = similarity_matrix[i][positive_mask[i]]    
                # 负样本相似度
                neg_sim = similarity_matrix[i][negative_mask[i]]
                # InfoNCE loss
                logits = torch.cat([pos_sim, neg_sim])
                labels = torch.zeros(len(logits), dtype=torch.long, device=features.device)
                
                loss = F.cross_entropy(logits.unsqueeze(0), labels[:1])
                losses.append(loss)
            
            return torch.stack(losses).mean()



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
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU: {gpu_memory:.2f}GB allocated, {gpu_reserved:.2f}GB reserved"
    else:
        import psutil
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        return f"CPU: {cpu_memory:.2f}GB used"

def clear_memory():
    """清理内存"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class AdapterDataset(Dataset):
    """适配器训练数据集 - 集成所有数据增强功能（支持多类别 + category 记录）"""
    
    def __init__(self, root_folder=None, real_folder=None, fake_folder=None, transform=None, max_samples_per_class=None,
                 blur_prob=0.0, blur_strength_range=(0.1, 0.5), blur_mode="global", 
                 mixed_mode_ratio=0.5, ccmba_data_dir=None, enable_strong_aug=False):
        self.transform = transform
        self.blur_prob = blur_prob
        self.blur_strength_range = blur_strength_range
        self.blur_mode = blur_mode
        self.mixed_mode_ratio = mixed_mode_ratio
        self.enable_strong_aug = enable_strong_aug
        
        # 初始化CCMBA数据加载器
        self.ccmba_loader = None
        if blur_mode in ["ccmba", "mixed"] and ccmba_data_dir:
            self.ccmba_loader = CCMBADataLoader(ccmba_data_dir)  # 传入新路径，categories=None 自动发现
        
        # 强力增强
        if enable_strong_aug:
            self.strong_transform = get_balanced_train_transforms()
        
        # 加载图像路径 - 支持多类别根目录结构，并记录 category
        self.data = []  # 改为 [(img_path, label, category)]
        
        if root_folder and os.path.exists(root_folder):  # 新结构：多类别根目录
            extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            root_path = Path(root_folder)
            real_count = 0
            fake_count = 0
            
            for category_folder in root_path.iterdir():  # 遍历子文件夹如 airplane, bird 等
                if category_folder.is_dir():
                    category_name = category_folder.name  # 记录 category 如 'airplane'
                    
                    real_subfolder = category_folder / "0_real"
                    fake_subfolder = category_folder / "1_fake"
                    
                    # 收集真实图片，并记录 category
                    if real_subfolder.exists():
                        for img_file in real_subfolder.rglob('*'):
                            if img_file.is_file() and img_file.suffix.lower() in extensions:
                                self.data.append((img_file, 0, category_name))  # (path, label=0, category)
                                real_count += 1
                    
                    # 收集虚假图片，并记录 category
                    if fake_subfolder.exists():
                        for img_file in fake_subfolder.rglob('*'):
                            if img_file.is_file() and img_file.suffix.lower() in extensions:
                                self.data.append((img_file, 1, category_name))  # (path, label=1, category)
                                fake_count += 1
            
            print(f"Dataset loaded from root {root_folder}: {real_count} real, {fake_count} fake images (multi-class structure with categories)")
        
        elif real_folder and fake_folder:  # 兼容原有独立文件夹结构（用于测试集）
            # 对于测试集，category 默认 'test' 或 None（不使用 CCMBA）
            self.real_images = self._get_image_files(real_folder)
            self.fake_images = self._get_image_files(fake_folder)
            for img_path in self.real_images:
                self.data.append((img_path, 0, 'test'))  # 默认 category
            for img_path in self.fake_images:
                self.data.append((img_path, 1, 'test'))
            print(f"Dataset loaded: {len(self.real_images)} real, {len(self.fake_images)} fake images (legacy structure)")
        
        else:
            raise ValueError("Must provide either root_folder or both real_folder and fake_folder")
        
        if max_samples_per_class:
            # 简单截取（实际可按 category 平衡，但保持简单）
            self.data = self.data[:max_samples_per_class * 2]  # 粗略平衡
        
        print(f"Blur mode: {self.blur_mode}, Blur prob: {self.blur_prob}")
        if self.enable_strong_aug:
            print("✓ Strong augmentation enabled")
        if self.ccmba_loader:
            print("✓ CCMBA data loader initialized with new multi-category structure")
    
    def _get_image_files(self, folder_path):
        """获取图像文件列表（兼容旧结构）"""
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
        """
        根据配置的模糊模式应用数据增强（修复设备一致性）
        """
        # 记住原始设备
        original_device = tensor_image.device
        
        if random.random() > self.blur_prob:
            return tensor_image, {'mode': 'no_blur', 'total_time': 0}
        
        blur_start_time = time.time()
        
        if self.blur_mode == "global":
            blur_strength = random.uniform(*self.blur_strength_range)
            # 确保输入在正确设备上
            if tensor_image.device != original_device:
                tensor_image = tensor_image.to(original_device)
            blurred_tensor = apply_motion_blur_batch_efficient(tensor_image.unsqueeze(0), blur_strength).squeeze(0)
            # 确保输出在原始设备上
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
                blurred_tensor = blurred_tensor.to(original_device)  # 确保设备一致
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
                    blurred_tensor = self.train_transform(blurred_pil)
                    # 关键修复：确保CCMBA处理后的tensor在正确设备上
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
                    blurred_tensor = blurred_tensor.to(original_device)  # 确保设备一致
                    blur_info = {
                        'mode': 'ccmba_fallback_global',
                        'blur_strength': blur_strength,
                        'ccmba_data_load_time': ccmba_data_time,
                        'total_time': time.time() - blur_start_time
                    }
        elif self.blur_mode == "mixed":
            # 对于mixed模式，也需要确保设备一致性
            blurred_tensor, blur_info = apply_mixed_blur_enhanced(
                tensor_image, self.ccmba_loader, image_name, category, is_real,
                self.blur_strength_range, self.mixed_mode_ratio
            )
            # 确保最终结果在原始设备上
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
                img_path, label, category = self.data[current_idx]  # 解包 category
                
                image = Image.open(img_path).convert('RGB')
                width, height = image.size
                if width == 0 or height == 0:
                    raise ValueError(f"Invalid image dimensions: {width}x{height}")
                
                np.array(image)  # 验证数据完整性
                
                # 应用数据变换
                if self.enable_strong_aug and random.random() < 0.4:
                    # 50%概率使用强力增强
                    tensor_image = self.strong_transform(image)
                else:
                    # 使用常规变换
                    tensor_image = self.transform(image)
                
                # 提取文件名用于CCMBA查找
                image_name = img_path.stem
                
                return tensor_image, label, image_name, category  # 返回真实的 category
                
            except Exception as e:
                add_corrupted_image(self.data[current_idx][0], str(e))
                print(f"Warning: Skipping corrupted image - {self.data[current_idx][0]} (category: {self.data[current_idx][2] if len(self.data[current_idx]) > 2 else 'unknown'})")
                retry_count += 1
                
                if retry_count >= max_retries:
                    raise RuntimeError(f"Too many consecutive corrupted images starting from index {idx}")
        
        raise RuntimeError("Unexpected error in __getitem__")

# ===============================
# 强力数据增强 - 基于提供的代码
# ===============================

# 导入额外的变换
from torchvision.transforms import functional as TF, InterpolationMode
import cv2
from io import BytesIO

# 添加JPEG压缩函数
def apply_PILJPEG(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")
    return img

def apply_cv2JPEG(img, quality):
    # convert PIL image to cv2 image
    img_cv2 = np.array(img)
    img_cv2 = img_cv2[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return Image.fromarray(decimg[:,:,::-1])

def apply_randomJPEG(img, quality):
    if random.random() < 0.5:
        img = apply_PILJPEG(img, quality)
    else:
        img = apply_cv2JPEG(img, quality)
    return img

def cutout(img, pad_size, replace=0):
    """Apply cutout to PIL image"""
    if isinstance(img, Image.Image):
        img_tensor = TF.pil_to_tensor(img)
    else:
        img_tensor = img
    
    channels, height, width = img_tensor.shape
    cutout_center_height = torch.randint(low=0, high=height, size=(1,)).item()
    cutout_center_width = torch.randint(low=0, high=width, size=(1,)).item()

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = max(0, height - cutout_center_height - pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = max(0, width - cutout_center_width - pad_size)

    cutout_shape = (height - (lower_pad + upper_pad),
                    width - (left_pad + right_pad))
    padding_dims = (left_pad, right_pad, upper_pad, lower_pad)
    cutout_mask = torch.nn.functional.pad(
        torch.zeros(cutout_shape, dtype=img_tensor.dtype),
        padding_dims, value=1
    )
    cutout_mask = cutout_mask.unsqueeze(dim=0)
    cutout_mask = torch.tile(cutout_mask, (channels,1,1))
    
    img_tensor = torch.where(
        cutout_mask==0,
        torch.ones_like(img_tensor, dtype=img_tensor.dtype) * replace,
        img_tensor
    )
    
    if isinstance(img, Image.Image):
        return TF.to_pil_image(img_tensor)
    else:
        return img_tensor

# RandAugment实现
class RandAugment_Strong(nn.Module):
    """强力RandAugment增强"""
    def __init__(self, num_ops=3, magnitude=10):
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.ops = [
            'AutoContrast', 'Equalize', 'Rotate', 'Solarize', 'Color', 
            'Posterize', 'Contrast', 'Brightness', 'Sharpness', 
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout'
        ]
        
    def forward(self, img):
        for _ in range(self.num_ops):
            op = random.choice(self.ops)
            magnitude = random.uniform(0.1, 1.0) * self.magnitude
            img = self.apply_op(img, op, magnitude)
        return img
    
    def apply_op(self, img, op, magnitude):
        if op == 'AutoContrast':
            return TF.autocontrast(img)
        elif op == 'Equalize':
            return TF.equalize(img)
        elif op == 'Rotate':
            angle = magnitude * 30 if random.random() > 0.5 else -magnitude * 30
            return TF.rotate(img, angle)
        elif op == 'Solarize':
            return TF.solarize(img, int(255 - magnitude * 255))
        elif op == 'Color':
            factor = 1.0 + magnitude * 0.9 if random.random() > 0.5 else 1.0 - magnitude * 0.9
            return TF.adjust_saturation(img, max(0.1, factor))
        elif op == 'Posterize':
            bits = int(8 - magnitude * 4)
            return TF.posterize(img, max(1, bits))
        elif op == 'Contrast':
            factor = 1.0 + magnitude * 0.9 if random.random() > 0.5 else 1.0 - magnitude * 0.9
            return TF.adjust_contrast(img, max(0.1, factor))
        elif op == 'Brightness':
            factor = 1.0 + magnitude * 0.9 if random.random() > 0.5 else 1.0 - magnitude * 0.9
            return TF.adjust_brightness(img, max(0.1, factor))
        elif op == 'Sharpness':
            factor = 1.0 + magnitude * 0.9 if random.random() > 0.5 else 1.0 - magnitude * 0.9
            return TF.adjust_sharpness(img, max(0.1, factor))
        elif op == 'ShearX':
            shear = magnitude * 0.3 if random.random() > 0.5 else -magnitude * 0.3
            return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear, 0])
        elif op == 'ShearY':
            shear = magnitude * 0.3 if random.random() > 0.5 else -magnitude * 0.3
            return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, shear])
        elif op == 'TranslateX':
            translate = int(magnitude * 100) if random.random() > 0.5 else -int(magnitude * 100)
            return TF.affine(img, angle=0, translate=[translate, 0], scale=1.0, shear=[0, 0])
        elif op == 'TranslateY':
            translate = int(magnitude * 100) if random.random() > 0.5 else -int(magnitude * 100)
            return TF.affine(img, angle=0, translate=[0, translate], scale=1.0, shear=[0, 0])
        elif op == 'Cutout':
            return cutout(img, int(magnitude * 40), replace=128)
        return img
        

"""
        # 保守的颜色增强 - 避免过度破坏AI生成图像的特征
        transforms.ColorJitter(
            brightness=0.1,  # 降低brightness变化
            contrast=0.1,    # 降低contrast变化  
            saturation=0.1,  # 降低saturation变化
            hue=0.05        # 降低hue变化
        ),
"""

def get_balanced_train_transforms():
    """平衡的数据增强策略 - 避免过度破坏判别特征"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
    
        transforms.RandomCrop((448, 448)),
        # 轻度几何变换
        # transforms.RandomHorizontalFlip(p=0.5),
        
        #transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),

        # 轻度JPEG压缩（只对部分样本）
        transforms.Lambda(lambda img: apply_light_jpeg(img) if random.random() < 0.3 else img),
        
        # 转换为tensor
        transforms.ToTensor(),
        
        # 轻度随机擦除
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
        
        # 标准化
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def apply_light_jpeg(img, quality_range=(85, 95)):
    """轻度JPEG压缩，避免过度破坏特征"""
    quality = random.randint(quality_range[0], quality_range[1])
    return apply_randomJPEG(img, quality)


# ===============================
# 主训练函数 - 重新设计的流程
# ===============================
def main_distributed(rank, world_size, args):
    """分布式训练主函数 - 改进的Teacher-Student流程"""
    start_time = time.time()
    
    try:
        # 初始化分布式环境
        print(f"[Rank {rank}] Initializing distributed training...")
        setup_distributed(rank, world_size)
        setup_logging(rank)
        
        # 设置设备
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
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
            print("IMPROVED TEACHER-STUDENT TRAINING")
            print("- Phase 1: Train Teacher (DinoV3 backbone + clean data)")
            print("- Phase 2: Train Student (Teacher backbone + blur/CCMBA data)")
            print("- Full autocast FP16 training")
            print(f"{'='*60}")
        
        # 创建输出目录
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 创建数据变换
        train_transform = get_train_transforms()
        test_transform = get_test_transforms()
        
        if rank == 0:
            print("Creating datasets...")

        # Teacher训练数据集（无模糊增强）
        teacher_train_dataset = AdapterDataset(
            root_folder=TRAIN_ROOT_FOLDER,  # 使用新根路径
            transform=train_transform,
            max_samples_per_class=None,
            blur_prob=0.0,  # Teacher阶段不使用模糊
            enable_strong_aug=True  # 使用其他数据增强
        )

        # Student训练数据集（有模糊增强，使用新 CCMBA）
        student_train_dataset = AdapterDataset(
            root_folder=TRAIN_ROOT_FOLDER,  # 使用新根路径
            transform=train_transform,
            max_samples_per_class=None,
            blur_prob=BLUR_PROB,  # Student阶段使用模糊
            blur_strength_range=BLUR_STRENGTH_RANGE,
            blur_mode=BLUR_MODE,
            mixed_mode_ratio=MIXED_MODE_RATIO,
            ccmba_data_dir=CCMBA_DATA_DIR,  # 新 CCMBA 路径
            enable_strong_aug=True
        )

        # 测试数据集（保持原有结构，无 CCMBA）
        test_dataset = AdapterDataset(
            real_folder=TEST_REAL_FOLDER,
            fake_folder=TEST_FAKE_FOLDER,
            transform=test_transform,
            max_samples_per_class=None,
            blur_prob=0.0,
            enable_strong_aug=False
        )

            
        if rank == 0:
            print(f"Dataset sizes - Teacher Train: {len(teacher_train_dataset)}")
            print(f"                Student Train: {len(student_train_dataset)}")
            print(f"                Test: {len(test_dataset)}")
        
        # 创建模型
        if rank == 0:
            print("Creating Teacher-Student model...")
            print(f"Memory before model loading: {get_memory_usage()}")

        try:
            model = TeacherStudentNetwork(
                dinov3_model_path=DINOV3_MODEL_ID,
                num_classes=2,
                projection_dim=PROJECTION_DIM,
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
            device_ids=[rank] if device.type == 'cuda' else None,
            output_device=rank if device.type == 'cuda' else None,
            find_unused_parameters=True,
        )
        
        if rank == 0:
            print("Starting improved teacher-student training process...")
        
        if world_size > 1:
            dist.barrier()
        
        # 创建GradScaler用于混合精度训练
        scaler = GradScaler()
        
        # ============================
        # 第一阶段：训练Teacher
        # ============================
        if rank == 0:
            print(f"{'='*60}")
            print("PHASE 1: TEACHER TRAINING")
            print(f"{'='*60}")
        
        # Teacher数据加载器
        teacher_train_sampler = DistributedSampler(
            teacher_train_dataset, num_replicas=world_size, rank=rank, 
            shuffle=True, drop_last=True
        )
        
        teacher_train_loader = DataLoader(
            teacher_train_dataset, batch_size=args.teacher_batch_size, 
            sampler=teacher_train_sampler, num_workers=NUM_WORKERS, 
            pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=2
        )
        
        # Teacher优化器
        teacher_params = list(model.module.teacher.projection.parameters()) + \
                        list(model.module.teacher.classifier.parameters())
        
        teacher_optimizer = optim.AdamW(teacher_params, lr=TEACHER_LEARNING_RATE, weight_decay=1e-4)
        teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=args.teacher_epochs)
        
        # 训练Teacher
        teacher_history = train_teacher_phase(
            model, teacher_train_loader,
            teacher_optimizer, teacher_scheduler, scaler, rank, world_size, device, args
        )
        
        # ============================
        # 第二阶段：训练Student
        # ============================
        if rank == 0:
            print(f"{'='*60}")
            print("PHASE 2: STUDENT TRAINING")
            print(f"{'='*60}")
        
        # Student数据加载器
        student_train_sampler = DistributedSampler(
            student_train_dataset, num_replicas=world_size, rank=rank, 
            shuffle=True, drop_last=True
        )
        
        student_train_loader = DataLoader(
            student_train_dataset, batch_size=args.student_batch_size, 
            sampler=student_train_sampler, num_workers=NUM_WORKERS, 
            pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=2
        )
        
        # Student优化器
        student_params = list(model.module.student_projection.parameters()) + \
                        list(model.module.student_classifier.parameters())
        
        student_optimizer = optim.AdamW(student_params, lr=STUDENT_LEARNING_RATE, weight_decay=1e-4)
        student_scheduler = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=args.student_epochs)
        
        # 训练Student
        student_history = train_student_phase(
            model, student_train_loader,
            student_optimizer, student_scheduler, scaler, rank, world_size, device, args
        )
        
        # ============================
        # 评估阶段
        # ============================
        if rank == 0:
            print("Starting final evaluation...")
        
        # 测试数据加载器
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, 
            shuffle=False, drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=args.student_batch_size, 
            sampler=test_sampler, num_workers=NUM_WORKERS, 
            pin_memory=True, persistent_workers=True
        )
        
        if world_size > 1:
            dist.barrier()
        
        # 评估模型
        results = evaluate_model_distributed(model, test_loader, args.output_dir, rank, world_size, device)
        
        # 保存训练历史
        if rank == 0:
            combined_history = {
                'teacher_history': teacher_history,
                'student_history': student_history,
                'model_config': {
                    'dinov3_model_id': DINOV3_MODEL_ID,
                    'projection_dim': PROJECTION_DIM,
                    'temperature': TEMPERATURE,
                    'alpha_distill': ALPHA_DISTILL,
                    'alpha_cls': ALPHA_CLS,
                    'alpha_feature': ALPHA_FEATURE
                },
                'training_config': {
                    'world_size': world_size,
                    'teacher_batch_size': args.teacher_batch_size * world_size,
                    'student_batch_size': args.student_batch_size * world_size,
                    'teacher_epochs': args.teacher_epochs,
                    'student_epochs': args.student_epochs,
                    'autocast_enabled': True,
                    'blur_augmentation': True
                }
            }
            
            with open(os.path.join(args.output_dir, 'improved_training_history.json'), 'w') as f:
                json.dump(combined_history, f, indent=2)
            
            print("Improved Teacher-Student training completed!")
            print(f"Results saved to: {args.output_dir}")
            if results:
                print("Final model performance:")
                print(f"Teacher (DinoV3) - Accuracy: {results['teacher']['accuracy']:.4f}")
                print(f"Student (Distilled) - Accuracy: {results['student']['accuracy']:.4f}")
            
            # 新增：保存模型 Checkpoint
            print("Saving model checkpoints...")
            # 保存 Teacher 模型
            teacher_path = os.path.join(args.output_dir, "nofocal_teacher_model_progan.pth")
            torch.save(model.module.state_dict(), teacher_path)  # 使用 module 解包 DDP
            print(f"Teacher model saved to: {teacher_path}")
            
            # 保存 Student 模型（完整模型，包括共享 backbone）
            student_path = os.path.join(args.output_dir, "nofocal_student_model_progan.pth")
            # 如果需要只保存 Student 部分，可以提取；这里保存完整模型以便加载
            torch.save(model.module.state_dict(), student_path)
            print(f"Student model saved to: {student_path}")
            
            # 保存损坏图片报告（原有）
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
    
    main_distributed(rank, world_size, args)

# ===============================
# 命令行接口
# ===============================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Improved Teacher-Student Training with DinoV3')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--teacher-epochs', type=int, default=8,
                        help='Number of teacher training epochs')
    parser.add_argument('--student-epochs', type=int, default=15,
                        help='Number of student training epochs')
    parser.add_argument('--teacher-batch-size', type=int, default=64,
                        help='Teacher batch size per GPU')
    parser.add_argument('--student-batch-size', type=int, default=32,
                        help='Student batch size per GPU')
    parser.add_argument('--output-dir', type=str, default="progan_dinov3_teacher_student",  # 修改默认名称
                        help='Output directory for models and logs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        if 'RANK' in os.environ:
            # 使用torchrun启动
            run_distributed_training(args)
        else:
            # 单进程模式
            print("Running in single process mode")
            main_distributed(0, 1, args)
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
