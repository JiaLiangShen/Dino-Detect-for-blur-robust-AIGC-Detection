"""
The bakcbone model of this training process can be found and downloaded in this website:
https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m

This python file is:
A distributed Teacher-Student training framework using DinoV3 as backbone.
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
    global corrupted_images
    corrupted_images.append({
        'path': str(image_path),
        'error': str(error_msg)
    })

def save_corrupted_images_report(output_dir, rank=0):
    """process of the corrrupted images"""
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


def setup_distributed(rank, world_size, backend=None):
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
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)

# ===============================
# 配置参数
# ===============================
DINOV3_MODEL_ID = "/home/work/xueyunqi/dinov3-vit7b16-pretrain-lvd1689m"

TRAIN_ROOT_FOLDER = "/home/work/xueyunqi/11ar_datasets/extracted"
#TRAIN_REAL_FOLDER = "/home/work/xueyunqi/11ar/fake"
#TRAIN_FAKE_FOLDER = "/home/work/xueyunqi/11ar/fake"


TEACHER_LEARNING_RATE = 1e-4
STUDENT_LEARNING_RATE = 5e-5
PROJECTION_DIM = 512
TEMPERATURE = 0.07
NUM_WORKERS = 4

# Loss weights
ALPHA_DISTILL = 1.0
ALPHA_CLS = 1.0
ALPHA_FEATURE = 0.5

DATASET_NAME = "0917_dinov3_teacher_student"

# Motion blur settings
BLUR_PROB = 0.1 #修改模糊应用比例请看这一行！！！
BLUR_STRENGTH_RANGE = (0.1, 0.3)
BLUR_MODE = "mixed"

MIXED_MODE_RATIO = 0.5    ## the ratio of using CCMBA motion blur vs global motion blur
CCMBA_DATA_DIR = "/home/work/xueyunqi/11ar_datasets/progan_ccmba_train"


# ===============================
# Motion Blur tools
# ===============================
def apply_motion_blur(image_pil, strength):

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

    if not isinstance(images, torch.Tensor):
        return apply_motion_blur_batch(images, strength)  # fallback
    
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

    if isinstance(images, torch.Tensor):
        return apply_motion_blur_batch_efficient(images, strength)
    elif isinstance(images, list):
        return [apply_motion_blur(img, strength) for img in images]
    else:
        return apply_motion_blur(images, strength)



# ===============================
# Data augmentations
# ===============================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        
        transforms.RandomCrop((448, 448)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# ===============================
# L2Norm layer, nn.Module
# ===============================
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class ImprovedDinoV3Adapter(nn.Module):
    
    def __init__(self, model_path, num_classes=2, projection_dim=512, 
                 adapter_layers=2, dropout_rate=0.1, device="cuda"):
        super().__init__()
        
        self.device = device
        self.projection_dim = projection_dim
        
        print(f"Loading DinoV3 model: {model_path}")
        
        self.backbone = self._load_dinov3_backbone(model_path, device)
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 4096)
        
        print(f"DinoV3 feature dimension: {self.hidden_size}")

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
            
            for param in backbone.parameters():
                param.requires_grad = False
            
            return backbone
            
        except Exception as e:
            print(f"Error loading DinoV3 backbone: {e}")
            raise e
    
    def _initialize_weights(self):
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
        with torch.no_grad():
            with autocast():
                outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
                
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0] 
                
                return features.float() 
    
    def forward(self, pixel_values, return_features=False):

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
        raw_features = self.extract_features(pixel_values)
        with autocast():
            projected_features = self.projection(raw_features)
        return projected_features
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_projection(self):
        for param in self.projection.parameters():
            param.requires_grad = True
    
    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True


class CCMBADataLoader:
    
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
        if self.categories is None:
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
                for key, value in category_real_mapping.items():
                    self.real_mapping[f"{category}_{key}"] = value
                print(f"  - {category}/nature (real): {len(category_real_mapping)} entries")
            
            if fake_dir.exists():
                category_fake_mapping = self._build_mapping(fake_dir)
                for key, value in category_fake_mapping.items():
                    self.fake_mapping[f"{category}_{key}"] = value
                print(f"  - {category}/ai (fake): {len(category_fake_mapping)} entries")
    
    def _build_mapping(self, class_dir):
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
        mapping = self.real_mapping if is_real else self.fake_mapping
        key = f"{category}_{image_name}"
        
        if key in mapping:
            return mapping[key]
        else:
            return None
    
    def load_ccmba_blur_data(self, image_name, category, is_real=True):
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
# Teacher-Student network
# ===============================
class TeacherStudentNetwork(nn.Module):
    
    def __init__(self, dinov3_model_path, num_classes=2, projection_dim=512, device="cuda"):
        super().__init__()
        
        self.device = device
        
        print(f"{'='*60}")
        print("INITIALIZING TEACHER-STUDENT NETWORK")
        print(f"{'='*60}")
        

        print("Creating Teacher model (DinoV3 + Adapter)...")
        self.teacher = ImprovedDinoV3Adapter(
            model_path=dinov3_model_path,
            num_classes=num_classes,
            projection_dim=projection_dim,
            adapter_layers=3,
            dropout_rate=0.1,
            device=device
        )
        
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
        
        self._initialize_student_weights()
        
        print(f"✓ Teacher-Student network initialized")
        print(f"  - Shared DinoV3 backbone: {self.teacher.hidden_size}D")
        print(f"  - Projection dimension: {projection_dim}D")
        print(f"  - Number of classes: {num_classes}")
    
    def _initialize_student_weights(self):
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
        outputs = self.teacher(pixel_values, return_features=True)
        return outputs['projected_features'], outputs['logits']
    
    def forward_student(self, pixel_values):

        raw_features = self.teacher.extract_features(pixel_values)
        
        with autocast():
            projected_features = self.student_projection(raw_features)
            logits = self.student_classifier(projected_features)
        
        return projected_features, logits
    
    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def unfreeze_teacher_head(self):
        self.teacher.unfreeze_projection()
        self.teacher.unfreeze_classifier()
    
    def unfreeze_student(self):
        for param in self.student_projection.parameters():
            param.requires_grad = True
        for param in self.student_classifier.parameters():
            param.requires_grad = True


class ImprovedTeacherStudentLoss(nn.Module):
    
    def __init__(self, temperature=0.07, alpha_distill=1.0, alpha_cls=1.0, 
                 alpha_feature=0.5, alpha_simclr=0.3):  
        super().__init__()
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_cls = alpha_cls
        self.alpha_feature = alpha_feature
        self.alpha_simclr = alpha_simclr
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.simclr_loss = SimCLRLoss(temperature=temperature)
    
    def distillation_loss(self, student_logits, teacher_logits):
        with autocast():
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
            
            kl_loss = self.kl_loss(student_log_probs, teacher_probs)
            return kl_loss * (self.temperature ** 2)
    
    def feature_alignment_loss(self, student_features, teacher_features):
        with autocast():
            student_norm = F.normalize(student_features, p=2, dim=1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=1)
            
            cosine_sim = (student_norm * teacher_norm).sum(dim=1)
            loss = 1.0 - cosine_sim.mean()
            
            return loss
    
    def forward(self, student_features, student_logits, teacher_features=None, 
                teacher_logits=None, labels=None, mode="student", student_features_aug=None):
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
                cls_loss = self.ce_loss(student_logits, labels)
                distill_loss = self.distillation_loss(student_logits, teacher_logits)
                feature_loss = self.feature_alignment_loss(student_features, teacher_features)
                simclr_loss = torch.tensor(0.0, device=cls_loss.device)
                if student_features_aug is not None:
                    combined_features = torch.cat([student_features, student_features_aug], dim=0)
                    simclr_loss = self.simclr_loss(combined_features)
                
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
    if rank == 0:
        print(f"Starting Teacher training phase...")
    
    if rank == 0:
        class_weights = calculate_class_weights(train_loader.dataset)
    else:
        class_weights = None
    
    if dist.is_initialized() and world_size > 1:
        if rank == 0:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            class_weights_tensor = torch.zeros(2, dtype=torch.float32, device=device)
        
        dist.broadcast(class_weights_tensor, src=0)
        class_weights = class_weights_tensor.cpu().tolist()
    
    criterion = ImprovedTeacherStudentLoss(
        temperature=TEMPERATURE,
        alpha_distill=ALPHA_DISTILL,
        alpha_cls=ALPHA_CLS,
        alpha_feature=ALPHA_FEATURE
    )
    
    history = {'train_loss': [], 'train_acc': []}
    best_acc = 0.0
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    actual_model.freeze_teacher()  
    actual_model.unfreeze_teacher_head()  
    
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
            if len(batch_data) == 4:
                images, labels, _, _ = batch_data
            else:
                images, labels = batch_data[:2]
            
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                teacher_features, teacher_logits = actual_model.forward_teacher(images)
                
                losses = criterion(
                    student_features=teacher_features,
                    student_logits=teacher_logits,     
                    labels=labels, 
                    mode="teacher"
                )
            
            scaler.scale(losses['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
    if rank == 0:
        print(f"Starting Student training phase with blur augmentation...")
    
    if rank == 0:
        class_weights = calculate_class_weights(train_loader.dataset)
    else:
        class_weights = None
    
    if dist.is_initialized() and world_size > 1:
        if rank == 0:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            class_weights_tensor = torch.zeros(2, dtype=torch.float32, device=device)
        
        dist.broadcast(class_weights_tensor, src=0)
        class_weights = class_weights_tensor.cpu().tolist()
    
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
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    actual_model.freeze_teacher() 
    actual_model.unfreeze_student()  
    
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
            if len(batch_data) == 4:
                images, labels, image_names, categories = batch_data
            else:
                images, labels = batch_data[:2]
                image_names = [f"img_{i}" for i in range(len(images))]
                categories = ["train"] * len(images)
            
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.no_grad():
                with autocast():
                    teacher_features, teacher_logits = actual_model.forward_teacher(images)
            
            student_imgs_list = []
            for i, (img_tensor, img_name, label, category) in enumerate(zip(images, image_names, labels, categories)):
                is_real = (label.item() == 0)
                
                if img_tensor.device != device:
                    img_tensor = img_tensor.to(device, non_blocking=True)
                
                if hasattr(train_loader.dataset, 'apply_blur_augmentation'):
                    blurred_tensor, _ = train_loader.dataset.apply_blur_augmentation(
                        img_tensor, img_name, category, is_real
                    )
                else:
                    blurred_tensor = img_tensor

                if blurred_tensor.device != device:
                    blurred_tensor = blurred_tensor.to(device, non_blocking=True)
                
                student_imgs_list.append(blurred_tensor)

            student_imgs_list = [t.to(device, non_blocking=True) if t.device != device else t 
                               for t in student_imgs_list]
            
            student_imgs = torch.stack(student_imgs_list)
            
            if student_imgs.device != device:
                student_imgs = student_imgs.to(device, non_blocking=True)
            
            with autocast():
                student_features, student_logits = actual_model.forward_student(student_imgs)
                
                student_imgs_aug_list = []
                for img_tensor in student_imgs:
                    blur_strength = random.uniform(0.1, 0.3)
                    aug_tensor = apply_motion_blur_batch_efficient(
                        img_tensor.unsqueeze(0), blur_strength
                    ).squeeze(0)
                    student_imgs_aug_list.append(aug_tensor)
                
                student_imgs_aug = torch.stack(student_imgs_aug_list).to(device, non_blocking=True)
                student_features_aug, _ = actual_model.forward_student(student_imgs_aug)

                losses = criterion(
                    student_features=student_features,
                    student_logits=student_logits,
                    teacher_features=teacher_features,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    mode="student",
                    student_features_aug=student_features_aug
                )
            
            scaler.scale(losses['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            
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


def all_reduce_tensor(tensor, world_size):
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
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features):
        with autocast():
            features = F.normalize(features, p=2, dim=1)
            batch_size = features.shape[0] // 2            
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            
            mask = torch.eye(batch_size * 2, dtype=torch.bool, device=features.device)
            positive_mask = torch.zeros_like(mask)
            for i in range(batch_size):
                positive_mask[i, i + batch_size] = True
                positive_mask[i + batch_size, i] = True
            
            negative_mask = ~(mask | positive_mask)
            
            losses = []
            for i in range(batch_size * 2):
                pos_sim = similarity_matrix[i][positive_mask[i]]    
                neg_sim = similarity_matrix[i][negative_mask[i]]
                # InfoNCE loss
                logits = torch.cat([pos_sim, neg_sim])
                labels = torch.zeros(len(logits), dtype=torch.long, device=features.device)
                
                loss = F.cross_entropy(logits.unsqueeze(0), labels[:1])
                losses.append(loss)
            
            return torch.stack(losses).mean()



def calculate_class_weights(train_dataset):
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
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU: {gpu_memory:.2f}GB allocated, {gpu_reserved:.2f}GB reserved"
    else:
        import psutil
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        return f"CPU: {cpu_memory:.2f}GB used"

def clear_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class AdapterDataset(Dataset):
    
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
        
        if enable_strong_aug:
            self.strong_transform = get_balanced_train_transforms()
        
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
                                self.data.append((img_file, 0, category_name))  # (path, label=0, category)
                                real_count += 1
                    
                    if fake_subfolder.exists():
                        for img_file in fake_subfolder.rglob('*'):
                            if img_file.is_file() and img_file.suffix.lower() in extensions:
                                self.data.append((img_file, 1, category_name))  # (path, label=1, category)
                                fake_count += 1
            
            print(f"Dataset loaded from root {root_folder}: {real_count} real, {fake_count} fake images (multi-class structure with categories)")
        
        elif real_folder and fake_folder:  
            self.real_images = self._get_image_files(real_folder)
            self.fake_images = self._get_image_files(fake_folder)
            for img_path in self.real_images:
                self.data.append((img_path, 0, 'test'))  
            for img_path in self.fake_images:
                self.data.append((img_path, 1, 'test'))
            print(f"Dataset loaded: {len(self.real_images)} real, {len(self.fake_images)} fake images (legacy structure)")
        
        else:
            raise ValueError("Must provide either root_folder or both real_folder and fake_folder")
        
        if max_samples_per_class:
            self.data = self.data[:max_samples_per_class * 2] 
        
        print(f"Blur mode: {self.blur_mode}, Blur prob: {self.blur_prob}")
        if self.enable_strong_aug:
            print("✓ Strong augmentation enabled")
        if self.ccmba_loader:
            print("✓ CCMBA data loader initialized with new multi-category structure")
    
    def _get_image_files(self, folder_path):
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
                    blurred_tensor = self.train_transform(blurred_pil)
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
                
                if self.enable_strong_aug and random.random() < 0.4:
                    tensor_image = self.strong_transform(image)
                else:
                    tensor_image = self.transform(image)

                image_name = img_path.stem
                
                return tensor_image, label, image_name, category  

            except Exception as e:
                add_corrupted_image(self.data[current_idx][0], str(e))
                print(f"Warning: Skipping corrupted image - {self.data[current_idx][0]} (category: {self.data[current_idx][2] if len(self.data[current_idx]) > 2 else 'unknown'})")
                retry_count += 1
                
                if retry_count >= max_retries:
                    raise RuntimeError(f"Too many consecutive corrupted images starting from index {idx}")
        
        raise RuntimeError("Unexpected error in __getitem__")

from torchvision.transforms import functional as TF, InterpolationMode
import cv2
from io import BytesIO

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

class RandAugment_Strong(nn.Module):
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
        


def get_balanced_train_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
    
        transforms.RandomCrop((448, 448))

        transforms.Lambda(lambda img: apply_light_jpeg(img) if random.random() < 0.3 else img),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def apply_light_jpeg(img, quality_range=(85, 95)):
    quality = random.randint(quality_range[0], quality_range[1])
    return apply_randomJPEG(img, quality)

def main_distributed(rank, world_size, args):
    start_time = time.time()
    try:
        print(f"[Rank {rank}] Initializing distributed training...")
        setup_distributed(rank, world_size)
        setup_logging(rank)

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
        
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
        
        train_transform = get_train_transforms()

        if rank == 0:
            print("Creating datasets...")

        teacher_train_dataset = AdapterDataset(
            root_folder=TRAIN_ROOT_FOLDER, 
            transform=train_transform,
            max_samples_per_class=None,
            blur_prob=0.0, 
            enable_strong_aug=True 
        )

        student_train_dataset = AdapterDataset(
            root_folder=TRAIN_ROOT_FOLDER,  
            transform=train_transform,
            max_samples_per_class=None,
            blur_prob=BLUR_PROB,  
            blur_strength_range=BLUR_STRENGTH_RANGE,
            blur_mode=BLUR_MODE,
            mixed_mode_ratio=MIXED_MODE_RATIO,
            ccmba_data_dir=CCMBA_DATA_DIR,  
            enable_strong_aug=True
        )

        if rank == 0:
            print(f"Dataset sizes - Teacher Train: {len(teacher_train_dataset)}")
            print(f"                Student Train: {len(student_train_dataset)}")
            # 不打印测试集大小

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
    
        scaler = GradScaler()
        
        # ============================
        # train Teacher
        # ============================
        if rank == 0:
            print(f"{'='*60}")
            print("PHASE 1: TEACHER TRAINING")
            print(f"{'='*60}")
        
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

        teacher_params = list(model.module.teacher.projection.parameters()) + \
                        list(model.module.teacher.classifier.parameters())
        
        teacher_optimizer = optim.AdamW(teacher_params, lr=TEACHER_LEARNING_RATE, weight_decay=1e-4)
        teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=args.teacher_epochs)

        teacher_history = train_teacher_phase(
            model, teacher_train_loader,
            teacher_optimizer, teacher_scheduler, scaler, rank, world_size, device, args
        )
        
        # ============================
        # train Student
        # ============================
        if rank == 0:
            print(f"{'='*60}")
            print("PHASE 2: STUDENT TRAINING")
            print(f"{'='*60}")

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

        student_params = list(model.module.student_projection.parameters()) + \
                        list(model.module.student_classifier.parameters())
        
        student_optimizer = optim.AdamW(student_params, lr=STUDENT_LEARNING_RATE, weight_decay=1e-4)
        student_scheduler = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=args.student_epochs)

        student_history = train_student_phase(
            model, student_train_loader,
            student_optimizer, student_scheduler, scaler, rank, world_size, device, args
        )
        
        if rank == 0:
            print("Training completed!")
            # 不调用测试和打印

            print("Saving model checkpoints...")
            teacher_path = os.path.join(args.output_dir, "nofocal_teacher_model_progan.pth")
            torch.save(model.module.state_dict(), teacher_path) 
            print(f"Teacher model saved to: {teacher_path}")
            
            student_path = os.path.join(args.output_dir, "nofocal_student_model_progan.pth")
            torch.save(model.module.state_dict(), student_path)
            print(f"Student model saved to: {student_path}")
            
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
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    main_distributed(rank, world_size, args)


def parse_args():
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
    parser.add_argument('--output-dir', type=str, default="progan_dinov3_teacher_student",  #
                        help='Output directory for models and logs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        if 'RANK' in os.environ:
            run_distributed_training(args)
        else:
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
