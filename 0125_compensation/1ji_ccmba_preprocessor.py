#!/usr/bin/env python3
"""
CCMBA预处理脚本 - 简化版
专为tstrain.py优化，只生成必要的文件
"""

import os
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import random

class CCMBAPreprocessor:
    """CCMBA预处理器 - 简化版"""
    
    def __init__(self):
        print("Initializing CCMBA Preprocessor (Simplified)...")
        
        # 简化的模糊核参数
        self.blur_kernels = self._generate_blur_kernels()
        
        # 简化的类别模糊配置
        self.class_blur_config = {
            'background': 0.1,
            'edge': 0.4,
            'texture': 0.25,
            'smooth': 0.15,
            'salient': 0.3
        }
        
        print(f"✓ Generated {len(self.blur_kernels)} blur kernels")
    
    def _generate_blur_kernels(self):
        """生成简化的模糊核"""
        kernels = []
        
        # 3种焦虑度 x 3种曝光时间 = 9种组合
        anxiety_levels = [0.005, 0.001, 0.00005]
        exposure_levels = [1/25, 1/10, 1/5]
        
        for i, anxiety in enumerate(anxiety_levels):
            for j, exposure in enumerate(exposure_levels):
                kernel = self._generate_motion_kernel(anxiety, exposure)
                kernels.append({
                    'kernel': kernel,
                    'anxiety': anxiety,
                    'exposure': exposure,
                    'level': f'L{i+1}E{j+1}'
                })
        
        return kernels
    
    def _generate_motion_kernel(self, anxiety, exposure, kernel_size=15):
        """生成运动模糊核"""
        num_steps = 100
        trajectory = []
        
        x, y = 0.0, 0.0
        vx, vy = np.random.uniform(-0.5, 0.5, 2)
        dt = exposure / num_steps
        
        for step in range(num_steps):
            dvx = anxiety * (np.random.normal(0, 1) - 0.1 * vx)
            dvy = anxiety * (np.random.normal(0, 1) - 0.1 * vy)
            
            dvx += 2 * anxiety * abs(vx) * np.random.uniform(-1, 1)
            dvy += 2 * anxiety * abs(vy) * np.random.uniform(-1, 1)
            
            vx += dvx * dt
            vy += dvy * dt
            x += vx * dt  
            y += vy * dt
            
            trajectory.append([x * 20, y * 20])
        
        trajectory = np.array(trajectory)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        
        for i in range(len(trajectory) - 1):
            x1 = int(trajectory[i][0] + center)
            y1 = int(trajectory[i][1] + center)
            x2 = int(trajectory[i+1][0] + center)  
            y2 = int(trajectory[i+1][1] + center)
            
            if (0 <= x1 < kernel_size and 0 <= y1 < kernel_size and
                0 <= x2 < kernel_size and 0 <= y2 < kernel_size):
                cv2.line(kernel, (y1, x1), (y2, x2), 1, 1)
        
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        else:
            kernel[center, :] = 1.0 / kernel_size
        
        return kernel
    
    def _generate_simple_masks(self, image_np):
        """生成简化的语义掩码"""
        height, width = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 1. 边缘掩码
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        edge_density = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edge_density = cv2.GaussianBlur(edge_density, (5, 5), 0)
        
        # 2. 纹理掩码
        gray_float = gray.astype(np.float32)
        mean_kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray_float, -1, mean_kernel)
        local_mean_sq = cv2.filter2D(gray_float**2, -1, mean_kernel)
        local_var = local_mean_sq - local_mean**2
        
        # 3. 显著性掩码
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        center_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        center_weight = 1 - (center_dist / np.max(center_dist))
        
        contrast = cv2.Laplacian(gray, cv2.CV_64F)
        contrast = np.abs(contrast)
        contrast_norm = contrast / (np.max(contrast) + 1e-8)
        saliency = center_weight * 0.6 + contrast_norm * 0.4
        
        # 生成掩码
        masks = {
            'background': (edge_density < np.percentile(edge_density, 30)).astype(np.float32),
            'edge': (edge_density > np.percentile(edge_density, 85)).astype(np.float32),
            'texture': (local_var > np.percentile(local_var, 70)).astype(np.float32),
            'smooth': (local_var < np.percentile(local_var, 40)).astype(np.float32),
            'salient': (saliency > np.percentile(saliency, 75)).astype(np.float32)
        }
        
        return masks
    
    def apply_ccmba_blur(self, image_np, masks, intensity=0.5):
        """应用CCMBA模糊"""
        height, width = image_np.shape[:2]
        
        # 随机选择要模糊的类别
        available_classes = list(masks.keys())
        num_classes_to_blur = random.randint(1, min(3, len(available_classes)))
        selected_classes = random.sample(available_classes, num_classes_to_blur)
        
        # 构建模糊掩码
        final_mask = np.zeros((height, width), dtype=np.float32)
        blur_info = {}
        
        for class_name in selected_classes:
            class_mask = masks[class_name]
            class_intensity = self.class_blur_config[class_name] * intensity
            weighted_mask = class_mask * class_intensity
            final_mask = np.maximum(final_mask, weighted_mask)
            
            blur_info[class_name] = {
                'intensity': float(class_intensity),
                'area_ratio': float(np.sum(class_mask) / (height * width))
            }
        
        # 随机选择模糊核
        kernel_info = random.choice(self.blur_kernels)
        kernel = kernel_info['kernel']
        
        # 应用模糊
        image_float = image_np.astype(np.float32) / 255.0
        foreground = image_float * final_mask[..., np.newaxis]
        
        blurred_foreground = np.zeros_like(foreground)
        for c in range(3):
            blurred_foreground[..., c] = cv2.filter2D(foreground[..., c], -1, kernel)
        
        blurred_mask = cv2.filter2D(final_mask, -1, kernel)
        blurred_mask = np.clip(blurred_mask, 0, 1)
        
        background_weight = 1 - blurred_mask
        preserved_background = image_float * background_weight[..., np.newaxis]
        
        final_image = blurred_foreground + preserved_background
        final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)
        
        return final_image, final_mask, blur_info, kernel_info
    
    def process_single_image(self, image_path, output_dir):
        """处理单张图像 - 只生成必要文件"""
        try:
            image_pil = Image.open(image_path).convert('RGB')
            image_np = np.array(image_pil)
            
            masks = self._generate_simple_masks(image_np)
            blurred_image, blur_mask, blur_info, kernel_info = self.apply_ccmba_blur(
                image_np, masks
            )
            
            base_name = Path(image_path).stem
            
            # 1. 保存模糊图像
            blur_img_path = output_dir / 'blurred_images' / f'{base_name}.jpg'
            Image.fromarray(blurred_image).save(blur_img_path, quality=95)
            
            # 2. 保存模糊掩码
            blur_mask_path = output_dir / 'blur_masks' / f'{base_name}.png'
            blur_mask_img = (blur_mask * 255).astype(np.uint8)
            Image.fromarray(blur_mask_img).save(blur_mask_path)
            
            # 3. 保存元数据（只保存必要信息）
            blur_pixels = blur_mask[blur_mask > 0]
            mean_intensity = float(np.mean(blur_pixels)) if len(blur_pixels) > 0 else 0.0
            
            metadata = {
                'original_image': str(image_path),
                'blur_info': blur_info,
                'kernel_info': {
                    'anxiety': float(kernel_info['anxiety']),
                    'exposure': float(kernel_info['exposure']),
                    'level': kernel_info['level']
                },
                'blur_mask_stats': {
                    'total_blur_ratio': float(np.sum(blur_mask) / blur_mask.size),
                    'max_intensity': float(np.max(blur_mask)),
                    'mean_intensity': mean_intensity
                }
            }
            
            metadata_path = output_dir / 'metadata' / f'{base_name}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True, metadata
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False, None
    
    def process_dataset(self, input_dir, output_dir, class_name):
        """处理整个数据集"""
        input_path = Path(input_dir)
        output_path = Path(output_dir) / class_name
        
        # 只创建必要的目录
        dirs_to_create = ['blurred_images', 'blur_masks', 'metadata']
        for dir_name in dirs_to_create:
            (output_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {'total_images': 0, 'processed_successfully': 0, 'failed': 0}
        
        print(f"Processing {len(image_files)} images from {class_name} class...")
        
        stats = {
            'total_images': len(image_files),
            'processed_successfully': 0,
            'failed': 0
        }
        
        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            success, _ = self.process_single_image(img_path, output_path)
            if success:
                stats['processed_successfully'] += 1
            else:
                stats['failed'] += 1
        
        # 保存简化的统计信息
        stats_path = output_path / 'ccmba_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ {class_name} processing completed:")
        print(f"  - Successful: {stats['processed_successfully']}")
        print(f"  - Failed: {stats['failed']}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='CCMBA Preprocessing - Simplified')
    parser.add_argument('--real_dir', required=True, help='Real images directory')
    parser.add_argument('--fake_dir', required=True, help='Fake images directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    processor = CCMBAPreprocessor()
    
    print("="*70)
    print("CCMBA PREPROCESSING PIPELINE (SIMPLIFIED)")
    print("="*70)
    
    # 处理真实和虚假图像
    real_stats = processor.process_dataset(args.real_dir, args.output_dir, 'real')
    fake_stats = processor.process_dataset(args.fake_dir, args.output_dir, 'fake')
    
    # 保存汇总统计
    total_stats = {
        'real': real_stats,
        'fake': fake_stats,
        'summary': {
            'total_processed': real_stats['processed_successfully'] + fake_stats['processed_successfully'],
            'total_failed': real_stats['failed'] + fake_stats['failed']
        }
    }
    
    summary_path = Path(args.output_dir) / 'ccmba_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    print("\n" + "="*70)
    print("CCMBA PREPROCESSING COMPLETED!")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Total processed: {total_stats['summary']['totalprocessed']}")
    print(f"Total failed: {total_stats['summary']['total_failed']}")
    print(f"Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
