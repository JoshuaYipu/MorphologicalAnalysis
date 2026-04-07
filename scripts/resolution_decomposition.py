import cv2
import numpy as np
from typing import List, Tuple

def multi_resolution_decompose(image: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """
    对输入图像进行多分辨率分解，构建高斯金字塔。
    
    Args:
        image (np.ndarray): 输入图像，灰度图或彩色图像。
        num_levels (int): 金字塔的层数（包括原始图像）。例如，num_levels=3 
                          将返回 [原图, 1/2尺寸图, 1/4尺寸图]。
    
    Returns:
        List[np.ndarray]: 一个列表，包含从高分辨率到低分辨率的图像。
    """
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1.")
    
    pyramid = [image.copy()]
    current_img = image.copy()
    
    # 进行下采样，构建金字塔
    for _ in range(1, num_levels):
        # cv2.pyrDown 默认使用5x5高斯核进行平滑后下采样
        current_img = cv2.pyrDown(current_img)
        pyramid.append(current_img)
    
    return pyramid

def fuse_vessel_maps(vessel_maps: List[np.ndarray], 
                    original_shape: Tuple[int, int]) -> np.ndarray:
    """
    融合多个尺度的血管检测结果。
    
    根据文献描述，使用高斯金字塔扩展将每个尺度的结果调整到原始图像大小，
    然后进行逻辑或操作（任何尺度检测到血管的位置都标记为血管）。
    
    Args:
        vessel_maps (List[np.ndarray]): 各尺度的二值血管图列表
        original_shape (Tuple[int, int]): 原始图像的形状 (height, width)
        
    Returns:
        np.ndarray: 融合后的最终血管图
    """
    fused_map = np.zeros(original_shape, dtype=np.uint8)
    
    for i, vessel_map in enumerate(vessel_maps):
        if i == 0:
            # 第0层是原始分辨率，直接使用
            if vessel_map.shape == original_shape:
                fused_map = np.logical_or(fused_map, vessel_map > 0)
            else:
                # 如果尺寸不匹配，进行调整
                resized_map = cv2.resize(vessel_map, (original_shape[1], original_shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
                fused_map = np.logical_or(fused_map, resized_map > 0)
        else:
            # 其他层需要上采样到原始尺寸
            resized_map = cv2.resize(vessel_map, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
            fused_map = np.logical_or(fused_map, resized_map > 0)
    
    # 转换为uint8格式，0为背景，255为血管
    final_map = (fused_map * 255).astype(np.uint8)
    return final_map

if __name__ == "__main__":
    # --- 调试示例 ---
    # 创建一个简单的测试图像 (模拟512x512的绿色通道图)
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # 设置参数
    num_levels = 4  # 分解为4个不同分辨率的图像
    
    # 执行多分辨率分解
    pyramid = multi_resolution_decompose(test_image, num_levels)
    print(f"分解完成，共 {len(pyramid)} 层。")
    for i, img in enumerate(pyramid):
        print(f"  Level {i}: {img.shape}")
    
    # 执行多分辨率融合 (仅作演示)
    fused = fuse_vessel_maps(pyramid, test_image.shape[:2])
    print(f"融合完成，最终图像尺寸: {fused.shape}")
    
    # 可视化（可选，需要在有图形界面的环境中运行）
    cv2.imshow('Original', test_image)
    cv2.imshow('Fused', fused)
    cv2.waitKey(0)
    cv2.destroyAllWindows()