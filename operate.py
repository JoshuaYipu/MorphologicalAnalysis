import numpy as np
import cv2
from typing import List, Tuple
import os
from scripts import resolution_decomposition, Modified_tophat, light_avg, adaptive_threshold

def detect_vessels(image: np.ndarray, num_levels:int = 3, window_size:int= 35, se_length:int = 25, 
                   block_size:int = 15, c:int = -2) -> np.ndarray:
    '''
    完整实现文献《Multi-scale morphological analysis for retinal vessel detection...》中提出的血管检测流程。

    算法步骤：
    1. 多分辨率分解：将输入图像分解为多个分辨率层次。
    2. 光照不均校正：对每个分辨率层次的图像进行光照不均校正。
    3. 改进顶帽滤波：使用不同方向的线性结构元素对校正后的图像进行改进顶帽滤波。
    4. 自适应阈值：将每个分辨率层次的响应图转换为二值血管图。
    5. 融合：将所有分辨率层次的二值血管图进行融合，得到最终的血管检测结果。
    
    Args:
        image (np.ndarray): 输入图像，建议为单通道灰度图 (H, W)。
        num_levels (int): 金字塔层数（包括原始图像）。例如，num_levels=3 将返回 [原图, 1/2尺寸图, 1/4尺寸图]。
        window_size (int): 最高层分辨率图像进行光照校正时使用的窗口大小，应该是一个正奇数。
        se_length (List[int]): 每个分辨率层次的结构元素长度列表，长度应等于 num_levels。
        block_size (int): 自适应阈值的邻域大小。
        c (int): 自适应阈值的常数偏移
    
    Returns:
        np.ndarray: 最终的二值血管图，0为背景，255为血管。
    '''
    # 设置初始参数
    window_sizes = [window_size - i * 4 for i in range(num_levels)]
    window_sizes = [ws if ws > 1 else 3 for ws in window_sizes]  # 确保窗口大小至少为3
    if window_sizes[-1] == 3 and window_sizes[-2] == 3:
        raise ValueError("window_size过小，无法为所有层次提供有效的窗口大小。请增加window_size或减少num_levels。")
    se_lengths = [se_length - i * 4 for i in range(num_levels)]
    se_lengths = [sl if sl > 3 else 3 for sl in se_lengths]  # 确保结构元素长度至少为3
    
    if len(image.shape) == 3:
        raise ValueError("输入图像必须是单通道灰度图。")
    
    # 1. 多分辨率分解
    pyramid = resolution_decomposition.multi_resolution_decompose(image, num_levels=num_levels)

    vessel_maps = []
    for level in range(num_levels):
        current_img = pyramid[level]
        # 2. 光照不均校正
        corrected = light_avg.uneven_illumination_correction(current_img, window_size=window_sizes[level], c=1e-6)

        if corrected.dtype != np.uint8:
            corrected_norm = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
            corrected_uint8 = corrected_norm.astype(np.uint8)
        else:
            corrected_uint8 = corrected
    
        # 3. 改进顶帽滤波   
        response_map = Modified_tophat.improved_top_hat_transform_v2(corrected_uint8, se_length=se_lengths[level])

        # 4. 自适应阈值
        binary_vessels = adaptive_threshold.adaptive_threshold_vessel_map(response_map, block_size=block_size, c=c)
        vessel_maps.append(binary_vessels)
    
    # 5. 融合
    fused_vessels = resolution_decomposition.fuse_vessel_maps(vessel_maps, original_shape=image.shape[:2])

    # 后处理：移除小的联通区域
    final_vessel_map = adaptive_threshold.remove_small_components(fused_vessels, min_size=200)
    return final_vessel_map



input_dir = 'data/FI-FFA/test/B'
output_dir = 'result/experiment_1'

os.makedirs(output_dir, exist_ok=True)

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
image_files = []
for file in os.listdir(input_dir):
    if any(file.lower().endswith(ext) for ext in image_extensions):
        image_files.append(file)

print(f"找到 {len(image_files)} 张图像进行处理。")

for i,filename in enumerate(image_files):
    print(f"处理图像 {i+1}/{len(image_files)}: {filename}")
    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"无法读取图像: {filename}，跳过。")
        continue
    
    vessel_map = detect_vessels(image, num_levels=3, window_size=35, se_length=25, block_size=15, c=-2)
    
    output_path = os.path.join(output_dir, f"vessel_{filename}")
    cv2.imwrite(output_path, vessel_map)
    print(f"保存结果到: {output_path}")

print("所有图像处理完成。")