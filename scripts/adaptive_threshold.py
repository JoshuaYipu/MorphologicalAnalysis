import cv2
import numpy as np

def adaptive_threshold_vessel_map(response_map: np.ndarray, block_size: int = 15, c: int = -2) -> np.ndarray:
    """
    使用自适应阈值法将形态学响应图转换为二值血管图。
    
    Args:
        response_map (np.ndarray): 改进顶帽滤波器生成的响应图，单通道。
        block_size (int): 自适应阈值计算的邻域大小（必须为奇数）。
        c (int): 从均值或加权均值中减去的常数，用于调整阈值。
                 负值会使更多像素被归类为前景（血管）。
    
    Returns:
        np.ndarray: 二值血管图，0为背景，255为血管。
    """
    # 确保输入是单通道
    if len(response_map.shape) == 3:
        raise ValueError("输入响应图必须是单通道图像。")
    
    # 将响应图归一化到 [0, 255] 范围并转换为 uint8
    # 这是 OpenCV 自适应阈值函数的要求
    normalized = cv2.normalize(response_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 应用高斯加权自适应阈值
    # cv2.THRESH_BINARY: 像素值 > 阈值 -> 255 (白色/血管), 否则 -> 0 (黑色/背景)
    binary_vessels = cv2.adaptiveThreshold(
        src=normalized,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权均值
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,  # 邻域大小
        C=c  # 常数偏移
    )
    
    return binary_vessels

def remove_small_components(binary_image: np.ndarray, min_size: int = 200) -> np.ndarray:
    """
    移除二值图像中小于指定大小的连通区域。
    
    Args:
        binary_image (np.ndarray): 输入的二值图像（0和255）
        min_size (int): 最小连通区域大小（像素数）
        
    Returns:
        np.ndarray: 移除小连通区域后的二值图像
    """
    # 确保输入是二值图像
    binary = (binary_image > 0).astype(np.uint8)
    
    # 找到所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # 创建输出图像
    output = np.zeros_like(binary_image)
    
    # 遍历所有连通区域（跳过背景标签0）
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            # 保留大于等于min_size的区域
            output[labels == label] = 255
    
    return output

# --- 完整流程示例 ---
if __name__ == "__main__":
    # 假设我们有一个经过改进顶帽滤波器处理后的响应图 `response`
    # 这里用一个模拟的响应图来演示
    H, W = 512, 512
    simulated_response = np.random.rand(H, W) * 10  # 模拟低强度背景
    # 添加一些高响应的“血管”区域
    cv2.line(simulated_response, (100, 100), (400, 400), 50, 3)
    cv2.line(simulated_response, (256, 50), (256, 450), 60, 2)
    
    # 应用自适应阈值
    binary_result = adaptive_threshold_vessel_map(
        response_map=simulated_response,
        block_size=21,  # 根据血管粗细调整，通常15-31
        c=-3            # 负值以增强弱血管的检测
    )
    
    print(f"Binary vessel map shape: {binary_result.shape}")
    print(f"Unique values: {np.unique(binary_result)}") # 应为 [0, 255]
