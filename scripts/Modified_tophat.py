import cv2
import numpy as np

def create_line_se(length: int, angle_deg: float) -> np.ndarray:
    """
    创建一个指定长度和角度的线性结构元素（SE）。
    
    Args:
        length (int): SE的长度（必须为奇数）。
        angle_deg (float): SE的角度（度），0度为水平向右。
    
    Returns:
        np.ndarray: 二值化的结构元素。
    """
    if length % 2 == 0:
        raise ValueError("SE的长度必须是奇数。")
    
    # 创建一个足够大的空白画布
    canvas_size = length
    se = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # 计算中心点
    center = canvas_size // 2
    
    # 计算线段的两个端点
    angle_rad = np.deg2rad(angle_deg)
    dx = int(np.cos(angle_rad) * (length - 1) / 2)
    dy = int(np.sin(angle_rad) * (length - 1) / 2)
    
    pt1 = (center - dx, center - dy)
    pt2 = (center + dx, center + dy)
    
    # 在画布上绘制白色线条
    cv2.line(se, pt1, pt2, color=255, thickness=1)
    
    # 转换为二值SE (0 or 1)
    se = (se > 0).astype(np.uint8)
    return se

def improved_top_hat_transform_v2(image: np.ndarray, se_length: int = 15) -> np.ndarray:
    """
    实现文献中定义的改进顶帽滤波器:
    ITH_θ = I_eq - min( (I_eq • S_θ) ∘ S_θ , I_eq )
    
    使用8个不同方向的线性结构元素，并将所有方向的响应图进行逐像素取最大值融合。
    
    Args:
        image (np.ndarray): 输入图像，应为单通道灰度图 (H, W)。
        se_length (int): 线性结构元素的长度。
    
    Returns:
        np.ndarray: 融合后的最终响应图像。
    """
    # 确保输入是灰度图
    if len(image.shape) == 3:
        raise ValueError("输入图像必须是单通道灰度图。")
    
    img_float = image.astype(np.float32)
    H, W = img_float.shape
    
    # 初始化最大响应图
    max_response = np.zeros_like(img_float)
    
    # 定义8个方向 (0° 到 157.5°, 间隔22.5°)
    angles = np.arange(0, 180, 22.5)  # [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    
    for angle in angles:
        # 1. 创建线性结构元素
        se = create_line_se(se_length, angle)
        
        # 2. 执行闭运算: (I_eq • S_θ)
        closed = cv2.morphologyEx(img_float, cv2.MORPH_CLOSE, se)
        
        # 3. 对闭运算结果执行开运算: (I_eq • S_θ) ∘ S_θ
        opened_of_closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)
        
        # 4. 逐像素取最小值: min(opened_of_closed, I_eq)
        min_val = np.minimum(opened_of_closed, img_float)
        
        # 5. 计算改进的顶帽响应: ITH_θ = I_eq - min_val
        ith_response = img_float - min_val
        
        # 6. 更新全局最大响应
        max_response = np.maximum(max_response, ith_response)
    
    return max_response

if __name__ == "__main__":
    # --- 调试示例 ---
    # 创建一个模拟的测试图像
    H, W = 512, 512
    test_image = np.zeros((H, W), dtype=np.uint8)
    
    # 添加几条不同方向的线来模拟血管
    cv2.line(test_image, (100, 100), (400, 400), 255, 2)   # 45度
    cv2.line(test_image, (100, 400), (400, 100), 255, 2)   # 135度
    cv2.line(test_image, (256, 50), (256, 450), 255, 2)    # 90度（垂直）
    cv2.line(test_image, (50, 256), (450, 256), 255, 2)    # 0度（水平）
    
    # 应用改进的顶帽滤波器
    response = improved_top_hat_transform_v2(test_image, se_length=15)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Response shape: {response.shape}")
    print(f"Response value range: [{response.min():.2f}, {response.max():.2f}]")
    
    