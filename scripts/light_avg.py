import cv2
import numpy as np

def uneven_illumination_correction(image: np.ndarray, window_size: int = 15, c: float = 1e-6) -> np.ndarray:
    """
    对输入图像进行光照不均校正。
    
    矫正公式：
    Ieq(x, y) = Is(x, y) / (Iav(x, y) + c)
    其中 Iav 是通过窗口均值滤波器计算得到的局部平均强度。
    
    Args:
        image (np.ndarray): 输入图像，建议为单通道灰度图 (H, W) 或 (H, W, 1)。
        window_size (int): 均值滤波器的窗口大小 (N x N)。必须为正奇数。
        c (float): 防止除零的小常数。
    
    Returns:
        np.ndarray: 校正后的图像，数据类型为 float32，范围通常大于 [0, 1]。
    """
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")
    
    # 确保输入是浮点类型以进行精确计算
    img_float = image.astype(np.float32)
    
    # 使用 OpenCV 的 boxFilter 计算局部平均强度 Iav
    # normalize=True 表示进行平均（即均值滤波）
    Iav = cv2.boxFilter(img_float, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # 执行光照均衡化公式
    Ieq = img_float / (Iav + c)
    
    return Ieq

if __name__ == "__main__":
    # --- 调试示例 ---
    # 创建一个模拟的、带有光照不均的测试图像
    H, W = 512, 512
    # 创建一个从左上到右下逐渐变暗的背景
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    background = 200 * (1 - (X + Y) / 2)
    
    # 添加一些模拟的“血管”结构（亮线）
    vessels = np.zeros_like(background)
    cv2.line(vessels, (50, 50), (400, 300), 255, 2)
    cv2.line(vessels, (100, 400), (450, 100), 255, 2)
    
    # 合成最终的测试图像
    test_image = np.clip(background + vessels, 0, 255).astype(np.uint8)
    
    # 应用光照不均校正
    corrected = uneven_illumination_correction(test_image, window_size=31, c=1e-6)
    
    print(f"Original image dtype: {test_image.dtype}, shape: {test_image.shape}")
    print(f"Corrected image dtype: {corrected.dtype}, shape: {corrected.shape}")
    print(f"Corrected image value range: [{corrected.min():.2f}, {corrected.max():.2f}]")
    