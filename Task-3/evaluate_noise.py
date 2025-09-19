#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import convolve


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


original_dir = "/Users/duqiu/Desktop/焦点计划/JotangRecrument-main/ML/task_3/image_pairs/original"
degraded_dir = "/Users/duqiu/Desktop/焦点计划/JotangRecrument-main/ML/task_3/image_pairs/blurred"

# 创建结果保存目录
result_dir = "noise_evaluation_results"
os.makedirs(result_dir, exist_ok=True)

def load_image_pairs(original_dir, degraded_dir):
    #从指定目录中读取并通过文件名匹配原始图像与退化图像
    pairs = []
    original_files = {f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    degraded_files = {f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    
    common_files = original_files & degraded_files
    for filename in common_files:
        orig_path = os.path.join(original_dir, filename)
        deg_path = os.path.join(degraded_dir, filename)
        #拼接图像的完整路径（目录名+文件名），确保能够精准定位
        orig = cv2.imread(orig_path)
        deg = cv2.imread(deg_path)
        #读取图像并返回一个numpy数组；BGR格式，而不是RGB格式
        if orig is not None and deg is not None:
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            deg_gray = cv2.cvtColor(deg, cv2.COLOR_BGR2GRAY)
            #将BGR彩色图转化为单通道灰度图；
            pairs.append((filename, orig_gray, deg_gray))
            #存储图像对
    
    print(f"Found {len(pairs)} matching image pairs")
    return pairs

def calculate_noise_residual(original, degraded):
    #计算退化图像与原始图像的噪声值差异，即计算噪声残差
    if original.shape != degraded.shape:
        degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
    return np.int16(degraded) - np.int16(original)

def analyze_gaussian_noise(residual, original):
    #使用Shapiro-Wilk检验高斯噪声特性；
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    #定义一个拉普拉斯边缘检测核
    edges = convolve(original, edge_kernel)
    #对原始图像进行卷积计算得到边缘强度图
    edge_mask = edges < 10  # 创建掩码
    
    flat_residual = residual[edge_mask].flatten()
    if len(flat_residual) < 1000:
        return None, None, None
    
    # Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(flat_residual[:10000])
    
    mean = np.mean(flat_residual)
    var = np.var(flat_residual)
    
    return mean, var, p_value

def analyze_salt_pepper_noise(residual, threshold=200):
    #分析图像中是否存在椒盐噪声并量化其特征；
    salt_mask = residual > threshold
    pepper_mask = residual < -threshold
    
    total_pixels = residual.size
    salt_ratio = np.sum(salt_mask) / total_pixels
    pepper_ratio = np.sum(pepper_mask) / total_pixels
    total_extreme_ratio = salt_ratio + pepper_ratio
    
    # 计算空间自相关指数，判定椒盐噪声的空间分布是否随机
    moran_i = calculate_moran_i(salt_mask | pepper_mask)
    
    return salt_ratio, pepper_ratio, total_extreme_ratio, moran_i

def calculate_moran_i(mask):
    #计算空间自相关指数，看噪声的空间分布
    mask = mask.astype(np.float32)
    n = mask.size
    mean = np.mean(mask)
    
    if np.sum((mask - mean) ** 2) == 0:
        return 0  # 无噪声像素
    
    # 计算四领域权重
    weights = np.zeros_like(mask)
    weights[1:-1, 1:-1] = (
        mask[:-2, 1:-1] + mask[2:, 1:-1] +
        mask[1:-1, :-2] + mask[1:-1, 2:]
    )#每个非边缘像素的权重值等于周围四个相邻像素的掩码之和
    
    moran = (n / np.sum(weights)) * np.sum(weights * (mask - mean) * (mask - mean)) / np.sum((mask - mean) ** 2)
    #空间加权协方差与总体方差的比值
    return moran
    #接近0说明噪声噪声像素在空间上呈现随机分布，接近1说明噪声像素存在聚集性，接近-1呈现规则分布；

def visualize_noise_analysis(filename, original, degraded, residual, mean, var, p_value, salt_ratio, pepper_ratio, moran_i):
    #可视化噪声分析结果
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Noise Analysis: {filename}', fontsize=16)
    
    #对于初始图像
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    #对于退化图像
    axes[0, 1].imshow(degraded, cmap='gray')
    axes[0, 1].set_title('Degraded')
    axes[0, 1].axis('off')
    
    #残差图的可视化
    residual_normalized = (residual - np.min(residual)) / (np.max(residual) - np.min(residual))
    axes[0, 2].imshow(residual_normalized, cmap='jet')
    axes[0, 2].set_title('Noise Residual')
    axes[0, 2].axis('off')
    
    #绘制残差直方图
    flat_residual = residual.flatten()
    axes[1, 0].hist(flat_residual, bins=50, density=True, alpha=0.6, color='b')
    
    #高斯分布拟合曲线绘制
    if mean is not None and var is not None:
        x = np.linspace(np.min(flat_residual), np.max(flat_residual), 100)
        gaussian = stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))
        axes[1, 0].plot(x, gaussian, 'r-', lw=2, label=f'Gaussian Fit (μ={mean:.2f}, σ={np.sqrt(var):.2f})')
    axes[1, 0].set_title('Residual Histogram')
    axes[1, 0].legend()
    
    #展示统计信息（注意使用默认的英文表示）
    axes[1, 1].axis('off')
    stats_text = [
        "Gaussian Noise Test:",
        f"  Mean (μ): {mean:.2f}" if mean is not None else "  Mean (μ): Not calculable",
        f"  Variance (σ²): {var:.2f}" if var is not None else "  Variance (σ²): Not calculable",
        f"  Shapiro-Wilk p-value: {p_value:.4f}" if p_value is not None else "  Shapiro-Wilk p-value: Not calculable",
        "  (p > 0.05 suggests Gaussian noise)",
        "",
        "Salt-and-Pepper Noise Test:",
        f"  Salt Ratio: {salt_ratio:.4%}",
        f"  Pepper Ratio: {pepper_ratio:.4%}",
        f"  Moran's I: {moran_i:.4f}",
        "  (Close to 0 indicates random distribution)"
    ]
    
    axes[1, 1].text(0.05, 0.95, '\n'.join(stats_text),
                    verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    #噪声类型的判定以及依据说明
    axes[1, 2].axis('off')
    decision_text = []
    if p_value is not None and p_value > 0.05 and var > 1:
        decision_text.append("Dominant Noise Type: Gaussian Noise")
    if (salt_ratio + pepper_ratio) > 0.001:
        decision_text.append("Dominant Noise Type: Salt-and-Pepper Noise")
    if (p_value is not None and p_value > 0.05 and var > 1) and (salt_ratio + pepper_ratio) > 0.001:
        decision_text.append("Dominant Noise Type: Mixed Noise")
    if not decision_text:
        decision_text.append("Dominant Noise Type: Not Identified")
    
    #噪声判别部分的展示
    decision_text.append("\nRationale:")
    if p_value is not None:
        decision_text.append(f"- Gaussian Noise: {'Likely' if p_value > 0.05 else 'Unlikely'} (p-value threshold 0.05)")
    decision_text.append(f"- Salt-and-Pepper Noise: {'Present' if (salt_ratio + pepper_ratio) > 0.001 else 'Absent'} (ratio threshold 0.1%)")
    
    axes[1, 2].text(0.05, 0.95, '\n'.join(decision_text),
                    verticalalignment='top', fontsize=12,
                    bbox=dict(facecolor='lightblue', alpha=0.8))
    
    # 保存结果
    save_path = os.path.join(result_dir, f'noise_analysis_{filename}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Analysis saved to: {save_path}")

def main():
    image_pairs = load_image_pairs(original_dir, degraded_dir)
    if not image_pairs:
        print("No matching image pairs found. Check paths.")
        return
    
    summary = []
    for filename, original, degraded in image_pairs:
        print(f"\nAnalyzing image: {filename}")
        
        residual = calculate_noise_residual(original, degraded)
        mean, var, p_value = analyze_gaussian_noise(residual, original)
        salt_ratio, pepper_ratio, total_extreme_ratio, moran_i = analyze_salt_pepper_noise(residual)
        
        visualize_noise_analysis(filename, original, degraded, residual, mean, var, p_value, salt_ratio, pepper_ratio, moran_i)
        
        summary.append({
            'filename': filename,
            'gaussian_p': p_value,
            'salt_ratio': salt_ratio,
            'pepper_ratio': pepper_ratio,
            'moran_i': moran_i
        })
    
    # 生成报告展示
    with open(os.path.join(result_dir, 'noise_analysis_summary.txt'), 'w') as f:
        f.write("Noise Analysis Summary Report\n")
        f.write("==============================\n\n")
        for item in summary:
            f.write(f"Image: {item['filename']}\n")
            f.write(f"  Gaussian p-value: {item['gaussian_p']:.4f}\n")
            f.write(f"  Salt Ratio: {item['salt_ratio']:.4%}\n")
            f.write(f"  Pepper Ratio: {item['pepper_ratio']:.4%}\n")
            f.write(f"  Moran's I: {item['moran_i']:.4f}\n\n")
        
        gaussian_count = sum(1 for item in summary if item['gaussian_p'] is not None and item['gaussian_p'] > 0.05)
        salt_pepper_count = sum(1 for item in summary if (item['salt_ratio'] + item['pepper_ratio']) > 0.001)
        
        f.write("Overall Conclusion:\n")
        f.write(f"- {gaussian_count}/{len(summary)} images suggest Gaussian noise\n")
        f.write(f"- {salt_pepper_count}/{len(summary)} images suggest salt-and-pepper noise\n")
        
        if gaussian_count > 0 and salt_pepper_count > 0:
            f.write("- Dataset contains mixed noise types\n")
        elif gaussian_count > 0:
            f.write("- Dataset primarily contains Gaussian noise\n")
        elif salt_pepper_count > 0:
            f.write("- Dataset primarily contains salt-and-pepper noise\n")

if __name__ == "__main__":
    main()
