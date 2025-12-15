"""
测试不同识别方法的精度
"""
import numpy as np
import rasterio
import cv2

def 测试图像识别(图像路径):
    print(f"\n测试图像: {图像路径}")

    with rasterio.open(图像路径) as src:
        # 读取图像
        图像数据 = src.read()
        if 图像数据.shape[0] <= 4:
            图像数据 = np.transpose(图像数据[:3], (1, 2, 0))

        # 归一化到0-1
        if 图像数据.max() > 1.0:
            图像数据 = 图像数据.astype(np.float32) / 255.0

        print(f"图像尺寸: {图像数据.shape}")
        print(f"数据范围: {图像数据.min():.3f} - {图像数据.max():.3f}")

        # 方法1：简单绿色阈值
        r, g, b = 图像数据[:,:,0], 图像数据[:,:,1], 图像数据[:,:,2]
        简单掩码 = ((g > r * 0.9) & (g > b * 0.9))
        print(f"\n方法1 - 简单绿色阈值:")
        print(f"  耕地像素: {np.sum(简单掩码):,}")
        print(f"  耕地比例: {np.mean(简单掩码)*100:.2f}%")

        # 方法2：HSV颜色空间
        hsv = cv2.cvtColor(图像数据, cv2.COLOR_RGB2HSV)
        # 绿色范围 (需要根据实际情况调整)
        下限 = np.array([35, 40, 40])
        上限 = np.array([85, 255, 255])
        hsv掩码 = cv2.inRange(hsv, 下限, 上限) > 0
        print(f"\n方法2 - HSV绿色范围:")
        print(f"  耕地像素: {np.sum(hsv掩码):,}")
        print(f"  耕地比例: {np.mean(hsv掩码)*100:.2f}%")

        # 方法3：增强的耕地检测（模拟您系统中的方法）
        # 基于NDVI或其他植被指数
        if 图像数据.shape[2] >= 4:
            nir = 图像数据[:,:,3]  # 近红外
            # 计算NDVI
            ndvi = (nir - r) / (nir + r + 1e-8)
            ndvi掩码 = ndvi > 0.2
            print(f"\n方法3 - NDVI (有近红外):")
            print(f"  耕地像素: {np.sum(ndvi掩码):,}")
            print(f"  耕地比例: {np.mean(ndvi掩码)*100:.2f}%")
        else:
            # 使用增强的RGB方法
            # 耕地通常是棕色到黄色
            棕色度 = (r - g) / (r + g + 1e-8)
            亮度 = (r + g + b) / 3
            rgb掩码 = (棕色度 > 0.1) & (亮度 > 0.2) & (亮度 < 0.8)
            print(f"\n方法3 - 增强RGB:")
            print(f"  耕地像素: {np.sum(rgb掩码):,}")
            print(f"  耕地比例: {np.mean(rgb掩码)*100:.2f}%")

        # 方法4：K-means聚类
        像素 = 图像数据.reshape(-1, 3)
        像素 = 像素[np.random.choice(像素.shape[0], 10000, replace=False)]

        # K-means聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(像素.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 找出最可能是耕地的聚类（通常是中亮度的颜色）
        亮度排序 = np.argsort(np.mean(centers, axis=1))
        耕地聚类 = 亮度排序[1]  # 取中间亮度

        # 应用聚类结果
        labels_all = cv2.kmeans(图像数据.reshape(-1, 3).astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1]
        kmeans掩码 = (labels_all.flatten() == 耕地聚类).reshape(图像数据.shape[:2])

        print(f"\n方法4 - K-means聚类:")
        print(f"  耕地像素: {np.sum(kmeans掩码):,}")
        print(f"  耕地比例: {np.mean(kmeans掩码)*100:.2f}%")

        # 保存结果对比
        print(f"\n结果对比:")
        print(f"  简单阈值: {np.sum(简单掩码):,} 像素")
        print(f"  HSV空间: {np.sum(hsv掩码):,} 像素")
        print(f"  增强方法: {np.sum(rgb掩码):,} 像素")
        print(f"  K-means: {np.sum(kmeans掩码):,} 像素")

if __name__ == "__main__":
    import glob

    # 查找TIF文件
    tif_files = glob.glob("*.tif") + glob.glob("*.tiff")

    if not tif_files:
        print("没有找到TIF文件！")
        print("请将TIF文件放在当前目录")
    else:
        print(f"找到 {len(tif_files)} 个TIF文件")

        for tif_file in tif_files[:3]:  # 只测试前3个
            try:
                测试图像识别(tif_file)
                print("-" * 60)
            except Exception as e:
                print(f"处理 {tif_file} 时出错: {e}")