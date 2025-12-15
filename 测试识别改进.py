"""
测试高精度颜色识别的改进效果
"""
import numpy as np
import cv2
import rasterio
from 高精度颜色识别 import 高精度颜色识别器
import glob
import time

def 测试识别改进():
    print("=" * 60)
    print("测试高精度颜色识别改进效果")
    print("=" * 60)

    # 查找TIF文件
    tif_files = glob.glob("*.tif") + glob.glob("*.tiff")

    if not tif_files:
        print("没有找到TIF文件！")
        return

    # 使用前两个文件进行测试
    test_files = tif_files[:2]

    # 创建识别器
    识别器 = 高精度颜色识别器()

    for i, tif_file in enumerate(test_files):
        print(f"\n{'='*20} 测试图像 {i+1}: {tif_file} {'='*20}")

        try:
            # 读取图像
            with rasterio.open(tif_file) as src:
                图像数据 = src.read()
                if 图像数据.shape[0] <= 4:
                    图像 = np.transpose(图像数据[:3], (1, 2, 0))
                else:
                    图像 = 图像数据[1:4]

                # 归一化
                if 图像.max() > 1.0:
                    图像 = 图像.astype(np.float32) / 255.0

                print(f"图像尺寸: {图像.shape}")
                print(f"数据范围: {图像.min():.3f} - {图像.max():.3f}")

                # 测试识别
                print("\n开始高精度识别...")
                start_time = time.time()

                # 使用中心区域进行测试
                h, w = 图像.shape[:2]
                测试块 = 图像[h//4:3*h//4, w//4:3*w//4]

                # 无去年数据的识别
                无去年结果 = 识别器.多方法融合(测试块)
                无去年像素 = np.sum(无去年结果 > 0.5)
                无去年比例 = np.mean(无去年结果 > 0.5)

                print(f"\n无去年数据参考:")
                print(f"  识别耕地像素: {无去年像素:,}")
                print(f"  耕地比例: {无去年比例*100:.2f}%")

                # 模拟有去年数据的识别
                假设去年掩码 = np.zeros_like(测试块[:,:,0])
                # 在部分区域假设是去年的耕地
                假设去年掩码[50:150, 50:150] = 1

                有去年结果 = 识别器.多方法融合(测试块, 假设去年掩码)
                有去年像素 = np.sum(有去年结果 > 0.5)
                有去年比例 = np.mean(有去年结果 > 0.5)

                print(f"\n有去年数据参考:")
                print(f"  识别耕地像素: {有去年像素:,}")
                print(f"  耕地比例: {有去年比例*100:.2f}%")

                # 测试一致性
                print("\n测试识别一致性...")
                一致性结果 = 识别器.多方法融合(测试块, 假设去年掩码)
                一致性像素 = np.sum(一致性结果 > 0.5)

                像素差异 = abs(有去年像素 - 一致性像素)
                相对差异 = 像素差异 / max(有去年像素, 1)

                print(f"  第一次识别: {有去年像素:,} 像素")
                print(f"  第二次识别: {一致性像素:,} 像素")
                print(f"  像素差异: {像素差异} 像素 ({相对差异*100:.2f}%)")

                end_time = time.time()
                print(f"\n识别耗时: {end_time - start_time:.2f} 秒")

                # 分析颜色分布
                print("\n颜色分布分析:")
                块_uint8 = (测试块 * 255).astype(np.uint8)
                R, G, B = 块_uint8[:,:,0], 块_uint8[:,:,1], 块_uint8[:,:,2]

                # 计算绿色区域
                green_pixels = np.sum((G > R * 1.2) & (G > B * 1.2))
                # 计算棕色区域
                brown_pixels = np.sum((R > G * 1.1) & (R > B * 1.1))
                # 计算蓝色区域（水体）
                blue_pixels = np.sum((B > R * 1.2) & (B > G * 1.2))

                total_pixels = 测试块.shape[0] * 测试块.shape[1]
                print(f"  绿色像素: {green_pixels:,} ({green_pixels/total_pixels*100:.1f}%)")
                print(f"  棕色像素: {brown_pixels:,} ({brown_pixels/total_pixels*100:.1f}%)")
                print(f"  蓝色像素: {blue_pixels:,} ({blue_pixels/total_pixels*100:.1f}%)")

        except Exception as e:
            print(f"处理 {tif_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("改进建议:")
    print("1. 如果识别耕地比例过高（>70%），说明阈值太宽松")
    print("2. 如果识别耕地比例过低（<10%），说明阈值太严格")
    print("3. 如果相同图像差异较大，检查随机种子或固定算法")
    print("4. 根据实际图像特征调整颜色阈值")
    print("=" * 60)

if __name__ == "__main__":
    测试识别改进()