"""
测试基础的颜色识别
不依赖复杂的库
"""
import numpy as np
import glob

def 基础颜色测试():
    print("=" * 60)
    print("基础颜色识别测试")
    print("=" * 60)

    # 查找TIF文件
    tif_files = glob.glob("*.tif") + glob.glob("*.tiff")

    if not tif_files:
        print("没有找到TIF文件！")
        return

    # 使用前两个文件进行测试
    test_files = tif_files[:2]

    results = []

    for i, tif_file in enumerate(test_files):
        print(f"\n{'='*20} 测试图像 {i+1}: {tif_file} {'='*20}")

        try:
            # 尝试使用rasterio读取
            try:
                import rasterio
                with rasterio.open(tif_file) as src:
                    图像数据 = src.read()
                    if 图像数据.shape[0] <= 4:
                        图像 = np.transpose(图像数据[:3], (1, 2, 0))
                    else:
                        图像 = 图像数据[1:4]

                    print(f"使用rasterio读取成功")
            except Exception as e:
                print(f"rasterio读取失败: {e}")
                continue

            # 归一化
            if 图像.max() > 1.0:
                图像 = 图像.astype(np.float32) / 255.0

            print(f"图像尺寸: {图像.shape}")
            print(f"数据范围: {图像.min():.3f} - {图像.max():.3f}")

            # 简单的颜色分析
            h, w = 图像.shape[:2]
            测试区域 = 图像[h//4:3*h//4, w//4:3*w//4]

            # 转换为0-255范围
            if 测试区域.max() <= 1.0:
                块 = (测试区域 * 255).astype(np.uint8)
            else:
                块 = 测试区域.astype(np.uint8)

            # 分离RGB通道
            R, G, B = 块[:,:,0].astype(np.float32), 块[:,:,1].astype(np.float32), 块[:,:,2].astype(np.float32)

            # 方法1：简单绿色检测
            绿色差异1 = G - R
            绿色差异2 = G - B
            绿色阈值 = 25
            green_mask = (绿色差异1 > 绿色阈值) & (绿色差异2 > 绿色阈值) & (G > 60)

            # 方法2：棕色检测
            棕色差异 = R - B
            棕色阈值 = 20
            brown_mask = (棕色差异 > 棕色阈值) & (R > 70) & (B < R * 0.8)

            # 方法3：黄色检测
            yellow_mask = (R > 100) & (G > 100) & (B < 120) & (np.abs(R - G) < 30)

            # 排除水体
            blue_mask = (B > R * 1.2) & (B > G * 1.2) & (B > 60)

            # 合并耕地掩码
            耕地掩码 = (green_mask | brown_mask | yellow_mask) & ~blue_mask

            # 统计结果
            总像素 = 测试区域.shape[0] * 测试区域.shape[1]
            绿色像素 = np.sum(green_mask)
            棕色像素 = np.sum(brown_mask)
            黄色像素 = np.sum(yellow_mask)
            水体像素 = np.sum(blue_mask)
            耕地像素 = np.sum(耕地掩码)

            print(f"\n颜色统计:")
            print(f"  绿色区域: {绿色像素:,} 像素 ({绿色像素/总像素*100:.1f}%)")
            print(f"  棕色区域: {棕色像素:,} 像素 ({棕色像素/总像素*100:.1f}%)")
            print(f"  黄色区域: {黄色像素:,} 像素 ({黄色像素/总像素*100:.1f}%)")
            print(f"  水体区域: {水体像素:,} 像素 ({水体像素/总像素*100:.1f}%)")
            print(f"\n耕地识别结果:")
            print(f"  耕地像素: {耕地像素:,}")
            print(f"  耕地比例: {耕地像素/总像素*100:.2f}%")

            # 计算颜色平均值
            耕地区域 = 耕地掩码
            if np.any(耕地区域):
                耕地R = np.mean(R[耕地区域])
                耕地G = np.mean(G[耕地区域])
                耕地B = np.mean(B[耕地区域])
                print(f"\n耕地平均颜色: R={耕地R:.1f}, G={耕地G:.1f}, B={耕地B:.1f}")

            results.append({
                'file': tif_file,
                'farmland_pixels': 耕地像素,
                'farmland_ratio': 耕地像素/总像素
            })

        except Exception as e:
            print(f"处理 {tif_file} 时出错: {e}")

    # 比较结果
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("结果对比:")
        print(f"{results[0]['file']}: {results[0]['farmland_ratio']*100:.2f}%")
        print(f"{results[1]['file']}: {results[1]['farmland_ratio']*100:.2f}%")

        if abs(results[0]['farmland_ratio'] - results[1]['farmland_ratio']) < 0.05:
            print("✅ 两张图像识别结果接近")
        else:
            print("⚠️ 两张图像识别结果差异较大")
    elif len(results) == 1:
        print("\n" + "=" * 60)
        print(f"单张图像结果: {results[0]['farmland_ratio']*100:.2f}%")

    print("\n" + "=" * 60)
    print("识别精度优化建议:")
    print("1. 如果耕地比例过高（>60%），说明阈值太宽松")
    print("2. 如果耕地比例过低（<5%），说明阈值太严格")
    print("3. 根据实际图像调整绿色阈值（当前:25）和棕色阈值（当前:20）")
    print("4. 考虑季节因素：春季绿色多，秋季棕色多")
    print("=" * 60)

if __name__ == "__main__":
    基础颜色测试()