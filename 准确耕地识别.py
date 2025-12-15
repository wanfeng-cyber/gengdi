"""
准确的耕地识别方法
结合多种方法，提高识别准确率
"""
import numpy as np
import cv2
import os

def 准确识别耕地(图像路径):
    """使用多方法融合识别耕地"""

    # 读取图像
    with rasterio.open(图像路径) as src:
        读取数据 = src.read()
        if 读取数据.shape[0] <= 4:
            图像数据 = np.transpose(读取数据[:3], (1, 2, 0))
        else:
            图像数据 = 读取_data[1:4]  # 假设RGB是前三个波段

        # 统一归一化处理
        print(f"原始数据范围: {图像数据.min():.3f} - {图像数据.max():.3f}")

        # 如果是整数类型（0-255），归一化到0-1
        if 图像数据.dtype == np.uint8:
            图像数据 = 图像数据.astype(np.float32) / 255.0
        # 如果是浮点但超过1，也归一化
        elif 图像数据.max() > 1.0:
            图像数据 = (图像数据 - 图像数据.min()) / (图像数据.max() - 图像数据.min())

        # 确保在0-1范围内
        图像数据 = np.clip(图像数据, 0.0, 1.0)
        print(f"处理后范围: {图像数据.min():.3f} - {图像数据.max():.3f}")

        # 方法1：基于植被指数的识别
        r, g, b = 图像数据[:,:,0], 图像数据[:,:,1], 图像数据[:,:,2]

        # 过度绿色指数 (Excess Green Index)
        exg = 2 * g - r - b
        exg_归一化 = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
        方法1掩码 = (exg_归一化 > 0.1).astype(np.uint8)  # 阈值可调

        # 形态学处理填充空洞
        方法1掩码 = cv2.morphologyEx(方法1掩码,
                                      cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        # 方法2：颜色比值分析
        # 耕地通常有特定的RGB比例
        # 春夏秋季：绿色为主
        # 冬季/收割后：棕色/黄色为主

        # 绿色植被指数
        绿度指数 = g / (r + g + b + 1e-8)
        方法2a掩码 = (绿度指数 > 0.35).astype(np.uint8)

        # 棕色土壤指数（适合裸露或休耕地）
        差值 = r - g
        亮度 = (r + g + b) / 3
        方法2b掩码 = ((差值 > 0.05) & (亮度 > 0.2) & (亮度 < 0.7)).astype(np.uint8)

        # 合并方法2
        方法2掩码 = cv2.bitwise_or(方法2a掩码, 方法2b掩码)

        # 方法融合：加权平均
        # 根据季节或图像特征调整权重
        绿色比例 = np.sum(方法1掩码) / 方法1掩码.size
        棕色比例 = np.sum(方法2b掩码) / 方法2掩码.size

        print(f"绿色植被占比: {绿色比例*100:.1f}%")
        print(f"棕色土壤占比: {棕色比例*100:.1f}%")

        # 动态权重
        if 绿色比例 > 0.3:
            # 生长季节，绿色为主
            最终掩码 = (方法1掩码 * 0.6 +
                       方法2掩码 * 0.4).astype(np.uint8)
        else:
            # 收获或休耕季节，棕色为主
            最终掩码 = (方法1掩码 * 0.3 +
                       方法2掩码 * 0.7).astype(np.uint8)

        # 后处理：去除小噪点
        标签数, 标签, 统计, _ = cv2.connectedComponentsWithStats(
            最终掩码, connectivity=8)

        # 过滤太小的区域（噪声）
        最小面积 = 500  # 像素
        清洁掩码 = np.zeros_like(最终掩码)

        for i in range(1, 标签数):
            面积 = 统计[i, cv2.CC_STAT_AREA]
            if 面积 >= 最小面积:
                清洁掩码[标签 == i] = 1

        # 统计结果
        总像素 = 清洁掩码.size
        耕地像素 = np.sum(清洁掩码)
        耕地比例 = 耕地像素 / 总像素

        print(f"\n识别结果:")
        print(f"  总像素: {总像素:,}")
        print(f"  耕地像素: {耕地像素:,}")
        print(f"  耕地比例: {耕地比例*100:.2f}%")

        return 清洁掩码

if __name__ == "__main__":
    import rasterio

    print("=" * 60)
    print("准确耕地识别测试")
    print("=" * 60)

    # 测试所有TIF文件
    import glob
    tif_files = glob.glob("*.tif") + glob.glob("*.tiff")

    for tif_file in tif_files:
        print(f"\n处理文件: {tif_file}")
        print("-" * 40)

        try:
            掩码 = 准确识别耕地(tif_file)

            # 保存结果可视化
            if 掩码 is not None:
                # 创建可视化
                可视化 = np.zeros((掩码.shape[0], 掩码.shape[1], 3), dtype=np.uint8)
                可视化[掩码 == 1] = [0, 255, 0]  # 绿色表示耕地

                # 读取原图并调整大小
                with rasterio.open(tif_file) as src:
                    原图 = src.read()
                    if 原图.shape[0] <= 4:
                        原图 = np.transpose(原图[:3], (1, 2, 0))

                    # 调整原图大小匹配掩码
                    if 原图.shape[:2] != 掩码.shape:
                        原图 = cv2.resize(原图, (掩码.shape[1], 掩码.shape[0]))

                    # 半透明叠加
                cv2.addWeighted(原图, 0.7, 可视化, 0.3, 0, 可视化)

                output_file = tif_file.replace('.tif', '_耕地识别.png')
                cv2.imwrite(output_file, cv2.cvtColor(可视化, cv2.COLOR_RGB2BGR))
                print(f"  可视化保存到: {output_file}")

        except Exception as e:
            print(f"  错误: {str(e)}")
            import traceback
            traceback.print_exc()