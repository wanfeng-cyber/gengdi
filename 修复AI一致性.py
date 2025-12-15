"""
修复AI模型识别的一致性问题
确保相同图像产生相同结果
"""

import numpy as np
import tensorflow as tf
import rasterio
import cv2
import os

def 设置固定随机种子():
    """设置所有随机种子，确保结果一致"""
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def 测试模型一致性(模型路径, 图像路径, 测试次数=3):
    """测试相同图像多次识别的一致性"""

    print("=" * 60)
    print("测试AI模型识别一致性")
    print("=" * 60)

    # 设置固定随机种子
    设置固定随机种子()

    # 加载模型
    print(f"\n加载模型: {os.path.basename(模型路径)}")
    model = tf.keras.models.load_model(模型_path, compile=False)

    # 读取并预处理图像
    print(f"\n读取图像: {os.path.basename(图像路径)}")
    with rasterio.open(图像路径) as src:
        读取数据 = src.read()
        if 读取数据.shape[0] <= 4:
            图像 = np.transpose(读取_data[:3], (1, 2, 0))
        else:
            图像 = 读取_data[1:4]

        # 记录原始尺寸
        原始尺寸 = 图像.shape[:2]
        print(f"图像尺寸: {原始尺寸}")

        # 归一化
        if 图像.max() > 1.0:
            图像 = 图像.astype(np.float32) / 255.0
        图像 = np.clip(图像, 0.0, 1.0)
        print(f"数据范围: {图像.min():.3f} - {图像.max():.3f}")

    # 调整到模型输入尺寸
    输入尺寸 = model.input_shape[1]
    print(f"模型输入尺寸: {输入尺寸}x{输入尺寸}")

    # 多次测试
    所有结果 = []

    for i in range(测试次数):
        print(f"\n--- 第 {i+1} 次测试 ---")

        # 重新设置随机种子（很重要！）
        设置固定随机种子()

        # 预处理
        if 原始尺寸[0] != 输入尺寸 or 原始尺寸[1] != 输入尺寸:
            调整图像 = cv2.resize(图像, (输入尺寸, 输入尺寸))
        else:
            调整图像 = 图像

        # 添加batch维度
        输入图像 = np.expand_dims(调整图像, axis=0)

        # 预测
        预测结果 = model.predict(输入图像, verbose=0)[0]

        # 二值化 - 使用固定阈值
        阈值 = 0.5
        二值结果 = (预测结果 > 阈值).astype(np.uint8)

        # 调整回原始尺寸
        if 原始尺寸 != (输入尺寸, 输入尺寸):
            二值结果 = cv2.resize(二值结果,
                                     (原始尺寸[1], 原始尺寸[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # 统计结果
        耕地像素 = np.sum(二值结果)
        总像素 = 二值结果.size
        耕地比例 = 耕地像素 / 总像素

        print(f"  耕地像素: {耕地像素:,}")
        print(f"  耕地比例: {耕地比例*100:.2f}%")

        所有结果.append({
            '像素数': 耕地像素,
            '比例': 耕地比例
        })

    # 分析一致性
    print(f"\n一致性分析 (测试{测试次数}次):")
    print("-" * 40)

    像素数列表 = [r['像素数'] for r in 所有结果]
    比例列表 = [r['比例'] for r in 所有结果]

    print(f"耕地像素数: {像素数列表}")
    print(f"最大差异: {max(像素数列表) - min(像素数列表)} 像素")
    print(f"标准差: {np.std(像素数列表):.2f} 像素")

    print(f"\n耕地比例: {[f'{p*100:.2f}%' for p in 比例列表]}")
    print(f"最大差异: {max(比例列表) - min(比例列表):.4%}")
    print(f"标准差: {np.std(比例列表):.4%}")

    # 判断是否一致
    最大像素差异 = max(像素数列表) - min(像素数列表)
    相对差异 = 最大像素差异 / np.mean(像素数列表)

    if 相对差异 < 0.001:  # 0.1%
        print("\n✅ 结果完全一致！")
    elif 相对差异 < 0.01:  # 1%
        print("\n✅ 结果基本一致（差异<1%）")
    elif 相对差异 < 0.05:  # 5%
        print("\n⚠️ 结果存在差异（差异<5%）")
    else:
        print(f"\n❌ 结果差异较大（差异: {相对差异*100:.2f}%）")

    # 显示改进建议
    print("\n改进建议:")
    if 相对差异 > 0:
        print("1. 检查图像预处理是否一致")
        print("2. 使用更高的二值化阈值（如0.6）减少边界不确定性")
        print("3. 在预测后添加形态学后处理")
        print("4. 考虑使用固定dropout或禁用dropout")

    return 所有结果

if __name__ == "__main__":
    # 设置参数
    模型路径 = "耕地识别模型.h5"

    # 查找TIF文件
    import glob
    tif_files = glob.glob("*.tif") + glob.glob("*.tiff")

    if not tif_files:
        print("没有找到TIF文件！")
        exit()

    if not os.path.exists(模型路径):
        print(f"模型文件不存在: {模型路径}")
        exit()

    # 测试第一个文件
    测试图像路径 = tif_files[0]
    print(f"测试图像: {测试图像路径}")

    # 运行测试
    结果 = 测试模型一致性(模型路径, 测试图像路径)