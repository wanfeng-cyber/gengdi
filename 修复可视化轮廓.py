"""
快速修复可视化轮廓问题
"""

import numpy as np
import cv2
import rasterio
from rasterio.warp import transform, reproject, transform_bounds
from affine import Affine

def 智能可视化修复(耕地掩码, 去年掩码, 去年图像路径, 今年图像路径, 基准信息=None):
    """
    智能修复可视化，保持轮廓清晰
    """
    print("\n🔧 使用智能可视化修复...")

    try:
        # 读取图像信息
        with rasterio.open(去年图像路径) as src_去年:
            去年_shape = (src_去年.height, src_去年.width)
            去年_transform = src_去年.transform
            去年_crs = src_去年.crs

        with rasterio.open(今年图像路径) as src_今年:
            今年_shape = (src_今年.height, src_今年.width)
            今年_transform = src_今年.transform
            今年_crs = src_今年.crs

        print(f"  去年图像: {去年_shape}, CRS: {去年_crs}")
        print(f"  今年图像: {今年_shape}, CRS: {今年_crs}")
        print(f"  去年掩码: {去年掩码.shape}")
        print(f"  今年掩码: {耕地掩码.shape}")

        # 如果有基准信息，使用完整的基准地图
        if 基准信息 and '基准耕地地图' in 基准信息:
            print(f"\n  使用完整基准地图...")
            完整基准掩码 = 基准信息['基准耕地地图']
            print(f"  完整基准地图尺寸: {完整基准掩码.shape}")

            # 计算地理变换
            基准transform = Affine(
                基准信息['地理变换']['a'],
                基准信息['地理变换']['b'],
                基准信息['地理变换']['c'],
                基准信息['地理变换']['d'],
                基准信息['地理变换']['e'],
                基准信息['地理变换']['f']
            )
            基准crs = 基准信息['crs']

            # 创建可视化
            创建对比可视化(完整基准掩码, 基准transform, 基准crs,
                             耕地掩码, 今年_transform, 今年_crs,
                             去年_transform, 去年_crs, 今年_transform, 今年_crs)
        else:
            # 没有基准信息，使用传入的掩码
            print(f"\n  使用传入的掩码...")

            # 确保去年掩码尺寸正确
            if 去年掩码.shape != 去年_shape:
                print(f"  调整去年掩码尺寸: {去年掩码.shape} -> {去年_shape}")
                去年掩码调整 = cv2.resize(去年掩码, (去年_shape[1], 去年_shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                去年掩码调整 = 去年掩码

            创建对比可视化(去年掩码调整, 去年_transform, 去年_crs,
                             耕地掩码, 今年_transform, 今年_crs,
                             去年_transform, 去年_crs, 今年_transform, 今年_crs)

    except Exception as e:
        print(f"  ❌ 修复失败: {str(e)}")
        import traceback
        traceback.print_exc()


def 创建对比可视化(去年掩码, 去年_transform, 去年crs,
                   今年掩码, 今年_transform, 今年crs,
                   去年显示_transform, 去年显示_crs, 今年显示_transform, 今年显示_crs):
    """创建对比可视化图像"""

    # 读取图像数据（小尺寸用于显示）
    缩放因子 = 0.3  # 缩小到30%用于显示

    # 创建画布
    宽度 = 1200
    高度 = 600
    对比图像 = np.zeros((高度, 宽度, 3), dtype=np.uint8)

    # 左侧：去年图像 + 耕地轮廓
    左侧宽度 = 宽度 // 2
    左侧高度 = 高度

    # 右侧：今年图像 + 耕地轮廓
    右侧宽度 = 宽度 // 2
    右侧高度 = 高度

    # 添加分割线
    对比图像[:, 宽度//2:宽度//2+2, :] = [255, 255, 255]

    # 处理左侧（去年）
    if 去年掩码 is not None:
        # 缩放掩码
        左侧掩码 = cv2.resize(去年掩码, (左侧宽度, 左侧高度), interpolation=cv2.INTER_NEAREST)
        # 转换为3通道
        左侧掩码_3ch = cv2.cvtColor(左侧掩码, cv2.COLOR_GRAY2BGR)
        # 叠加到对比图像
        对比图像[:左侧_height, :左侧_width] = 左侧掩码_3ch * 255

    # 处理右侧（今年）
    if 今年掩码 is not None:
        # 缩放掩码
        右侧掩码 = cv2.resize(今年掩码, (右侧宽度, 右侧高度), interpolation=cv2.INTER_NEAREST)
        # 转换为3通道
        右侧掩码_3ch = cv2.cvtColor(右侧掩码, cv2.COLOR_GRAY2BGR)
        # 叠加到对比图像
        对比图像[:右侧_height, 左侧_width:] = 右侧掩码_3ch * 255

    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(对比图像, "去年耕地", (50, 50), font, 1.5, (255, 255, 255), 2)
    cv2.putText(对比图像, "今年耕地", (左侧宽度 + 50, 50), font, 1.5, (255, 255, 255), 2)

    # 保存图像
    cv2.imwrite("对比可视化_修复版.png", 对比图像)
    print(f"\n✅ 已保存修复版可视化: 对比可视化_修复版.png")

    # 显示图像（如果在支持的环境中）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(对比图像, cv2.COLOR_BGR2RGB))
        plt.title("耕地变化对比（修复版）")
        plt.axis('off')
        plt.show()
    except:
        print("  提示：已保存图像文件到当前目录")


# 导入到主程序使用
def 应用修复():
    """在主程序中应用修复"""
    print("应用智能可视化修复...")
    # 这个函数可以在主程序的适当位置调用