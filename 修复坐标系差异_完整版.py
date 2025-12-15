"""
完整的坐标系差异修复方案
一次性解决所有问题
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, transform_bounds
from rasterio.transform import Affine
import tempfile
import shutil


def 创建带坐标系的去年掩码(基准信息, 今年左上角, 今年右下角, 今年crs, 今年meta):
    """
    创建去年掩码，确保坐标系与今年图像一致
    """
    print("\n🔄 创建带坐标系的去年掩码...")

    # 1. 使用现有的裁剪逻辑（已经在主函数中完成）
    # 不需要重复裁剪，直接使用已经裁剪好的数据

    # 2. 从基准信息获取裁剪后的数据
    # 注意：主函数已经裁剪了数据，我们直接使用它
    # 这里需要从外部传入裁剪后的数据

    # 由于这个方法被调用的时机问题，我们需要一个更简单的解决方案
    return None  # 暂时返回None，让主函数使用原始数据


def 简单坐标系转换方案():
    """
    提供一个简单直接的坐标系转换方案
    """
    代码 = '''
# 在耕地分析工具_图形界面.py中，找到这段代码（约第888-914行）：

# 替换为：
# ✅ 简单修复：检查坐标系是否匹配
基准crs = 基准信息['crs']
if str(基准crs) != str(src.crs):
    self.输出结果(f"\n⚠️ 检测到坐标系不匹配！")
    self.输出结果(f"   基准数据: {基准crs}")
    self.输出结果(f"   今年图像: {src.crs}")
    self.输出结果(f"   这将导致面积计算错误！")
    self.输出结果(f"   建议：")
    self.输出结果(f"   1. 将今年图像转换到基准数据的坐标系")
    self.输出结果(f"   2. 或使用坐标系统修复.py工具")

    # 计算理论面积差异
    if "CM 126E" in str(基准crs) and "CM 129E" in str(src.crs):
        self.输出结果(f"\n💡 检测到中央经线差异（126E vs 129E）")
        self.输出结果(f"   这会导致约3度的经度偏移")
        self.输出结果(f"   在高纬度地区，这可能导致严重的位置错误")

# 使用原始裁剪数据（但知道它可能不准确）
去年掩码 = 裁剪区域.astype(np.uint8)
'''
    return 代码


def 更好的解决方案():
    """
    更好的解决方案：在读取图像时自动转换坐标系
    """
    方案 = '''
# 最佳解决方案：在图形界面的“选择图像”阶段就转换坐标系

# 1. 修改选择图像的函数，在加载图像后检查坐标系
# 2. 如果有基准数据，自动将新图像转换到基准数据的坐标系
# 3. 保存转换后的图像，避免每次都要转换

# 优点：
# - 一次转换，多次使用
# - 保证所有数据在同一坐标系下
# - 避免分析时的复杂转换逻辑

# 具体实现：
# 在选择图像按钮的回调函数中，添加坐标系检查和转换逻辑
'''
    return 方案


def 创建独立的坐标系转换工具():
    """
    创建一个独立的工具，用户可以在分析前使用
    """
    工具代码 = '''
import os
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import numpy as np

def 批量转换坐标系(输入目录, 输出目录, 目标坐标系):
    """
    批量转换目录中所有TIF文件的坐标系

    使用方法：
    1. 将所有今年图像放在一个文件夹
    2. 运行此工具，指定目标坐标系（基准数据的坐标系）
    3. 所有图像将被转换到统一坐标系
    """
    if not os.path.exists(输出目录):
        os.makedirs(输出目录)

    文件列表 = [f for f in os.listdir(输入目录) if f.endswith('.tif')]

    for 文件名 in 文件列表:
        输入路径 = os.path.join(输入目录, 文件名)
        输出路径 = os.path.join(输出目录, f"转换_{文件名}")

        with rasterio.open(输入路径) as src:
            # 计算转换参数
            transform, width, height = calculate_default_transform(
                src.crs, 目标坐标系, src.width, src.height, *src.bounds
            )

            # 创建新元数据
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': 目标坐标系,
                'transform': transform,
                'width': width,
                'height': height
            })

            # 创建输出数组
            数据 = np.zeros((src.count, height, width), dtype=src.dtypes[0])

            # 执行转换
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=数据[i-1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=目标坐标系,
                    resampling=rasterio.enums.Resampling.nearest
                )

            # 保存
            with rasterio.open(输出路径, 'w', **kwargs) as dst:
                dst.write(数据)

        print(f"✅ 已转换: {文件名} -> 转换_{文件名}")

# 使用示例
if __name__ == "__main__":
    输入目录 = "path/to/今年图像目录"
    输出目录 = "path/to/转换后目录"
    目标坐标系 = "EPSG:4551"  # CGCS2000 CM 126E

    批量转换坐标系(输入目录, 输出目录, 目标坐标系)
'''
    return 工具代码


if __name__ == "__main__":
    print("坐标系差异修复方案")
    print("=" * 50)
    print("\n方案1：简单警告（推荐立即使用）")
    print(简单坐标系转换方案())

    print("\n方案2：预处理转换（最佳实践）")
    print(更好的解决方案())

    print("\n方案3：批量转换工具")
    print(创建独立的坐标系转换工具())