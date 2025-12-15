"""
坐标系统修复工具
处理不同坐标系之间的转换问题
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.crs import CRS
import cv2

def 检查并转换坐标系(输入文件路径, 目标_crs="EPSG:4526"):
    """
    检查文件的坐标系，如果需要则转换

    参数:
        输入文件路径: 输入TIF文件路径
        目标_crs: 目标坐标系，默认为CGCS2000 CM 126E (EPSG:4526)

    返回:
        转换后的数组和元数据
    """
    with rasterio.open(输入文件路径) as src:
        print(f"原始文件: {输入文件路径}")
        print(f"原始坐标系: {src.crs}")

        # 检查是否需要转换
        if src.crs == CRS.from_string(目标_crs):
            print("✅ 坐标系已匹配，无需转换")
            return src.read(), src.meta

        print(f"⚠️ 坐标系不匹配，需要转换到: {目标_crs}")

        # 计算转换参数
        transform, width, height = calculate_default_transform(
            src.crs,
            CRS.from_string(目标_crs),
            src.width,
            src.height,
            *src.bounds
        )

        # 创建新的元数据
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': CRS.from_string(目标_crs),
            'transform': transform,
            'width': width,
            'height': height
        })

        # 创建转换后的数组
        数组 = np.zeros((src.count, height, width), dtype=src.dtypes[0])

        # 执行坐标转换
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=数组[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=CRS.from_string(目标_crs),
                resampling=rasterio.enums.Resampling.nearest
            )

        print(f"✅ 坐标系转换完成")
        print(f"新尺寸: {width} x {height}")

        return 数组, kwargs

def 统一坐标系到基准数据(基准数据路径, 今年图像路径, 输出路径):
    """
    将今年图像转换到基准数据的坐标系

    参数:
        基准数据路径: 基准数据文件路径
        今年图像路径: 今年图像文件路径
        输出路径: 转换后的输出路径
    """
    # 1. 获取基准数据的坐标系
    with rasterio.open(基准数据路径) as src:
        基准_crs = src.crs
        print(f"基准数据坐标系: {基准_crs}")

    # 2. 转换今年图像到基准数据的坐标系
    try:
        转换后数组, 新元数据 = 检查并转换坐标系(
            今年图像路径,
            str(基准_crs)
        )

        # 3. 保存转换后的文件
        with rasterio.open(输出路径, 'w', **新元数据) as dst:
            dst.write(转换后数组)

        print(f"\n✅ 已保存转换后的文件: {输出路径}")
        print(f"坐标系已统一为: {基准_crs}")

        return 输出路径

    except Exception as e:
        print(f"\n❌ 转换失败: {str(e)}")
        return None

def 验证两个文件的对齐情况(文件1路径, 文件2路径):
    """
    验证两个文件是否对齐
    """
    with rasterio.open(文件1路径) as src1, rasterio.open(文件2路径) as src2:
        print(f"\n验证文件对齐情况:")
        print(f"文件1: {文件1路径}")
        print(f"  坐标系: {src1.crs}")
        print(f"  范围: {src1.bounds}")
        print(f"  分辨率: {abs(src1.transform.a):.6f} x {abs(src1.transform.e):.6f}")

        print(f"\n文件2: {文件2路径}")
        print(f"  坐标系: {src2.crs}")
        print(f"  范围: {src2.bounds}")
        print(f"  分辨率: {abs(src2.transform.a):.6f} x {abs(src2.transform.e):.6f}")

        # 检查是否对齐
        if src1.crs == src2.crs:
            print(f"\n✅ 坐标系一致")

            # 检查范围是否匹配
            bounds1 = src1.bounds
            bounds2 = src2.bounds

            # 允许小的误差
            误差阈值 = 0.001

            if (abs(bounds1.left - bounds2.left) < 误差阈值 and
                abs(bounds1.right - bounds2.right) < 误差阈值 and
                abs(bounds1.top - bounds2.top) < 误差阈值 and
                abs(bounds1.bottom - bounds2.bottom) < 误差阈值):
                print(f"✅ 地理范围一致")
                return True
            else:
                print(f"⚠️ 地理范围不完全一致")
                print(f"  范围差异:")
                print(f"    左: {abs(bounds1.left - bounds2.left):.6f}")
                print(f"    右: {abs(bounds1.right - bounds2.right):.6f}")
                print(f"    上: {abs(bounds1.top - bounds2.top):.6f}")
                print(f"    下: {abs(bounds1.bottom - bounds2.bottom):.6f}")
        else:
            print(f"\n❌ 坐标系不一致！")

        return False

def 修复耕地分析的坐标系问题(基准数据路径, 今年图像路径):
    """
    修复耕地分析中的坐标系问题
    """
    print("=" * 60)
    print("🔧 修复耕地分析坐标系问题")
    print("=" * 60)

    # 1. 检查原始文件对齐情况
    print("\n1️⃣ 检查原始文件:")
    验证两个文件的对齐情况(基准数据路径, 今年图像路径)

    # 2. 转换今年图像到基准数据坐标系
    print(f"\n2️⃣ 转换坐标系:")
    输出路径 = 今年图像路径.replace(".tif", "_转换到基准坐标系.tif")
    结果路径 = 统一坐标系到基准数据(基准数据路径, 今年图像路径, 输出路径)

    if 结果路径:
        # 3. 验证转换后的文件
        print(f"\n3️⃣ 验证转换后的文件:")
        验证两个文件的对齐情况(基准数据路径, 结果路径)

        print(f"\n" + "=" * 60)
        print(f"✅ 修复完成！")
        print(f"请使用转换后的文件进行分析: {结果路径}")
        print(f"这样应该能解决面积差异过大的问题。")
        print("=" * 60)

        return 结果路径
    else:
        print(f"\n❌ 转换失败，请检查文件是否存在或是否可读")
        return None


# 使用示例
if __name__ == "__main__":
    # 示例：修复坐标系问题
    基准数据路径 = "path/to/基准数据.tif"  # 替换为实际路径
    今年图像路径 = "path/to/今年图像.tif"    # 替换为实际路径

    if 基准数据路径 and 今年图像路径:
        修复耕地分析的坐标系问题(基准数据路径, 今年图像路径)
    else:
        print("请提供正确的文件路径")