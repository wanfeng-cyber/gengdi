"""
快速转换图像坐标系
将CM 129E的图像转换到CM 126E
"""

import os
import rasterio
from rasterio.warp import reproject, calculate_default_transform

def 转换图像(输入路径, 输出路径=None):
    """转换图像坐标系从CM 129E到CM 126E"""

    if 输出路径 is None:
        dir_name = os.path.dirname(输入路径)
        base_name = os.path.basename(输入路径)
        name, ext = os.path.splitext(base_name)
        输出路径 = os.path.join(dir_name, f"{name}_CM126E{ext}")

    try:
        with rasterio.open(输入路径) as src:
            print(f"\n正在转换: {base_name}")
            print(f"  原始坐标系: {src.crs}")

            # 目标坐标系
            from rasterio.crs import CRS
            目标crs = CRS.from_epsg(4551)  # CGCS2000 CM 126E

            if str(src.crs) == str(目标crs):
                print(f"  ✅ 已经是CM 126E，无需转换")
                return 输入路径

            print(f"  目标坐标系: {目标crs}")

            # 计算转换参数
            transform, width, height = calculate_default_transform(
                src.crs, 目标crs, src.width, src.height, *src.bounds
            )

            # 创建新元数据
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': 目标crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # 创建输出数组
            数据 = np.zeros((src.count, height, width), dtype=src.dtypes[0])

            # 执行坐标转换
            print(f"  🔄 正在转换...")
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=数据[i-1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=目标crs,
                    resampling=rasterio.enums.Resampling.nearest
                )

            # 保存转换后的文件
            with rasterio.open(输出路径, 'w', **kwargs) as dst:
                dst.write(数据)

            print(f"  ✅ 已保存: {os.path.basename(输出路径)}")
            print(f"  新尺寸: {width}x{height}")
            print(f"  新分辨率: {abs(transform.a):.6f} 米/像素")

            return 输出路径

    except Exception as e:
        print(f"  ❌ 转换失败: {str(e)}")
        return None

if __name__ == "__main__":
    import numpy as np
    import glob

    print("=" * 60)
    print("快速转换工具 - CM 129E → CM 126E")
    print("=" * 60)

    # 查找当前目录的TIF文件
    文件列表 = glob.glob("*.tif") + glob.glob("*.tiff")

    if not 文件列表:
        print("\n⚠️ 当前目录没有找到TIF文件")
        print("请将需要转换的TIF文件放在当前目录")
        input("\n按回车键退出...")

    print(f"\n找到 {len(文件列表)} 个TIF文件:")
    for i, f in enumerate(文件列表, 1):
        print(f"  {i}. {f}")

    # 转换所有文件
    for 文件路径 in 文件列表:
        转换图像(文件路径)

    print("\n" + "=" * 60)
    print("转换完成！")
    print("转换后的文件会添加 '_CM126E' 后缀")
    print("=" * 60)