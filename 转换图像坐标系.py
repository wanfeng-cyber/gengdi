"""
简单的坐标系转换工具
将今年图像转换到与基准数据相同的坐标系
"""

import os
import sys
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import numpy as np
from tkinter import filedialog, messagebox, Tk
import glob

def 转换单个文件(输入路径, 目标crs, 输出路径=None):
    """转换单个文件的坐标系"""
    if not os.path.exists(输入路径):
        print(f"❌ 文件不存在: {输入路径}")
        return None

    # 生成输出路径
    if 输出路径 is None:
        dir_name = os.path.dirname(输入路径)
        base_name = os.path.basename(输入路径)
        name, ext = os.path.splitext(base_name)
        输出路径 = os.path.join(dir_name, f"{name}_已转换{ext}")

    try:
        with rasterio.open(输入路径) as src:
            print(f"\n正在处理: {base_name}")
            print(f"  原始坐标系: {src.crs}")
            print(f"  目标坐标系: {目标crs}")

            # 检查是否需要转换
            if str(src.crs) == str(目标crs):
                print(f"  ✅ 坐标系已匹配，无需转换")
                return 输入路径

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
            return 输出路径

    except Exception as e:
        print(f"  ❌ 转换失败: {str(e)}")
        return None


def 批量转换目录(输入目录, 目标crs, 输出目录=None):
    """批量转换目录中的所有TIF文件"""
    if 输出目录 is None:
        输出目录 = os.path.join(输入目录, "已转换")

    if not os.path.exists(输出目录):
        os.makedirs(输出目录)

    # 查找所有TIF文件
    文件列表 = glob.glob(os.path.join(输入目录, "*.tif")) + \
              glob.glob(os.path.join(输入目录, "*.tiff"))

    if not 文件列表:
        print(f"⚠️ 在 {输入目录} 中没有找到TIF文件")
        return []

    print(f"\n找到 {len(文件列表)} 个文件待转换")

    成功列表 = []
    for 文件路径 in 文件列表:
        base_name = os.path.basename(文件路径)
        输出路径 = os.path.join(输出目录, base_name)

        结果 = 转换单个文件(文件路径, 目标crs, 输出路径)
        if 结果:
            成功列表.append(结果)

    print(f"\n✅ 批量转换完成: {len(成功列表)}/{len(文件列表)} 个文件成功转换")
    return 成功列表


def 图形界面选择():
    """使用图形界面选择文件"""
    root = Tk()
    root.withdraw()

    # 选择文件或目录
    选项 = messagebox.askyesno(
        "选择方式",
        "选择转换方式：\n\n"
        "【是】 - 转换单个文件\n"
        "【否】 - 批量转换整个目录"
    )

    if 选项:
        # 选择单个文件
        print("\n请选择要转换的TIF文件:")
        文件路径 = filedialog.askopenfilename(
            title="选择TIF文件",
            filetypes=[("TIF文件", "*.tif *.tiff"), ("所有文件", "*.*")]
        )
        if 文件路径:
            # 选择基准数据文件来获取目标坐标系
            print("\n请选择基准数据文件（用于获取目标坐标系）:")
            基准文件 = filedialog.askopenfilename(
                title="选择基准TIF文件",
                filetypes=[("TIF文件", "*.tif *.tiff"), ("所有文件", "*.*")]
            )
            if 基准文件:
                with rasterio.open(基准文件) as src:
                    目标crs = src.crs
                print(f"\n目标坐标系: {目标crs}")
                转换单个文件(文件路径, 目标crs)
    else:
        # 选择目录
        print("\n请选择要转换的目录:")
        输入目录 = filedialog.askdirectory(title="选择包含TIF文件的目录")
        if 输入目录:
            # 选择基准数据文件
            print("\n请选择基准数据文件（用于获取目标坐标系）:")
            基准文件 = filedialog.askopenfilename(
                title="选择基准TIF文件",
                filetypes=[("TIF文件", "*.tif *.tiff"), ("所有文件", "*.*")]
            )
            if 基准文件:
                with rasterio.open(基准文件) as src:
                    目标crs = src.crs
                print(f"\n目标坐标系: {目标crs}")
                批量转换目录(输入目录, 目标crs)

    root.destroy()


def main():
    print("=" * 60)
    print("TIF图像坐标系转换工具")
    print("=" * 60)
    print("\n使用说明：")
    print("1. 此工具将图像转换到指定的坐标系")
    print("2. 转换后的文件会添加'_已转换'后缀")
    print("3. 使用最近邻重采样，保持像素值不变")
    print("4. 建议在分析前先转换所有图像到统一坐标系")

    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        if len(sys.argv) == 3:
            输入路径 = sys.argv[1]
            目标crs = sys.argv[2]
            if os.path.isfile(输入路径):
                转换单个文件(输入路径, 目标crs)
            else:
                批量转换目录(输入路径, 目标crs)
        else:
            print("\n命令行用法:")
            print(f"  {sys.argv[0]} <输入文件/目录> <目标CRS>")
            print(f"  例如: {sys.argv[0]} image.tif EPSG:4551")
            print(f"  例如: {sys.argv[0]} ./images EPSG:4551")
    else:
        # 图形界面模式
        图形界面选择()

    print("\n转换完成！现在可以使用转换后的文件进行分析了。")


if __name__ == "__main__":
    main()