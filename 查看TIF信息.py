"""
查看TIF文件信息的工具
"""

import os
import numpy as np
from PIL import Image
import struct
import sys

def 读取tiff_header(文件路径):
    """读取TIFF文件头信息"""
    try:
        with open(文件路径, 'rb') as f:
            # 读取TIFF头
            f.seek(0)
            byte_order = f.read(2)
            if byte_order == b'II':
                # Little endian
                endian = '<'
            elif byte_order == b'MM':
                # Big endian
                endian = '>'
            else:
                return None

            # 读取版本号
            version = struct.unpack(endian + 'H', f.read(2))[0]
            if version != 42:
                return None

            # 读取第一个IFD偏移
            ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]

            # 读取IFD
            f.seek(ifd_offset)
            num_entries = struct.unpack(endian + 'H', f.read(2))[0]

            # 查找所需的标签
            标签信息 = {}
            for i in range(num_entries):
                tag = struct.unpack(endian + 'H', f.read(2))[0]
                类型 = struct.unpack(endian + 'H', f.read(2))[0]
                count = struct.unpack(endian + 'I', f.read(4))[0]
                value_offset = struct.unpack(endian + 'I', f.read(4))[0]

                # 保存重要的标签
                if tag == 256:  # ImageWidth
                    标签信息['宽度'] = value_offset if 类型 == 4 or 类型 == 3 else None
                elif tag == 257:  # ImageLength
                    标签信息['高度'] = value_offset if 类型 == 4 or 类型 == 3 else None
                elif tag == 282:  # XResolution
                    if 类型 == 5:  # Rational
                        保存位置 = f.tell()
                        f.seek(value_offset)
                        分子 = struct.unpack(endian + 'I', f.read(4))[0]
                        分母 = struct.unpack(endian + 'I', f.read(4))[0]
                        标签信息['X分辨率'] = 分子 / 分母
                        f.seek(保存位置)
                elif tag == 283:  # YResolution
                    if 类型 == 5:  # Rational
                        保存位置 = f.tell()
                        f.seek(value_offset)
                        分子 = struct.unpack(endian + 'I', f.read(4))[0]
                        分母 = struct.unpack(endian + 'I', f.read(4))[0]
                        标签信息['Y分辨率'] = 分子 / 分母
                        f.seek(保存位置)
                elif tag == 284:  # PlanarConfiguration
                    标签信息['平面配置'] = value_offset
                elif tag == 273:  # StripOffsets
                    标签信息['数据偏移'] = value_offset
                elif tag == 277:  # SamplesPerPixel
                    标签信息['样本数'] = value_offset
                elif tag == 278:  # RowsPerStrip
                    标签信息['每带行数'] = value_offset
                elif tag == 279:  # StripByteCounts
                    标签信息['带字节数'] = value_offset

            return 标签信息

    except Exception as e:
        print(f"读取错误: {e}")
        return None

def 使用PIL查看(文件路径):
    """使用PIL查看图像信息"""
    try:
        with Image.open(文件路径) as img:
            info = {
                '格式': img.format,
                '模式': img.mode,
                '尺寸': f"{img.width} x {img.height}",
                '宽度': img.width,
                '高度': img.height,
            }

            # 尝试获取DPI信息
            if hasattr(img, 'info'):
                if 'dpi' in img.info:
                    info['DPI'] = img.info['dpi']
                if 'resolution' in img.info:
                    info['分辨率'] = img.info['resolution']

            return info
    except Exception as e:
        print(f"PIL读取错误: {e}")
        return None

def 主函数():
    目录 = r'C:\Users\jiao\Desktop\1(1)\1'

    print("=" * 60)
    print("📁 TIF文件信息查看器")
    print("=" * 60)

    # 列出目录中的TIF文件
    tif_files = []
    for 文件 in os.listdir(目录):
        if 文件.lower().endswith('.tif') or 文件.lower().endswith('.tiff'):
            tif_files.append(文件)

    if not tif_files:
        print("目录中没有找到TIF文件")
        return

    print(f"找到 {len(tif_files)} 个TIF文件:")
    for f in tif_files:
        print(f"  - {f}")

    # 分析每个文件
    for 文件名 in tif_files:
        文件路径 = os.path.join(目录, 文件名)
        文件大小 = os.path.getsize(文件路径) / (1024*1024)  # MB

        print(f"\n{'='*40}")
        print(f"文件: {文件名}")
        print(f"大小: {文件大小:.2f} MB")
        print(f"{'='*40}")

        # 使用PIL查看
        pil_info = 使用PIL查看(文件路径)
        if pil_info:
            print(f"\n📊 图像信息 (PIL):")
            print(f"  格式: {pil_info.get('格式', 'Unknown')}")
            print(f"  颜色模式: {pil_info.get('模式', 'Unknown')}")
            print(f"  尺寸: {pil_info.get('尺寸', 'Unknown')}")
            if 'DPI' in pil_info:
                print(f"  DPI: {pil_info['DPI']}")

        # 尝试读取TIFF头
        tiff_info = 读取tiff_header(文件路径)
        if tiff_info:
            print(f"\n🔍 TIFF头信息:")
            if '宽度' in tiff_info:
                print(f"  图像宽度: {tiff_info['宽度']} 像素")
            if '高度' in tiff_info:
                print(f"  图像高度: {tiff_info['高度']} 像素")
            if 'X分辨率' in tiff_info:
                print(f"  X分辨率: {tiff_info['X分辨率']}")
            if 'Y分辨率' in tiff_info:
                print(f"  Y分辨率: {tiff_info['Y分辨率']}")

        print(f"\n💡 建议:")
        print("  要获取准确的地理坐标和分辨率信息，")
        print("  请使用专业GIS软件如QGIS或安装rasterio库:")
        print("  pip install rasterio")

if __name__ == "__main__":
    主函数()