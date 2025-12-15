"""
坐标系处理模块
集成到耕地分析系统中的自动坐标系处理功能
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.crs import CRS
import cv2
import os
import tempfile
from datetime import datetime

class 坐标系处理器:
    """处理坐标系相关问题的工具类"""

    @staticmethod
    def 检查坐标系一致性(文件1路径, 文件2路径=None):
        """
        检查两个文件的坐标系是否一致

        参数:
            文件1路径: 第一个文件路径
            文件2路径: 第二个文件路径（可选）

        返回:
            dict: 包含坐标系信息和一致性检查结果
        """
        结果 = {
            '文件1': {'路径': 文件1路径, 'crs': None, 'bounds': None, '单位': None},
            '文件2': {'路径': 文件2路径, 'crs': None, 'bounds': None, '单位': None} if 文件2路径 else None,
            '一致': True,
            '问题描述': None
        }

        # 读取第一个文件
        try:
            with rasterio.open(文件1路径) as src:
                结果['文件1']['crs'] = str(src.crs)
                结果['文件1']['bounds'] = src.bounds
                结果['文件1']['单位'] = src.crs.linear_units if src.crs and hasattr(src.crs, 'linear_units') else '未知'
        except Exception as e:
            结果['问题描述'] = f"无法读取文件1: {str(e)}"
            return 结果

        # 如果有第二个文件，读取并比较
        if 文件2路径:
            try:
                with rasterio.open(文件2路径) as src:
                    结果['文件2']['crs'] = str(src.crs)
                    结果['文件2']['bounds'] = src.bounds
                    结果['文件2']['单位'] = src.crs.linear_units if src.crs and hasattr(src.crs, 'linear_units') else '未知'

                    # 检查一致性
                    if 结果['文件1']['crs'] != 结果['文件2']['crs']:
                        结果['一致'] = False
                        结果['问题描述'] = "坐标系不一致"
            except Exception as e:
                结果['问题描述'] = f"无法读取文件2: {str(e)}"

        return 结果

    @staticmethod
    def 自动转换坐标系(输入文件路径, 目标crs, 输出目录=None):
        """
        自动转换坐标系

        参数:
            输入文件路径: 输入TIF文件路径
            目标crs: 目标坐标系（可以是EPSG码或CRS对象）
            输出目录: 输出目录，如果为None则使用输入文件所在目录

        返回:
            str: 转换后的文件路径，如果失败返回None
        """
        if 输出目录 is None:
            输出目录 = os.path.dirname(输入文件路径)

        # 生成输出文件名
        文件名 = os.path.basename(输入文件路径)
        名称, 扩展名 = os.path.splitext(文件名)
        输出文件路径 = os.path.join(输出目录, f"{名称}_坐标系转换_{datetime.now().strftime('%H%M%S')}{扩展名}")

        try:
            with rasterio.open(输入文件路径) as src:
                # 确保目标CRS是正确的格式
                if isinstance(目标crs, str):
                    目标crs = CRS.from_string(目标crs)

                # 检查是否需要转换
                if src.crs == 目标crs:
                    print(f"  ✅ 坐标系已匹配，无需转换")
                    return 输入文件路径

                # 计算转换参数
                transform, width, height = calculate_default_transform(
                    src.crs,
                    目标crs,
                    src.width,
                    src.height,
                    *src.bounds
                )

                # 创建新的元数据
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': 目标crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                # 创建转换后的数组
                数组 = np.zeros((src.count, height, width), dtype=src.dtypes[0])

                # 执行坐标转换
                print(f"  🔄 正在转换坐标系: {src.crs} -> {目标crs}")
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=数组[i-1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=目标crs,
                        resampling=rasterio.enums.Resampling.nearest
                    )

                # 保存转换后的文件
                with rasterio.open(输出文件路径, 'w', **kwargs) as dst:
                    dst.write(数组)

                print(f"  ✅ 坐标系转换完成: {输出文件路径}")
                return 输出文件路径

        except Exception as e:
            print(f"  ❌ 坐标系转换失败: {str(e)}")
            return None

    @staticmethod
    def 智能处理坐标系差异(基准数据路径, 当前图像路径, 输出目录=None):
        """
        智能处理坐标系差异

        参数:
            基准数据路径: 基准数据文件路径
            当前图像路径: 当前图像文件路径
            输出目录: 输出目录

        返回:
            dict: 处理结果
        """
        结果 = {
            '需要转换': False,
            '转换后路径': 当前图像路径,
            '基准crs': None,
            '当前crs': None,
            '信息': []
        }

        # 检查坐标系一致性
        检查结果 = 坐标系处理器.检查坐标系一致性(基准数据路径, 当前图像路径)
        结果['基准crs'] = 检查结果['文件1']['crs']
        结果['当前crs'] = 检查结果['文件2']['crs']

        if not 检查结果['一致']:
            结果['需要转换'] = True
            结果['信息'].append(f"检测到坐标系不匹配")
            结果['信息'].append(f"  基准数据: {结果['基准crs']}")
            结果['信息'].append(f"  当前图像: {结果['当前crs']}")

            # 自动转换
            转换后路径 = 坐标系处理器.自动转换坐标系(
                当前图像路径,
                结果['基准crs'],
                输出目录
            )

            if 转换后路径 and 转换后路径 != 当前图像路径:
                结果['转换后路径'] = 转换后路径
                结果['信息'].append(f"✅ 已自动转换坐标系")
            else:
                结果['信息'].append(f"❌ 坐标系转换失败")
        else:
            结果['信息'].append(f"✅ 坐标系一致")

        return 结果


# 集成函数 - 修改耕地分析系统使用
def 预处理图像坐标系(基准数据路径, 图像路径, 输出目录=None):
    """
    预处理图像，确保坐标系一致

    这个函数可以被耕地分析系统调用，自动处理坐标系问题

    参数:
        基准数据路径: 基准数据路径（如果有）
        图像路径: 要处理的图像路径
        输出目录: 临时文件输出目录

    返回:
        str: 处理后的图像路径
    """
    if not 基准数据路径 or not os.path.exists(基准数据路径):
        # 没有基准数据，直接返回原图像
        return 图像路径

    # 检查坐标系
    处理结果 = 坐标系处理器.智能处理坐标系差异(
        基准数据路径,
        图像路径,
        输出目录 or tempfile.gettempdir()
    )

    # 打印处理信息
    for 信息 in 处理结果['信息']:
        print(f"  {信息}")

    return 处理结果['转换后路径']


# 批量处理函数
def 批量转换坐标系目录(输入目录, 目标crs, 输出目录=None):
    """
    批量转换目录中所有TIF文件的坐标系

    参数:
        输入目录: 输入目录
        目标crs: 目标坐标系
        输出目录: 输出目录，如果为None则在输入目录创建子目录

    返回:
        list: 转换成功的文件列表
    """
    if 输出目录 is None:
        输出目录 = os.path.join(输入目录, "坐标系转换")

    if not os.path.exists(输出目录):
        os.makedirs(输出目录)

    转换成功 = []

    for 文件名 in os.listdir(输入目录):
        if 文件名.lower().endswith(('.tif', '.tiff')):
            输入路径 = os.path.join(输入目录, 文件名)
            输出路径 = 坐标系处理器.自动转换坐标系(
                输入路径,
                目标crs,
                输出目录
            )
            if 输出路径:
                转换成功.append(输出路径)

    print(f"\n✅ 批量转换完成，成功转换 {len(转换成功)} 个文件")
    return 转换成功


if __name__ == "__main__":
    # 测试功能
    print("坐标系处理模块测试")
    print("=" * 50)

    # 示例：检查两个文件
    文件1 = "path/to/基准数据.tif"
    文件2 = "path/to/今年图像.tif"

    if os.path.exists(文件1) and os.path.exists(文件2):
        print("\n1. 检查坐标系一致性:")
        检查结果 = 坐标系处理器.检查坐标系一致性(文件1, 文件2)
        print(f"  文件1 CRS: {检查结果['文件1']['crs']}")
        print(f"  文件2 CRS: {检查结果['文件2']['crs']}")
        print(f"  一致性: {'✅ 一致' if 检查结果['一致'] else '❌ 不一致'}")

        if not 检查结果['一致']:
            print("\n2. 智能处理坐标系差异:")
            处理结果 = 坐标系处理器.智能处理坐标系差异(文件1, 文件2)
            print(f"  需要转换: {'是' if 处理结果['需要转换'] else '否'}")
            print(f"  处理后路径: {处理结果['转换后路径']}")
    else:
        print("请提供有效的文件路径进行测试")