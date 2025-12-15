"""
智能坐标匹配器
在不resize的情况下处理坐标系差异
"""

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.transform import Affine
import cv2

class 智能坐标匹配器:
    """智能处理不同分辨率和坐标系的图像匹配"""

    @staticmethod
    def 创建坐标映射(基准图像, 基准transform, 基准crs, 目标图像, 目标transform, 目标crs):
        """
        创建两个图像之间的坐标映射

        参数:
            基准图像: 基准图像数据（去年掩码）
            基准transform: 基准图像的仿射变换
            基准crs: 基准图像的坐标系
            目标图像: 目标图像信息
            目标transform: 目标图像的仿射变换
            目标crs: 目标图像的坐标系

        返回:
            坐标映射函数
        """

        # 如果坐标系相同，直接使用像素坐标映射
        if str(基准crs) == str(目标crs):
            print(f"  坐标系相同，使用直接像素映射")
            return 智能坐标匹配器._直接像素映射
        else:
            print(f"  坐标系不同，创建地理坐标映射")
            return 智能坐标匹配器._地理坐标映射

    @staticmethod
    def _直接像素映射(基准x, 基准y, 基准形状, 目标形状, **kwargs):
        """直接像素坐标映射（用于相同尺寸的图像）"""
        # 简单的比例映射
        scale_x = 目标形状[1] / 基准形状[1]
        scale_y = 目标形状[0] / 基准形状[0]

        目标x = int(基准x * scale_x)
        目标y = int(基准y * scale_y)

        return 目标x, 目标y

    @staticmethod
    def _地理坐标映射(基准x, 基准y, 基准transform, 基准crs, 目标transform, 目标crs, **kwargs):
        """地理坐标映射（处理不同坐标系）"""
        # 将基准像素坐标转换为地理坐标
        基准_geo_x, 基准_geo_y = 基准transform * (基准x, 基准y)

        # 转换坐标系
        from rasterio.warp import transform
        目标_geo_x, 目标_geo_y = transform(
            基准crs, 目标crs, [基准_geo_x], [基准_geo_y]
        )

        # 将地理坐标转换为目标图像的像素坐标
        目标x, 目标y = ~目标transform * (目标_geo_x[0], 目标_geo_y[0])

        return int(目标x), int(目标y)

    @staticmethod
    def 应用掩码转换(基准掩码, 目标形状, 映射函数, **映射参数):
        """
        将基准掩码转换到目标坐标系，保持清晰度

        参数:
            基准掩码: 基准掩码数据
            目标形状: 目标图像的形状
            映射函数: 坐标映射函数
            映射参数: 映射函数的参数

        返回:
            转换后的掩码
        """
        print(f"  开始智能掩码转换...")
        print(f"  基准尺寸: {基准掩码.shape}")
        print(f"  目标尺寸: {目标形状}")

        # 创建目标掩码
        目标掩码 = np.zeros(目标形状, dtype=np.uint8)

        # 获取基准掩码中的非零像素位置
        耕地像素 = np.argwhere(基准掩码 > 0.5)

        print(f"  转换 {len(耕地像素)} 个像素...")

        # 对每个耕地像素进行坐标转换
        转换计数 = 0
        for y, x in 耕地像素:
            try:
                目标x, 目标y = 映射函数(
                    x, y,
                    基准形状=基准掩码.shape,
                    目标形状=目标形状,
                    **映射参数
                )

                # 检查坐标是否在目标范围内
                if 0 <= 目标x < 目标形状[1] and 0 <= 目标y < 目标形状[0]:
                    目标掩码[目标y, 目标x] = 1
                    转换计数 += 1

            except Exception as e:
                # 忽略转换失败的像素
                pass

        print(f"  成功转换 {转换计数} 个像素")
        print(f"  转换后耕地像素数: {np.sum(目标掩码)}")

        # 使用形态学操作填充小空洞，保持轮廓清晰
        kernel = np.ones((3, 3), np.uint8)
        目标掩码 = cv2.morphologyEx(目标掩码, cv2.MORPH_CLOSE, kernel)

        return 目标掩码


def 创建带坐标系的去年掩码_优化版(基准信息, 今年左上角, 今年右下角, 今年crs, 今年meta):
    """
    优化版本的坐标系处理，避免图像变形
    """
    print("\n🔄 使用智能坐标匹配器...")

    # 获取裁剪区域（这部分保持不变）
    from rasterio.warp import transform_bounds
    from affine import Affine

    # 获取基准信息
    基准地图 = 基准信息['基准耕地地图']
    基准crs = 基准信息['crs']

    # 将今年图像的WGS84坐标转换到基准数据的坐标系
    基准_bounds = transform_bounds('EPSG:4326', 基准crs,
                                   今年左上角[0], 今年右下角[1],
                                   今年右下角[0], 今年左上角[1])

    # 计算在基准地图中的位置
    基准transform = Affine(
        基准信息['地理变换']['a'],
        基准信息['地理变换']['b'],
        基准信息['地理变换']['c'],
        基准信息['地理变换']['d'],
        基准信息['地理变换']['e'],
        基准信息['地理变换']['f']
    )

    # 裁剪去年的掩码
    左上_col, 左上_row = ~基准transform * (基准_bounds[0], 基准_bounds[3])
    右下_col, 右下_row = ~基准transform * (基准_bounds[2], 基准_bounds[1])

    row_min = max(0, int(min(左上_row, 右下_row)))
    row_max = min(基准地图.shape[0], int(max(左上_row, 右下_row)))
    col_min = max(0, int(min(左上_col, 右下_col)))
    col_max = min(基准地图.shape[1], int(max(左上_col, 右下_col)))

    裁剪区域 = 基准地图[row_min:row_max, col_min:col_max]

    # 创建坐标映射
    映射函数 = 智能坐标匹配器.创建坐标映射(
        裁剪区域,
        Affine(
            基准信息['地理变换']['a'],
            基准信息['地理变换']['b'],
            基准信息['地理变换']['c'] + 基准信息['地理变换']['a'] * col_min,
            基准信息['地理变换']['d'],
            基准信息['地理变换']['e'],
            基准信息['地理变换']['f'] + 基准信息['地理变换']['e'] * row_min
        ),
        基准crs,
        None,  # 目标图像
        今年meta['transform'],
        今年crs
    )

    # 应用智能转换
    目标形状 = (今年meta['height'], 今年meta['width'])
    转换后掩码 = 智能坐标匹配器.应用掩码转换(
        裁剪区域,
        目标形状,
        映射函数,
        基准transform=基准transform,
        基准crs=基准crs,
        目标transform=今年meta['transform'],
        目标crs=今年crs
    )

    return 转换后掩码


if __name__ == "__main__":
    print("智能坐标匹配器测试")
    print("=" * 50)
    print("这个模块提供智能的坐标转换功能，")
    print("避免图像变形，保持轮廓清晰。")