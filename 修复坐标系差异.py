"""
修复坐标系差异的直接解决方案
在图形界面中集成自动坐标系转换
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import rowcol
import tempfile
import shutil

def 创建带坐标系的去年掩码(基准信息, 今年左上角, 今年右下角, 今年crs, 今年meta):
    """
    创建去年掩码，确保坐标系与今年图像一致

    参数:
        基准信息: 基准数据信息
        今年左上角: 今年图像左上角坐标（WGS84）
        今年右下角: 今年图像右下角坐标（WGS84）
        今年crs: 今年图像的坐标系
        今年meta: 今年图像的元数据

    返回:
        np.ndarray: 与今年图像坐标系一致的去年掩码
    """
    from 坐标系处理模块 import 坐标系处理器

    print("\n🔄 创建带坐标系的去年掩码...")

    # 1. 从基准地图裁剪区域
    基准地图 = 基准信息['基准耕地地图']
    基准crs = 基准信息['crs']

    # 将今年图像的WGS84坐标转换到基准数据的坐标系
    from rasterio.warp import transform
    今年左上_基准 = transform('EPSG:4326', 基准crs, [今年左上角[0]], [今年左上角[1]])
    今年右下_基准 = transform('EPSG:4326', 基准crs, [今年右下角[0]], [今年右下角[1]])

    裁剪_左x, 裁剪_上y = 今年左上_基准
    裁剪_右x, 裁剪_下y = 今年右下_基准

    # 使用基准数据的地理变换
    from affine import Affine
    基准transform = Affine(
        基准信息['地理变换']['a'],
        基准信息['地理变换']['b'],
        基准信息['地理变换']['c'],
        基准信息['地理变换']['d'],
        基准信息['地理变换']['e'],
        基准信息['地理变换']['f']
    )

    # 计算在基准地图中的像素位置
    # 使用 rasterio 的 rowcol 方法
    # 注意：rowcol 的参数顺序是 (xs, ys)
    左上_col, 左上_row = rowcol(~基准transform, [裁剪_左x], [裁剪_上y])
    右下_col, 右下_row = rowcol(~基准transform, [裁剪_右x], [裁剪_下y])

    # 获取单个值
    左上_col = 左上_col[0]
    左上_row = 左上_row[0]
    右下_col = 右下_col[0]
    右下_row = 右下_row[0]

    # 裁剪范围
    row_min = max(0, int(min(左上_row, 右下_row)))
    row_max = min(基准地图.shape[0], int(max(左上_row, 右下_row)))
    col_min = max(0, int(min(左上_col, 右下_col)))
    col_max = min(基准地图.shape[1], int(max(左上_col, 右下_col)))

    # 裁剪去年的掩码
    裁剪区域 = 基准地图[row_min:row_max, col_min:col_max]

    print(f"   基准地图坐标系: {基准crs}")
    print(f"   今年图像坐标系: {今年crs}")

    # 2. 创建临时TIF文件（带基准坐标系）
    temp_dir = tempfile.mkdtemp()

    # 基准坐标系的元数据
    基准meta = 今年meta.copy()
    基准meta.update({
        'crs': 基准crs,
        'height': 裁剪区域.shape[0],
        'width': 裁剪区域.shape[1],
        'transform': Affine(
            基准transform.a, 基准transform.b,
            基准transform.c + 基准transform.a * col_min,
            基准transform.d, 基准transform.e,
            基准transform.f + 基准transform.e * row_min
        )
    })

    # 保存裁剪区域（带基准坐标系）
    基准文件 = os.path.join(temp_dir, "去年_基准坐标系.tif")
    with rasterio.open(基准文件, 'w', **基准meta) as dst:
        dst.write(裁剪区域[np.newaxis, :, :])

    print(f"   已保存去年数据（基准坐标系）")

    # 3. 检查是否需要转换坐标系
    if 基准crs != 今年crs:
        print(f"   ⚠️ 坐标系不同，正在转换...")

        # 转换到今年图像的坐标系
        转换后文件 = 坐标系处理器.自动转换坐标系(
            基准文件,
            今年crs,
            temp_dir
        )

        if 转换后文件:
            # 读取转换后的数据
            with rasterio.open(转换后文件) as src:
                转换后数据 = src.read(1)

            print(f"   ✅ 坐标系转换完成")
            print(f"   转换后尺寸: {转换后数据.shape}")

            # 清理临时文件
            shutil.rmtree(temp_dir)

            return 转换后数据.astype(np.uint8)
        else:
            print(f"   ❌ 坐标系转换失败")
            shutil.rmtree(temp_dir)
            return None
    else:
        print(f"   ✅ 坐标系一致，无需转换")
        # 清理临时文件
        shutil.rmtree(temp_dir)
        return 裁剪区域.astype(np.uint8)


def 修复图形界面的坐标系问题():
    """
    提供修复图形界面坐标系问题的代码
    """
    修复代码 = '''
# 在耕地分析工具_图形界面.py中找到这段代码（约第888行）：

# 替换这部分代码：
# 去年掩码 = 裁剪区域.astype(np.uint8)  # 直接使用原始分辨率

# 改为：
# ✅ 修复坐标系差异
去年掩码 = 创建带坐标系的去年掩码(
    基准信息,
    今年左上角,
    今年右下角,
    src.crs,
    src.meta
)

if 去年掩码 is None:
    self.输出结果("⚠️ 坐标系转换失败，使用原始数据（可能不准确）")
    去年掩码 = 裁剪区域.astype(np.uint8)
else:
    self.输出结果("✅ 已处理坐标系差异，确保数据一致性")
'''

    print("修复代码：")
    print(修复代码)


if __name__ == "__main__":
    print("修复坐标系差异解决方案")
    print("=" * 50)

    # 提供修复代码
    修复图形界面的坐标系问题()