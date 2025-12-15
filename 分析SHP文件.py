"""
分析SHP文件，理解基准数据偏差的原因
"""
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from affine import Affine
import os

def 分析SHP文件():
    print("="*60)
    print("SHP文件分析工具")
    print("="*60)

    shp_path = "通北局种植作物.shp"

    if not os.path.exists(shp_path):
        print(f"❌ 找不到文件: {shp_path}")
        return

    # 读取SHP文件
    print(f"\n📁 读取SHP文件: {shp_path}")
    gdf = gpd.read_file(shp_path)

    print(f"\n📊 SHP文件信息:")
    print(f"   几何数量: {len(gdf)}")
    print(f"   坐标系: {gdf.crs}")
    print(f"   属性列: {list(gdf.columns)}")

    # 查看属性
    if len(gdf) > 0:
        print(f"\n📋 属性示例:")
        print(gdf.head())

    # 分析几何信息
    print(f"\n🔍 几何分析:")
    总面积_平方米 = gdf.geometry.area.sum()
    总面积_亩 = 总面积_平方米 / 666.67
    print(f"   几何总面积: {总面积_平方米:,.0f} 平方米")
    print(f"   几何总面积: {总面积_亩:,.2f} 亩")

    # 检查是否有耕地标识字段
    耕地字段 = None
    for col in gdf.columns:
        if '耕地' in col or 'crop' in col.lower() or '种植' in col or '类型' in col:
            耕地字段 = col
            break

    if 耕地字段:
        print(f"\n🏷️  发现耕地字段: {耕地字段}")
        print(f"   唯一值: {gdf[耕地字段].unique()}")

        # 统计耕地面积
        耕地_gdf = gdf[gdf[耕地字段] == 1]  # 假设1表示耕地
        耕地面积_平方米 = 耕地_gdf.geometry.area.sum()
        耕地面积_亩 = 耕地面积_平方米 / 666.67
        print(f"   耕地面积: {耕地面积_平方米:,.0f} 平方米")
        print(f"   耕地面积: {耕地面积_亩:,.2f} 亩")
    else:
        print(f"\n⚠️  未找到明显的耕地标识字段")
        print(f"   可能所有几何都是耕地，或有其他标识方式")

    # 分析基准数据生成过程
    print(f"\n🔧 模拟基准数据生成:")

    # 获取基准数据信息
    if os.path.exists("耕地识别模型_基准数据.pkl"):
        import pickle
        with open("耕地识别模型_基准数据.pkl", 'rb') as f:
            基准信息 = pickle.load(f)

        基准地图 = 基准信息['基准耕地地图']
        基准像素分辨率 = 基准信息.get('像素分辨率_米', 0.218)

        print(f"   基准地图尺寸: {基准地图.shape}")
        print(f"   基准像素分辨率: {基准像素分辨率} 米/像素")

        # 计算基准地图中的耕地面积
        基准耕地像素 = np.sum(基准地图 > 0.5)
        基准耕地面积_平方米 = 基准耕地像素 * (基准像素分辨率 ** 2)
        基准耕地面积_亩 = 基准耕地面积_平方米 / 666.67

        print(f"\n   基准数据统计:")
        print(f"   耕地像素数: {基准耕地像素:,}")
        print(f"   耕地面积: {基准耕地面积_亩:,.2f} 亩")

        # 与SHP几何面积对比
        if 耕地字段 and '耕地_gdf' in locals():
            差异 = 基准耕地面积_亩 - 耕地面积_亩
            差异百分比 = 差异 / 耕地面积_亩 * 100 if 耕地面积_亩 > 0 else 0

            print(f"\n📊 面积对比:")
            print(f"   SHP几何面积: {耕地面积_亩:,.2f} 亩")
            print(f"   基准数据面积: {基准耕地面积_亩:,.2f} 亩")
            print(f"   差异: {差异:+,.2f} 亩 ({差异百分比:+.1f}%)")

    # 检查测试区域（您的12.6亩区域）
    print(f"\n🎯 测试区域分析:")
    print(f"   根据之前的日志，测试区域:")
    print(f"   - 从基准地图裁剪: 840x754 像素")
    print(f"   - 耕地像素: 191,887")
    print(f"   - 计算面积: 13.679 亩")
    print(f"   - 实际应该: 12.6 亩")
    print(f"   - 偏差: +1.079 亩 (+8.6%)")

    # 分析可能的原因
    print(f"\n❓ 偏差可能原因:")
    print(f"   1. SHP文件本身包含了边界缓冲区")
    print(f"   2. 几何精度问题（坐标点过于密集）")
    print(f"   3. 基准数据生成时的降采样（4倍）")
    print(f"   4. 坐标系转换的微小误差")
    print(f"   5. SHP文件中的耕地定义过于宽松")

    print("\n" + "="*60)

if __name__ == "__main__":
    分析SHP文件()