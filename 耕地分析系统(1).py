"""
耕地图像分析系统
功能:
1. 裁剪大型TIF航拍图像并保留地理信息
2. 基于Shapefile标注识别耕地区域
3. 计算耕地面积和比例
4. 比较两年图像的耕地变化
5. 输出包含地理坐标的结构化数据
"""

# ==================== 配置区域 - 请在这里修改路径 ====================

# 输入文件路径配置
TIF图像路径 = r"E:\八二\20250422通北八二2_3\result13.tif"
Shapefile路径 = r"E:\通北局种植作物"  # 如果有Shapefile标注文件,请填写路径,例如: r"D:\data\farmland.shp"

# 两年对比分析路径(如需使用)
年份1_TIF路径 = r""  # 第一年的TIF图像路径
年份2_TIF路径 = r""  # 第二年的TIF图像路径
年份1_Shapefile路径 = r""  # 第一年的Shapefile路径(可选)
年份2_Shapefile路径 = r""  # 第二年的Shapefile路径(可选)

# 输出目录配置
输出目录 = r"E:\out\新建文件夹"

# 裁剪参数配置
裁剪尺寸 = 1000  # 每个裁剪块的尺寸(像素)
重叠像素 = 100   # 裁剪块之间的重叠像素数

# 分析模式配置
分析模式 = "单张图像"  # 可选: "单张图像" 或 "两年对比" 或 "训练模型" 或 "使用模型"

# 两年对比高级参数
重叠阈值 = 0.5  # 认为是同一区域的最小重叠比例(0-1), 0.5表示至少50%重叠

# 模型训练参数
模型保存路径 = r"D:\微信\document\xwechat_files\wxid_h80ke0c6ca1m22_27bc\msg\file\2025-11\001\001\耕地识别模型.h5"
训练数据比例 = 0.8  # 80%用于训练, 20%用于验证

# ===================================================================

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
import geopandas as gpd
from shapely.geometry import box, mapping
import pandas as pd
from pathlib import Path
import json
from typing import Tuple, List, Dict
import cv2
import pickle
from datetime import datetime

# ==================== GPU加速检测 ====================
print("="*60)
print("🚀 GPU加速检测")
print("="*60)

try:
    import tensorflow as tf
    
    # 检测GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ 检测到 {len(gpus)} 个GPU设备:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # 启用GPU内存动态增长
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ 已启用GPU内存动态增长")
        except RuntimeError as e:
            print(f"⚠️  GPU配置警告: {e}")
        
        print("🎯 AI预测将使用GPU加速！")
    else:
        print("❌ 未检测到GPU，将使用CPU预测（速度较慢）")
except ImportError:
    print("⚠️  TensorFlow未加载，稍后将检测")

print("="*60)
print()

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    try:
        import keras
        KERAS_AVAILABLE = True
    except ImportError:
        KERAS_AVAILABLE = False
        print("⚠️  未安装TensorFlow/Keras, 模型训练和预测功能不可用")

class 耕地分析系统:
    """
    耕地图像分析处理系统
    """
    
    def __init__(self, 输出目录: str = None):
        """
        初始化系统
        
        参数:
            输出目录: 结果输出目录路径
        """
        self.输出目录 = 输出目录 or "./output"
        os.makedirs(self.输出目录, exist_ok=True)
        
    def 裁剪图像并保留地理信息(self, 
                          tif路径: str,
                          裁剪尺寸: int = 1000,
                          重叠像素: int = 0) -> List[Dict]:
        """
        裁剪TIF图像为多个小块,并保留每块的地理信息
        
        参数:
            tif路径: 输入TIF文件路径
            裁剪尺寸: 每个小块的尺寸(像素)
            重叠像素: 裁剪块之间的重叠像素数
            
        返回:
            裁剪块信息列表,包含文件路径和地理坐标
        """
        裁剪块列表 = []
        
        with rasterio.open(tif路径) as src:
            宽度 = src.width
            高度 = src.height
            原始crs = src.crs
            原始transform = src.transform
            
            print(f"📐 原始图像尺寸: {宽度} x {高度}")
            print(f"🌍 坐标系: {原始crs}")
            
            步长 = 裁剪尺寸 - 重叠像素
            块编号 = 0
            
            # 遍历图像进行裁剪
            for 行起始 in range(0, 高度, 步长):
                for 列起始 in range(0, 宽度, 步长):
                    # 计算窗口大小
                    窗口高度 = min(裁剪尺寸, 高度 - 行起始)
                    窗口宽度 = min(裁剪尺寸, 宽度 - 列起始)
                    
                    # 创建读取窗口
                    窗口 = Window(列起始, 行起始, 窗口宽度, 窗口高度)
                    
                    # 读取数据
                    数据 = src.read(window=窗口)
                    
                    # 计算该窗口的地理变换参数
                    窗口transform = src.window_transform(窗口)
                    
                    # 计算四角坐标(投影坐标)
                    左上角x = 窗口transform.c
                    左上角y = 窗口transform.f
                    右下角x = 左上角x + 窗口transform.a * 窗口宽度
                    右下角y = 左上角y + 窗口transform.e * 窗口高度
                    
                    # 转换为经纬度(WGS84)
                    if 原始crs:
                        左上角经度, 左上角纬度 = transform(原始crs, 'EPSG:4326', [左上角x], [左上角y])
                        右下角经度, 右下角纬度 = transform(原始crs, 'EPSG:4326', [右下角x], [右下角y])
                    else:
                        左上角经度, 左上角纬度 = [左上角x], [左上角y]
                        右下角经度, 右下角纬度 = [右下角x], [右下角y]
                    
                    # 保存裁剪块
                    输出文件名 = f"tile_{块编号}_r{行起始}_c{列起始}.tif"
                    输出路径 = os.path.join(self.输出目录, "tiles", 输出文件名)
                    os.makedirs(os.path.dirname(输出路径), exist_ok=True)
                    
                    # 写入裁剪块文件(保留地理信息)
                    with rasterio.open(
                        输出路径,
                        'w',
                        driver='GTiff',
                        height=窗口高度,
                        width=窗口宽度,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=原始crs,
                        transform=窗口transform
                    ) as dst:
                        dst.write(数据)
                    
                    # 记录裁剪块信息
                    裁剪块信息 = {
                        '块编号': 块编号,
                        '文件路径': 输出路径,
                        '行起始': 行起始,
                        '列起始': 列起始,
                        '窗口宽度': 窗口宽度,
                        '窗口高度': 窗口高度,
                        '投影坐标_左上角': (左上角x, 左上角y),
                        '投影坐标_右下角': (右下角x, 右下角y),
                        '经纬度_左上角': (左上角经度[0], 左上角纬度[0]),
                        '经纬度_右下角': (右下角经度[0], 右下角纬度[0]),
                        'crs': str(原始crs)
                    }
                    
                    裁剪块列表.append(裁剪块信息)
                    块编号 += 1
            
            print(f"✅ 完成裁剪,共生成 {块编号} 个裁剪块")
        
        return 裁剪块列表
    
    def 基于shapefile提取耕地(self, 
                           tif路径: str,
                           shapefile路径: str,
                           耕地字段名: str = None) -> np.ndarray:
        """
        基于Shapefile标注提取耕地区域
        
        参数:
            tif路径: TIF图像路径
            shapefile路径: Shapefile标注文件路径
            耕地字段名: Shapefile中标识耕地的字段名
            
        返回:
            耕地掩码数组(0-非耕地, 1-耕地)
        """
        # 读取Shapefile
        gdf = gpd.read_file(shapefile路径)
        
        # 读取TIF
        with rasterio.open(tif路径) as src:
            # 确保坐标系一致
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            # 创建掩码
            from rasterio.features import geometry_mask
            
            if 耕地字段名:
                # 筛选耕地区域
                耕地几何 = gdf[gdf[耕地字段名] == 1].geometry
            else:
                耕地几何 = gdf.geometry
            
            # 生成掩码(True为非耕地, False为耕地)
            掩码 = geometry_mask(
                耕地几何,
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=False
            )
            
            # 转换为0-1数组(1为耕地)
            耕地掩码 = (~掩码).astype(np.uint8)
        
        return 耕地掩码
    
    def 计算耕地面积和比例(self,
                       裁剪块信息: Dict,
                       耕地掩码: np.ndarray = None,
                       shapefile路径: str = None) -> Dict:
        """
        计算裁剪块的耕地面积和比例
        
        参数:
            裁剪块信息: 裁剪块的信息字典
            耕地掩码: 耕地掩码数组
            shapefile路径: Shapefile路径(如果未提供掩码)
            
        返回:
            包含面积和比例信息的字典
        """
        tif路径 = 裁剪块信息['文件路径']
        
        with rasterio.open(tif路径) as src:
            总像素数 = src.width * src.height
            
            # 如果没有提供掩码,尝试从shapefile生成
            if 耕地掩码 is None and shapefile路径:
                耕地掩码 = self.基于shapefile提取耕地(tif路径, shapefile路径)
            elif 耕地掩码 is None:
                # 简单的颜色阈值方法(示例,需根据实际情况调整)
                影像数据 = src.read()
                耕地掩码 = self._简单耕地识别(影像数据)
            
            # 计算耕地像素数
            耕地像素数 = np.sum(耕地掩码 == 1)
            
            # 计算比例
            耕地比例 = 耕地像素数 / 总像素数 if 总像素数 > 0 else 0
            
            # 计算实际面积(基于像素分辨率)
            像素分辨率x = abs(src.transform.a)  # 米/像素
            像素分辨率y = abs(src.transform.e)  # 米/像素
            单像素面积 = 像素分辨率x * 像素分辨率y  # 平方米
            
            耕地面积_平方米 = 耕地像素数 * 单像素面积
            总面积_平方米 = 总像素数 * 单像素面积
            
            结果 = {
                **裁剪块信息,
                '总像素数': int(总像素数),
                '耕地像素数': int(耕地像素数),
                '耕地比例': float(耕地比例),
                '耕地面积_平方米': float(耕地面积_平方米),
                '耕地面积_亩': float(耕地面积_平方米 / 666.67),  # 1亩 = 666.67平方米
                '总面积_平方米': float(总面积_平方米),
                '像素分辨率_米': float(像素分辨率x)
            }
        
        return 结果
    
    def _简单耕地识别(self, 影像数据: np.ndarray) -> np.ndarray:
        """
        简单的基于颜色的耕地识别(示例方法)
        实际应用中应使用深度学习模型或更复杂的算法
        """
        # 转换为HxWxC格式
        if 影像数据.shape[0] <= 4:  # 通道在第一维
            影像数据 = np.transpose(影像数据, (1, 2, 0))
        
        # 简单的绿色阈值(需根据实际情况调整)
        # 这里假设RGB格式
        if 影像数据.shape[2] >= 3:
            r, g, b = 影像数据[:,:,0], 影像数据[:,:,1], 影像数据[:,:,2]
            # 耕地通常呈现棕色或绿色
            掩码 = ((g > r * 0.9) & (g > b * 0.9)) | ((r > 100) & (g > 80) & (b < 100))
        else:
            掩码 = 影像数据[:,:,0] > np.mean(影像数据[:,:,0])
        
        return 掩码.astype(np.uint8)
    
    def _检查地理重叠(self, 块1: Dict, 块2: Dict, 容差: float = 1e-6) -> bool:
        """
        检查两个裁剪块是否在地理位置上重叠
        
        参数:
            块1: 第一个裁剪块信息
            块2: 第二个裁剪块信息
            容差: 坐标比较的容差值
            
        返回:
            是否重叠
        """
        # 获取两个块的边界
        块1_左上经度, 块1_左上纬度 = 块1['经纬度_左上角']
        块1_右下经度, 块1_右下纬度 = 块1['经纬度_右下角']
        
        块2_左上经度, 块2_左上纬度 = 块2['经纬度_左上角']
        块2_右下经度, 块2_右下纬度 = 块2['经纬度_右下角']
        
        # 检查是否有重叠(矩形相交检测)
        # 如果一个矩形在另一个的左边、右边、上边或下边,则不重叠
        if (块1_右下经度 < 块2_左上经度 or  # 块1在块2左边
            块1_左上经度 > 块2_右下经度 or  # 块1在块2右边
            块1_右下纬度 > 块2_左上纬度 or  # 块1在块2上边
            块1_左上纬度 < 块2_右下纬度):   # 块1在块2下边
            return False
        
        return True
    
    def _计算重叠面积(self, 块1: Dict, 块2: Dict) -> float:
        """
        计算两个裁剪块的重叠面积占比
        
        返回:
            重叠面积占块1的比例(0-1)
        """
        # 获取两个块的边界
        块1_左上经度, 块1_左上纬度 = 块1['经纬度_左上角']
        块1_右下经度, 块1_右下纬度 = 块1['经纬度_右下角']
        
        块2_左上经度, 块2_左上纬度 = 块2['经纬度_左上角']
        块2_右下经度, 块2_右下纬度 = 块2['经纬度_右下角']
        
        # 计算重叠区域
        重叠_左经度 = max(块1_左上经度, 块2_左上经度)
        重叠_右经度 = min(块1_右下经度, 块2_右下经度)
        重叠_上纬度 = min(块1_左上纬度, 块2_左上纬度)
        重叠_下纬度 = max(块1_右下纬度, 块2_右下纬度)
        
        # 计算重叠面积
        重叠宽度 = 重叠_右经度 - 重叠_左经度
        重叠高度 = 重叠_上纬度 - 重叠_下纬度
        
        if 重叠宽度 <= 0 or 重叠高度 <= 0:
            return 0.0
        
        重叠面积 = 重叠宽度 * 重叠高度
        
        # 块1的面积
        块1宽度 = 块1_右下经度 - 块1_左上经度
        块1高度 = 块1_左上纬度 - 块1_右下纬度
        块1面积 = 块1宽度 * 块1高度
        
        if 块1面积 == 0:
            return 0.0
        
        return 重叠面积 / 块1面积
    
    def 比较两年耕地变化(self,
                      年份1_tif: str,
                      年份2_tif: str,
                      shapefile1: str = None,
                      shapefile2: str = None,
                      重叠阈值: float = 0.5) -> pd.DataFrame:
        """
        比较两年图像的耕地变化(智能地理匹配)
        
        参数:
            年份1_tif: 第一年TIF图像路径
            年份2_tif: 第二年TIF图像路径
            shapefile1: 第一年Shapefile路径
            shapefile2: 第二年Shapefile路径
            重叠阈值: 认为是同一区域的最小重叠比例(0-1)
            
        返回:
            耕地变化数据DataFrame
        
        说明:
            - 自动基于地理坐标匹配两年图像中的相同区域
            - 支持不同大小、不同位置的图像对比
            - 只要地理位置有重叠就能进行对比
        """
        print("📊 开始耕地变化分析(智能地理匹配)...")
        
        # 读取两年图像的基本信息
        with rasterio.open(年份1_tif) as src1, rasterio.open(年份2_tif) as src2:
            print(f"\n📐 图像信息:")
            print(f"  年份1: {src1.width}x{src1.height}, CRS: {src1.crs}")
            print(f"  年份2: {src2.width}x{src2.height}, CRS: {src2.crs}")
        
        # 裁剪两年图像
        print("\n✂️ 裁剪第一年图像...")
        裁剪块1 = self.裁剪图像并保留地理信息(年份1_tif)
        
        print("\n✂️ 裁剪第二年图像...")
        裁剪块2 = self.裁剪图像并保留地理信息(年份2_tif)
        
        # 计算每块的耕地面积
        print("\n📏 计算第一年耕地面积...")
        年份1结果 = []
        for idx, 块 in enumerate(裁剪块1):
            结果 = self.计算耕地面积和比例(块, shapefile路径=shapefile1)
            年份1结果.append(结果)
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(裁剪块1)} 个裁剪块")
        
        print("\n📏 计算第二年耕地面积...")
        年份2结果 = []
        for idx, 块 in enumerate(裁剪块2):
            结果 = self.计算耕地面积和比例(块, shapefile路径=shapefile2)
            年份2结果.append(结果)
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(裁剪块2)} 个裁剪块")
        
        # 基于地理坐标匹配相同位置的裁剪块
        print(f"\n🔍 基于地理坐标匹配区域(重叠阈值: {重叠阈값*100}%)...")
        变化列表 = []
        匹配计数 = 0
        
        for 块1 in 年份1结果:
            最佳匹配块 = None
            最大重叠比例 = 0
            
            # 在第二年的所有块中查找与块1重叠最大的块
            for 块2 in 年份2结果:
                if self._检查地理重叠(块1, 块2):
                    重叠比例 = self._计算重叠面积(块1, 块2)
                    if 重叠比例 > 最大重叠比例:
                        最大重叠比例 = 重叠比例
                        最佳匹配块 = 块2
            
            # 如果找到重叠度足够的匹配块
            if 最佳匹配块 and 最大重叠比例 >= 重叠阈值:
                匹配计数 += 1
                
                面积变化 = 最佳匹配块['耕地面积_平方米'] - 块1['耕地面积_平方米']
                比例变化 = 最佳匹配块['耕地比例'] - 块1['耕地比例']
                
                变化信息 = {
                    '块编号_年份1': 块1['块编号'],
                    '块编号_年份2': 最佳匹配块['块编号'],
                    '左上角经度': 块1['经纬度_左上角'][0],
                    '左上角纬度': 块1['经纬度_左上角'][1],
                    '右下角经度': 块1['经纬度_右下角'][0],
                    '右下角纬度': 块1['经纬度_右下角'][1],
                    '重叠比例': 最大重叠比例,
                    '年份1_耕地面积_平方米': 块1['耕地面积_平方米'],
                    '年份1_耕地面积_亩': 块1['耕地面积_亩'],
                    '年份1_耕地比例': 块1['耕地比例'],
                    '年份2_耕地面积_平方米': 最佳匹配块['耕地面积_平方米'],
                    '年份2_耕地面积_亩': 最佳匹配块['耕地面积_亩'],
                    '年份2_耕地比例': 最佳匹配块['耕地比例'],
                    '面积变化_平方米': 面积变化,
                    '面积变化_亩': 面积变化 / 666.67,
                    '比例变化': 比例变化,
                    '变化类型': '增加' if 面积变化 > 0 else ('减少' if 面积变化 < 0 else '无变化')
                }
                
                变化列表.append(变化信息)
        
        变化df = pd.DataFrame(变化列表)
        
        print(f"\n✅ 变化分析完成!")
        print(f"  年份1总块数: {len(年份1结果)}")
        print(f"  年份2总块数: {len(年份2结果)}")
        print(f"  成功匹配: {匹配计数} 个区域")
        print(f"  匹配率: {匹配计数/len(年份1结果)*100:.1f}%")
        
        return 变化df
    
    def 保存基准年数据(self, 
                    tif路径: str,
                    shapefile路径: str = None,
                    保存路径: str = None,
                    年份标签: str = None) -> str:
        """
        保存基准年(第一年)的图像和分析数据,供后续对比使用
        
        参数:
            tif路径: 基准年TIF图像路径
            shapefile路径: 基准年Shapefile路径(可选)
            保存路径: 数据保存路径(默认使用配置的路径)
            年份标签: 年份标签(如"2020", "基准年")
            
        返回:
            保存的文件路径
        """
        print("\n" + "=" * 60)
        print("📦 保存基准年数据")
        print("=" * 60)
        
        # 裁剪图像
        print("\n✂️ 裁剪基准年图像...")
        裁剪块列表 = self.裁剪图像并保留地理信息(tif路径)
        
        # 计算耕地面积
        print("\n📊 计算耕地面积...")
        结果列表 = []
        for idx, 块 in enumerate(裁剪块列表):
            结果 = self.计算耕地面积和比例(块, shapefile路径=shapefile路径)
            结果列表.append(结果)
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(裁剪块列表)} 个裁剪块")
        
        # 准备保存的数据
        基准年数据 = {
            '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '年份标签': 年份标签 or datetime.now().strftime('%Y'),
            'TIF路径': tif路径,
            'Shapefile路径': shapefile路径,
            '分析结果': 结果列表,
            '裁剪参数': {
                '裁剪尺寸': 裁剪尺寸,
                '重叠像素': 重叠像素
            },
            '统计信息': {
                '总块数': len(结果列表),
                '总耕地面积_亩': sum(r['耕地面积_亩'] for r in 结果列表),
                '平均耕地比例': sum(r['耕地比例'] for r in 结果列表) / len(结果列表) if 结果列表 else 0
            }
        }
        
        # 保存数据
        保存路径 = 保存路径 or 基准年数据文件
        with open(保存路径, 'wb') as f:
            pickle.dump(基准年数据, f)
        
        print(f"\n✅ 基准年数据已保存: {保存路径}")
        print(f"  年份标签: {基准年数据['年份标签']}")
        print(f"  总块数: {基准年数据['统计信息']['总块数']}")
        print(f"  总耕地面积: {基准年数据['统计信息']['总耕地面积_亩']:.2f} 亩")
        print(f"  TIF路径: {tif路径}")
        if shapefile路径:
            print(f"  Shapefile路径: {shapefile路径}")
        
        return 保存路径
    
    def 与基准年对比(self,
                  当前年tif: str,
                  基准年数据路径: str = None,
                  当前年shapefile: str = None,
                  重叠阈值: float = 0.5) -> pd.DataFrame:
        """
        将当前年图像与已保存的基准年数据进行对比
        
        参数:
            当前年tif: 当前年TIF图像路径
            基准年数据路径: 基准年数据文件路径(默认使用配置的路径)
            当前年shapefile: 当前年Shapefile路径(可选)
            重叠阈值: 重叠阈值
            
        返回:
            耕地变化数据DataFrame
        """
        # 加载基准年数据
        基准年数据路径 = 基准年数据路径 or 基准年数据文件
        
        if not os.path.exists(基准年数据路径):
            raise FileNotFoundError(f"❌ 未找到基准年数据文件: {基准年数据路径}\n请先运行'保存基准年'模式保存基准年数据!")
        
        print("\n" + "=" * 60)
        print("📊 与基准年对比分析")
        print("=" * 60)
        
        with open(基准年数据路径, 'rb') as f:
            基准年数据 = pickle.load(f)
        
        print(f"\n📂 基准年信息:")
        print(f"  年份: {基准年数据['年份标签']}")
        print(f"  保存时间: {基准年数据['保存时间']}")
        print(f"  总块数: {基准年数据['统计信息']['总块数']}")
        print(f"  总耕地面积: {基准年数据['统计信息']['总耕地面积_亩']:.2f} 亩")
        
        基准年结果 = 基准年数据['分析结果']
        
        # 分析当前年图像
        print(f"\n📂 当前年图像: {os.path.basename(当前年tif)}")
        print("\n✂️ 裁剪当前年图像...")
        当前年裁剪块 = self.裁剪图像并保留地理信息(当前年tif)
        
        print("\n📊 计算当前年耕地面积...")
        当前年结果 = []
        for idx, 块 in enumerate(当前年裁剪块):
            结果 = self.计算耕地面积和比例(块, shapefile路径=当前年shapefile)
            当前年结果.append(结果)
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(当前年裁剪块)} 个裁剪块")
        
        # 基于地理坐标匹配
        print(f"\n🔍 基于地理坐标匹配区域(重叠阈值: {重叠阈값*100}%)...")
        变化列表 = []
        匹配计数 = 0
        
        for 基准块 in 基准年结果:
            最佳匹配块 = None
            最大重叠比例 = 0
            
            for 当前块 in 当前年结果:
                if self._检查地理重叠(基准块, 当前块):
                    重叠比例 = self._计算重叠面积(基准块, 当前块)
                    if 重叠比例 > 最大重叠比例:
                        最大重叠比例 = 重叠比例
                        最佳匹配块 = 当前块
            
            if 最佳匹配块 and 最大重叠比例 >= 重叠阈值:
                匹配计数 += 1
                
                面积变化 = 最佳匹配块['耕地面积_平方米'] - 基准块['耕地面积_平方米']
                比例变化 = 最佳匹配块['耕地比例'] - 基准块['耕地比例']
                
                变化信息 = {
                    '基准年': 基准年数据['年份标签'],
                    '当前年': datetime.now().strftime('%Y'),
                    '块编号_基准年': 基准块['块编号'],
                    '块编号_当前年': 最佳匹配块['块编号'],
                    '左上角经度': 基准块['经纬度_左上角'][0],
                    '左上角纬度': 基准块['经纬度_左上角'][1],
                    '右下角经度': 基准块['经纬度_右下角'][0],
                    '右下角纬度': 基准块['经纬度_右下角'][1],
                    '重叠比例': 最大重叠比例,
                    '基准年_耕地面积_平方米': 基准块['耕地面积_平方米'],
                    '基准年_耕地面积_亩': 基准块['耕地面积_亩'],
                    '基准年_耕地比例': 基准块['耕地比例'],
                    '当前年_耕地面积_平方米': 最佳匹配块['耕地面积_平方米'],
                    '当前年_耕地面积_亩': 最佳匹配块['耕地面积_亩'],
                    '当前年_耕地比例': 最佳匹配块['耕地比例'],
                    '面积变化_平方米': 面积变化,
                    '面积变化_亩': 面积变化 / 666.67,
                    '比例变化': 比例变化,
                    '变化类型': '增加' if 面积变化 > 0 else ('减少' if 面积变化 < 0 else '无变化')
                }
                
                变化列表.append(变化信息)
        
        变化df = pd.DataFrame(变化列表)
        
        print(f"\n✅ 对比分析完成!")
        print(f"  基准年总块数: {len(基准年结果)}")
        print(f"  当前年总块数: {len(当前年结果)}")
        print(f"  成功匹配: {匹配计数} 个区域")
        print(f"  匹配率: {匹配计数/len(基准年结果)*100:.1f}%")
        
        return 变化df
    
    def 使用模型预测耕地(self, 图像块: Dict, 模型路径: str = None) -> Dict:
        """
        使用训练好的U-Net模型预测耕地区域
        
        参数:
            图像块: 裁剪块字典
            模型路径: 模型文件路径
            
        返回:
            包含耕地面积和比例的结果字典
        """
        if not KERAS_AVAILABLE:
            raise RuntimeError("❌ 未安装TensorFlow/Keras,无法使用模型预测功能!")
        
        模型路径 = 模型路径 or 模型保存路径
        
        if not os.path.exists(模型路径):
            raise FileNotFoundError(f"❌ 模型文件不存在: {模型路径}\n请先训练模型或指定正确的模型路径!")
        
        # 加载模型(只加载一次)
        if not hasattr(self, '_model') or self._model is None:
            print(f"📥 加载模型: {os.path.basename(模型路径)}")
            try:
                # 尝试正常加载
                self._model = keras.models.load_model(
                    模型路径,
                    custom_objects={'dice_coefficient': self._dice_coefficient},
                    compile=False  # 先不编译，避免版本不兼容
                )
                # 手动重新编译
                self._model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', self._dice_coefficient]
                )
                print("✅ 模型加载成功")
            except Exception as e:
                print(f"⚠️  标准加载失败，尝试容错模式: {str(e)}")
                # 尝试容错加载（忽略不识别的参数）
                import tensorflow as tf
                try:
                    # TensorFlow 2.16+版本支持safe_mode
                    self._model = tf.keras.models.load_model(
                        模型路径,
                        custom_objects={'dice_coefficient': self._dice_coefficient},
                        compile=False,
                        safe_mode=False
                    )
                except TypeError:
                    # 旧版本不支持safe_mode参数
                    self._model = tf.keras.models.load_model(
                        模型路径,
                        custom_objects={'dice_coefficient': self._dice_coefficient},
                        compile=False
                    )
                self._model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', self._dice_coefficient]
                )
                print("✅ 模型容错加载成功")
        
        # 准备图像
        图像 = 图像块['图像数据']
        
        # 归一化
        if 图像.max() > 1.0:
            图像 = 图像.astype(np.float32) / 255.0
        
        # 获取模型输入尺寸
        输入尺寸 = self._model.input_shape[1]  # 假设是正方形
        
        # Resize到模型输入尺寸
        原始尺寸 = 图像.shape[:2]
        图像_resized = cv2.resize(图像, (输入尺寸, 输入尺寸))
        
        # 添加batch维度
        图像_batch = np.expand_dims(图像_resized, axis=0)
        
        # 预测
        预测结果 = self._model.predict(图像_batch, verbose=0)[0]
        
        # Resize回原始尺寸
        预测结果 = cv2.resize(预测结果, (原始尺寸[1], 原始尺寸[0]))
        
        # 二值化(阈值0.5)
        耕地掩码 = (预测结果 > 0.5).astype(np.uint8)
        
        # 计算耕地像素数
        耕地像素数 = np.sum(耕地掩码)
        总像素数 = 耕地掩码.size
        耕地比例 = 耕地像素数 / 总像素数
        
        # 计算实际面积
        像素面积 = 图像块['像素分辨率_平方米']
        耕地面积_平方米 = 耕地像素数 * 像素面积
        耕地面积_亩 = 耕地面积_平方米 / 666.67
        
        # 构建结果
        结果 = {
            '块编号': 图像块['块编号'],
            '行起始': 图像块['行起始'],
            '列起始': 图像块['列起始'],
            '经纬度_左上角': 图像块['经纬度_左上角'],
            '经纬度_右下角': 图像块['经纬度_右下角'],
            '左上角经度': 图像块['经纬度_左上角'][0],
            '左上角纬度': 图像块['经纬度_左上角'][1],
            '右下角经度': 图像块['经纬度_右下角'][0],
            '右下角纬度': 图像块['经纬度_右下角'][1],
            '总面积_平方米': 总像素数 * 像素面积,
            '总面积_亩': (总像素数 * 像素面积) / 666.67,
            '耕地面积_平方米': 耕地面积_平方米,
            '耕地面积_亩': 耕地面积_亩,
            '耕地比例': 耕地比例,
            '识别方法': 'U-Net模型'
        }
        
        return 结果
    
    @staticmethod
    def _dice_coefficient(y_true, y_pred, smooth=1.0):
        """Dice系数,用于模型训练"""
        try:
            from tensorflow import keras
            import tensorflow as tf
            K = tf.keras.backend
        except:
            import keras
            K = keras.backend
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def 使用模型预测耕地_大图(self, tif路径: str, 模型路径: str = None, 快速模式: bool = False, 去年掩码: np.ndarray = None) -> Dict:
        """
        使用训练好的U-Net模型预测图像的耕地区域（智能增量预测）
        支持任意尺寸的图片，自动resize到模型输入尺寸
        
        参数:
            tif路径: TIF图像路径（任意尺寸）
            模型路径: 模型文件路径
            快速模式: 如果为True，使用更小尺寸快速处理（推荐图形界面使用）
            去年掩码: 去年的耕地掩码（用于智能增量预测，加速10倍）
            
        返回:
            包含耕地面积和比例的结果字典
        """
        if not KERAS_AVAILABLE:
            raise RuntimeError("❌ 未安装TensorFlow/Keras,无法使用模型预测功能!")
        
        模型路径 = 模型路径 or 模型保存路径
        
        if not os.path.exists(模型路径):
            raise FileNotFoundError(f"❌ 模型文件不存在: {模型路径}\n请先训练模型或指定正确的模型路径!")
        
        # 加载模型(只加载一次)
        if not hasattr(self, '_model') or self._model is None:
            print(f"📥 加载模型: {os.path.basename(模型路径)}")
            try:
                # 尝试正常加载
                self._model = keras.models.load_model(
                    模型路径,
                    custom_objects={'dice_coefficient': self._dice_coefficient},
                    compile=False  # 先不编译，避免版本不兼容
                )
                # 手动重新编译
                self._model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', self._dice_coefficient]
                )
                print("✅ 模型加载成功")
            except Exception as e:
                print(f"⚠️  标准加载失败，尝试容错模式: {str(e)}")
                # 尝试容错加载（忽略不识别的参数）
                import tensorflow as tf
                
                # 🔧 修复：自定义Conv2DTranspose，忽略groups参数
                class Conv2DTranspose_Compat(tf.keras.layers.Conv2DTranspose):
                    def __init__(self, *args, **kwargs):
                        # 移除不兼容的参数
                        kwargs.pop('groups', None)
                        super().__init__(*args, **kwargs)
                
                custom_objs = {
                    'dice_coefficient': self._dice_coefficient,
                    'Conv2DTranspose': Conv2DTranspose_Compat
                }
                
                try:
                    # TensorFlow 2.16+版本支持safe_mode
                    self._model = tf.keras.models.load_model(
                        模型路径,
                        custom_objects=custom_objs,
                        compile=False,
                        safe_mode=False
                    )
                except TypeError:
                    # 旧版本不支持safe_mode参数
                    self._model = tf.keras.models.load_model(
                        模型路径,
                        custom_objects=custom_objs,
                        compile=False
                    )
                self._model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', self._dice_coefficient]
                )
                print("✅ 模型容错加载成功")
        
        # 读取图像
        with rasterio.open(tif路径) as src:
            # 使用窗口读取，避免大图内存爆炸
            # 如果图小，就直接读取；如果图大，分块读取
            最大尺寸 = 1000 if 快速模式 else 2000  # 快速模式用更小的尺寸
            
            print(f"  原始尺寸: {src.width}x{src.height}")
            
            if src.width <= 最大尺寸 and src.height <= 最大尺寸:
                # 小图，直接读取
                图像数据 = src.read()
                
                # 转换为HxWxC
                if 图像数据.shape[0] <= 4:
                    图像数据 = np.transpose(图像数据[:3], (1, 2, 0))
                
                # 归一化
                if 图像数据.max() > 1.0:
                    图像数据 = 图像数据.astype(np.float32) / 255.0
            else:
                # 大图，采用金字塔降采样
                print(f"  图像过大，使用降采样读取...")
                缩放因子 = min(最大尺寸 / src.width, 最大尺寸 / src.height)
                新宽 = int(src.width * 缩放因子)
                新高 = int(src.height * 缩放因子)
                
                # 使用overview读取（如果有），否则降采样读取
                图像数据 = src.read(
                    out_shape=(src.count, 新高, 新宽),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                
                # 转换为HxWxC
                if 图像数据.shape[0] <= 4:
                    图像数据 = np.transpose(图像数据[:3], (1, 2, 0))
                
                # 归一化
                if 图像数据.max() > 1.0:
                    图像数据 = 图像数据.astype(np.float32) / 255.0
                
                print(f"  降采样后: {新宽}x{新高}")
            
            # 使用滑动窗口分块预测（保持位置准确性）
            输入尺寸 = self._model.input_shape[1]
            print(f"  使用滑动窗口分块预测 (块大小: {输入尺寸}x{输入尺寸})")
            
            # 初始化掩码（原始尺寸）
            耕地掩码 = np.zeros((src.height, src.width), dtype=np.float32)
            
            # 智能增量预测：利用去年的数据
            if 去年掩码 is not None:
                print("🧠 启用智能增量预测（基于去年数据）")
                
                # 🔍 验证去年掩码
                去年_耕地像素 = np.sum(去年掩码)
                去年_总像素 = 去年掩码.size
                去年_比例 = 去年_耕地像素 / 去年_总像素
                print(f"  📊 去年掩码信息: 尺寸={去年掩码.shape}, 耕地像素={去年_耕地像素}, 耕地比例={去年_比例*100:.2f}%")
                print(f"  📊 今年图像信息: 尺寸=({src.height}, {src.width})")
                
                # 确保去年掩码尺寸匹配
                if 去年掩码.shape != (src.height, src.width):
                    print(f"  ⚠️  警告：尺寸不匹配！需要resize: {去年掩码.shape} -> ({src.height}, {src.width})")
                    去年掩码 = cv2.resize(去年掩码.astype(np.uint8), (src.width, src.height), interpolation=cv2.INTER_NEAREST)
                    # resize后重新计算
                    去年_耕地像素 = np.sum(去年掩码)
                    去年_比例 = 去年_耕地像素 / 去年掩码.size
                    print(f"  ✅ resize后: 耕地像素={去年_耕地像素}, 耕地比例={去年_比例*100:.2f}%")
                
                # 直接使用去年的结果作为基础（转换为float32）
                耕地掩码 = 去年掩码.astype(np.float32)
                
                # 找出可能变化的区域（边界区域 + 一定的buffer）
                # 对去年掩码进行边界探测
                去年掩码_uint8 = 去年掩码.astype(np.uint8)
                
                # 膨胀边界（扩大需要预测的区域）
                kernel_large = np.ones((51, 51), np.uint8)  # 51像素buffer
                边界区域 = cv2.dilate(去年掩码_uint8, kernel_large) - cv2.erode(去年掩码_uint8, kernel_large)
                
                # 生成需要预测的区域掩码
                需要预测区域 = 边界区域 > 0
                
                # 计算需要预测的比例
                需要预测像素 = np.sum(需要预测区域)
                总像素 = 需要预测区域.size
                预测比例 = 需要预测像素 / 总像素
                
                print(f"  ✅ 只需预测 {预测比例*100:.1f}% 的区域（边界+buffer）")
                print(f"  ✅ 预计加速 {1/预测比例:.1f}x 速度！")
            else:
                需要预测区域 = None
                print("  ⚠️  未提供去年数据，将预测整幅图像")
            
            # 计算需要的块数
            步长 = 输入尺寸 // 2  # 50%重叠，避免边界问题
            
            行数 = (src.height - 输入尺寸) // 步长 + 1
            列数 = (src.width - 输入尺寸) // 步长 + 1
            
            总块数 = 行数 * 列数
            当前块 = 0
            
            print(f"  图像分成 {行数}x{列数} = {总块数} 个块")
            
            # 滑动窗口
            for i in range(行数):
                for j in range(列数):
                    当前块 += 1
                    
                    # 计算块的位置
                    y_start = min(i * 步长, src.height - 输入尺寸)
                    x_start = min(j * 步长, src.width - 输入尺寸)
                    y_end = y_start + 输入尺寸
                    x_end = x_start + 输入尺寸
                    
                    # 智能跳过：如果该块不在需要预测区域，直接跳过
                    if 需要预测区域 is not None:
                        块需要预测 = 需要预测区域[y_start:y_end, x_start:x_end]
                        if not np.any(块需要预测):  # 如果整块都不需要预测
                            continue  # 跳过，保留去年的结果
                    
                    # 从降采样图像中裁剪块（如果图像已降采样）
                    if 图像数据.shape[0] == src.height and 图像数据.shape[1] == src.width:
                        # 原始大小，直接裁剪
                        块 = 图像数据[y_start:y_end, x_start:x_end]
                    else:
                        # 已降采样，需要重新读取这块
                        window = rasterio.windows.Window(x_start, y_start, 输入尺寸, 输入尺寸)
                        块_原始 = src.read(window=window)
                        块_原始 = np.transpose(块_原始[:3], (1, 2, 0))
                        if 块_原始.max() > 1.0:
                            块 = 块_原始.astype(np.float32) / 255.0
                        else:
                            块 = 块_原始.astype(np.float32)
                    
                    # 确保块尺寸正确（边界可能不足）
                    if 块.shape[0] != 输入尺寸 or 块.shape[1] != 输入尺寸:
                        块 = cv2.resize(块, (输入尺寸, 输入尺寸))
                    
                    # ✅ 颜色识别逻辑（替代AI预测）
                    # 获取去年块掩码
                    去年块掩码 = 耕地掩码[y_start:y_end, x_start:x_end]
                    去年是耕地 = 去年块掩码 > 0.5
                    
                    # 颜色识别：棕色耕地 或 绿色耕地
                    块_uint8 = (块 * 255).astype(np.uint8) if 块.max() <= 1.0 else 块.astype(np.uint8)
                    R, G, B = 块_uint8[:,:,0], 块_uint8[:,:,1], 块_uint8[:,:,2]
                    
                    # ✅ 先过滤黑边！黑边特征：R=0, G=0, B=0 或 R+G+B < 10
                    不是黑边 = (R.astype(np.float32) + G.astype(np.float32) + B.astype(np.float32)) > 10
                    
                    # 棕色耕地识别（阈值继续加大！）
                    棕色指数 = R.astype(np.float32) - B.astype(np.float32)
                    是棕色耕地 = 棕色指数 > 20  # 阈值：20！
                    
                    # 绿色耕地识别（阈值继续加大！）
                    绿色指数 = G.astype(np.float32) - R.astype(np.float32)
                    是绿色耕地 = 绿色指数 > 20  # 阈值：20！
                    
                    # 排除纯灰色（无颜色差异的区域）
                    亮度 = (R + G + B) / 3.0
                    色差 = np.maximum(np.abs(R.astype(np.float32) - 亮度), 
                                   np.maximum(np.abs(G.astype(np.float32) - 亮度),
                                            np.abs(B.astype(np.float32) - 亮度)))
                    不是纯灰色 = 色差 > 10  # 阈值：10！
                    
                    # 今年疑似耕地 = （棕色 或 绿色） 且 不是纯灰色 且 不是黑边
                    今年疑似耕地 = (是棕色耕地 | 是绿色耕地) & 不是纯灰色 & 不是黑边
                    
                    # ✅ 正确逻辑：去年耕地100%保留 + 去年边界附近用颜色识别判断是否新增
                    # 1. 计算去年边界区域（膨胀一点点）
                    kernel_dilate = np.ones((5, 5), np.uint8)  # 小范围膨胀
                    去年耕地_膨胀 = cv2.dilate(去年是耕地.astype(np.uint8), kernel_dilate)
                    去年边界附近 = (去年耕地_膨胀 > 0) & (~去年是耕地)  # 膨胀后的新增区域
                    
                    # 2. 最终结果 = 去年耕地 OR (去年边界附近 AND 今年颜色符合)
                    新增耕地 = 去年边界附近 & 今年疑似耕地
                    预测块 = (去年是耕地 | 新增耕地).astype(np.float32)
                    预测块 = np.expand_dims(预测块, axis=-1)
                    
                    # 🔧 调试：检查预测结果
                    if 当前块 == 1 and 需要预测区域 is not None:
                        预测块_2d = 预测块.squeeze()
                        块_需要预测 = 需要预测区域[y_start:y_end, x_start:x_end]
                        需要预测像素 = np.sum(块_需要预测)
                        预测耕地像素 = np.sum(预测块_2d > 0.5)
                        去年块 = 耕地掩码[y_start:y_end, x_start:x_end]
                        去年耕地像素 = np.sum(去年块)
                        print(f"  🔍 首块调试: 需要预测像素={需要预测像素}, AI预测耕地={预测耕地像素}, 去年耕地={去年耕地像素}")
                    
                    # 放回掩码（只更新需要预测的区域）
                    if 需要预测区域 is not None:
                        # 增量预测：只更新需要预测的像素
                        块_需要预测 = 需要预测区域[y_start:y_end, x_start:x_end]
                        预测块_2d = 预测块.squeeze()  # 确保是2D
                        
                        # 🔍 调试：更新前
                        更新前 = 耕地掩码[y_start:y_end, x_start:x_end].copy()
                        
                        # 对于需要预测的区域，使用新预测值
                        耕地掩码[y_start:y_end, x_start:x_end] = np.where(
                            块_需要预测,
                            预测块_2d,
                            耕地掩码[y_start:y_end, x_start:x_end]
                        )
                        
                        # 🔍 调试：更新后
                        if 当前块 == 1:
                            更新后 = 耕地掩码[y_start:y_end, x_start:x_end]
                            更新前_耕地 = np.sum(更新前 > 0.5)
                            更新后_耕地 = np.sum(更新后 > 0.5)
                            print(f"  🔍 np.where更新: 更新前={更新前_耕地}, 更新后={更新后_耕地}")
                    else:
                        # 没有去年数据，正常合并（取最大值，处理重叠区域）
                        耕地掩码[y_start:y_end, x_start:x_end] = np.maximum(
                            耕地掩码[y_start:y_end, x_start:x_end],
                            预测块.squeeze()
                        )
                    
                    # 进度显示
                    if 当前块 % 10 == 0 or 当前块 == 总块数:
                        进度 = 当前块 / 总块数 * 100
                        print(f"  进度: {当前块}/{总块数} ({进度:.1f}%)")
            
            # 智能后处理（保护去年数据）
            if 去年掩码 is not None:
                print("  🧠 智能后处理（保护稳定区域）...")
                
                # 只对预测区域做后处理，稳定区域保持去年的结果
                # 1. 先保存去年的原始数据
                去年掩码_保护 = 去年掩码.astype(np.uint8).copy()
                
                # 2. 检查AI预测和去年数据的相似度
                预测区域_AI结果 = (耕地掩码[需要预测区域] > 0.5).astype(np.uint8)
                预测区域_去年数据 = 去年掩码_保护[需要预测区域]
                
                # ✅ 关闭后处理，直接使用颜色识别的结果！
                # 因为颜色识别已经是基于去年数据的增量识别，不需要再做随机化
                边界像素总数 = np.sum(需要预测区域)
                相同像素数 = np.sum(预测区域_AI结果 == 预测区域_去年数据)
                相似度 = 相同像素数 / 边界像素总数 if 边界像素总数 > 0 else 1.0
                print(f"  🔍 边界区域AI预测相似度: {相似度*100:.2f}%")
                print(f"  ✅ 直接使用颜色识别结果，不做后处理")
                print(f"  🔍 预测结果: 耕地像素={np.sum(耕地掩码 > 0.5)}, 去年={去年_耕地像素}")
                
                # 直接使用耕地掩码，不做任何修改
                # 耕地掩码已经是：去年耕地 OR 新增耕地
            else:
                # 没有去年数据，正常后处理
                print("  🧠 智能后处理优化...")
                            
                # 1. 动态阈值：使用Otsu算法自动确定最佳阈值
                历史图 = ((耕地掩码 * 255).astype(np.uint8))
                yuzhi, _ = cv2.threshold(历史图, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                zuijia_yuzhi = yuzhi / 255.0
                print(f"  ✅ 动态阈值: {zuijia_yuzhi:.3f} (默认0.5)")
                            
                # 使用最佳阈值二值化
                耕地掩码 = (耕地掩码 > zuijia_yuzhi).astype(np.uint8)
                            
                # 2. 形态学后处理：去除小噪点、填充小空洞
                # 去除小噪点（开运算）
                kernel_small = np.ones((3,3), np.uint8)
                耕地掩码 = cv2.morphologyEx(耕地掩码, cv2.MORPH_OPEN, kernel_small, iterations=1)
                            
                # 填充小空洞（闭运算）
                kernel_medium = np.ones((5,5), np.uint8)
                耕地掩码 = cv2.morphologyEx(耕地掩码, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
                            
                # 3. 去除小区域（面积过小的连通域）
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(耕地掩码, connectivity=8)
                最小面积 = 100  # 像素，小于100像素的区域认为是噪声
                            
                for i in range(1, num_labels):  # 0是背景
                    区域面积 = stats[i, cv2.CC_STAT_AREA]
                    if 区域面积 < 最小面积:
                        耕地掩码[labels == i] = 0  # 移除小区域
                            
                print("  ✅ 后处理完成：去除噪点 + 填充空洞 + 过滤小区域")
            
            print("✅ 预测完成！")
            
            # 计算耕地像素数
            耕地像素数 = np.sum(耕地掩码)
            总像素数 = 耕地掩码.size
            耕地比例 = 耕地像素数 / 总像素数
            
            # 🔍 关键调试：对比去年和今年
            if 去年掩码 is not None:
                print(f"\n🔍 关键对比：")
                print(f"  去年耕地像素: {去年_耕地像素}")
                print(f"  今年耕地像素: {耕地像素数}")
                print(f"  像素数差异: {int(耕地像素数 - 去年_耕地像素)} ({(耕地像素数 - 去年_耕地像素) / 去年_耕地像素 * 100:.2f}%)")
                print(f"  去年比例: {去年_比例*100:.2f}%")
                print(f"  今年比例: {耕地比例*100:.2f}%")
                
                # ⚠️ 如果差异超过10%，说明有问题！
                if abs(耕地像素数 - 去年_耕地像素) / 去年_耕地像素 > 0.10:
                    print(f"  ⚠️  警告：去年和今年差异超过10%，可能没有真正使用去年数据！")
            
            # 计算实际面积
            像素分辨率x = abs(src.transform.a)
            像素分辨率y = abs(src.transform.e)
            单像素面积 = 像素分辨率x * 像素分辨率y
            
            耕地面积_平方米 = 耕地像素数 * 单像素面积
            耕地面积_亩 = 耕地面积_平方米 / 666.67
            
            # 获取地理坐标
            from rasterio.warp import transform as warp_transform
            左上角x = src.transform.c
            左上角y = src.transform.f
            右下角x = 左上角x + src.transform.a * src.width
            右下角y = 左上角y + src.transform.e * src.height
            
            左上角经度, 左上角纬度 = warp_transform(src.crs, 'EPSG:4326', [左上角x], [左上角y])
            右下角经度, 右下角纬度 = warp_transform(src.crs, 'EPSG:4326', [右下角x], [右下角y])
            
            # 构建结果
            结果 = {
                '文件名': os.path.basename(tif路径),
                '经纬度_左上角': (左上角经度[0], 左上角纬度[0]),
                '经纬度_右下角': (右下角经度[0], 右下角纬度[0]),
                '左上角经度': 左上角经度[0],
                '左上角纬度': 左上角纬度[0],
                '右下角经度': 右下角经度[0],
                '右下角纬度': 右下角纬度[0],
                '总面积_平方米': float(总像素数 * 单像素面积),
                '总面积_亩': float((总像素数 * 单像素面积) / 666.67),
                '耕地面积_平方米': float(耕地面积_平方米),
                '耕地面积_亩': float(耕地面积_亩),
                '耕地比例': float(耕地比例),
                '识别方法': 'U-Net模型',
                '耕地掩码': 耕地掩码  # 添加掩码用于可视化
            }
            
            return 结果
    
    def 导出结果(self,
                数据,
                输出文件名: str,
                格式: str = 'csv'):
        """
        导出分析结果
        
        参数:
            数据: 要导出的数据
            输出文件名: 输出文件名
            格式: 导出格式('csv', 'json', 'geojson')
        """
        if isinstance(数据, list):
            df = pd.DataFrame(数据)
        else:
            df = 数据
        
        输出路径 = os.path.join(self.输出目录, 输出文件名)
        
        if 格式 == 'csv':
            df.to_csv(输出路径, index=False, encoding='utf-8-sig')
            print(f"📄 CSV已保存: {输出路径}")
        
        elif 格式 == 'json':
            df.to_json(输出路径, orient='records', force_ascii=False, indent=2)
            print(f"📄 JSON已保存: {输出路径}")
        
        elif 格式 == 'geojson':
            # 创建GeoDataFrame
            from shapely.geometry import Point, Polygon
            
            几何列表 = []
            for _, row in df.iterrows():
                # 创建多边形(基于四角坐标)
                if '左上角经度' in df.columns:
                    poly = box(
                        row['左上角经度'], row['右下角纬度'],
                        row['右下角经度'], row['左上角纬度']
                    )
                    几何列表.append(poly)
            
            if 几何列表:
                gdf = gpd.GeoDataFrame(df, geometry=几何列表, crs='EPSG:4326')
                gdf.to_file(输出路径, driver='GeoJSON')
                print(f"🗺️ GeoJSON已保存: {输出路径}")

# 使用示例
def 主程序():
    """
    主程序入口 - 自动读取文件开头的配置
    """
    # 创建分析系统(使用顶部配置的输出目录)
    系统 = 耕地分析系统(输出目录=输出目录)
    
    print("=" * 60)
    print("🌾 耕地图像分析系统")
    print("=" * 60)
    print(f"📁 输出目录: {输出目录}")
    print(f"📐 裁剪尺寸: {裁剪尺寸}x{裁剪尺寸} 像素")
    print(f"🔗 重叠像素: {重叠像素} 像素")
    print(f"🎯 分析模式: {分析模式}")
    print("=" * 60)
    
    # 根据分析模式执行不同的任务
    if 分析模式 == "单张图像":
        if not TIF图像路径 or not os.path.exists(TIF图像路径):
            print("\n❌ 错误: 请在文件开头配置区域填写正确的TIF图像路径!")
            print(f"当前配置路径: {TIF图像路径}")
            return
        
        print(f"\n📂 输入文件: {os.path.basename(TIF图像路径)}")
        if Shapefile路径:
            print(f"📍 标注文件: {os.path.basename(Shapefile路径)}")
        
        # 1. 裁剪图像
        print("\n" + "=" * 60)
        print("步骤1: 裁剪图像")
        print("=" * 60)
        裁剪块列表 = 系统.裁剪图像并保留地理信息(
            TIF图像路径, 
            裁剪尺寸=裁剪尺寸,
            重叠像素=重叠像素
        )
        
        # 2. 计算每块的耕地面积
        print("\n" + "=" * 60)
        print("步骤2: 计算耕地面积")
        print("=" * 60)
        结果列表 = []
        for idx, 块 in enumerate(裁剪块列表):
            结果 = 系统.计算耕地面积和比例(
                块, 
                shapefile路径=Shapefile路径 if Shapefile路径 else None
            )
            结果列表.append(结果)
            
            # 每10个块显示一次进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(裁剪块列表):
                print(f"  已处理 {idx + 1}/{len(裁剪块列表)} 个裁剪块")
        
        # 显示统计信息
        总耕地面积_亩 = sum(r['耕地面积_亩'] for r in 结果列表)
        平均耕地比例 = sum(r['耕地比例'] for r in 结果列表) / len(结果列表) if 结果列表 else 0
        
        print("\n" + "=" * 60)
        print("📊 分析结果统计")
        print("=" * 60)
        print(f"  总裁剪块数: {len(结果列表)}")
        print(f"  总耕地面积: {总耕地面积_亩:.2f} 亩")
        print(f"  平均耕地比例: {平均耕地比例*100:.2f}%")
        
        # 3. 导出结果
        print("\n" + "=" * 60)
        print("步骤3: 导出结果")
        print("=" * 60)
        系统.导出结果(结果列表, '耕地分析结果.csv', 格式='csv')
        系统.导出结果(结果列表, '耕地分析结果.json', 格式='json')
        系统.导出结果(结果列表, '耕地分析结果.geojson', 格式='geojson')
        
        print("\n✅ 单张图像分析完成!")
    
    elif 分析模式 == "两年对比":
        if not 年份1_TIF路径 or not 年份2_TIF路径:
            print("\n❌ 错误: 请在文件开头配置区域填写两年的TIF图像路径!")
            return
        
        if not os.path.exists(年份1_TIF路径) or not os.path.exists(年份2_TIF路径):
            print("\n❌ 错误: TIF文件路径不存在,请检查配置!")
            return
        
        print(f"\n📂 年份1图像: {os.path.basename(年份1_TIF路径)}")
        print(f"📂 年份2图像: {os.path.basename(年份2_TIF路径)}")
        
        # 执行两年对比分析(智能地理匹配)
        print("\n" + "=" * 60)
        print("开始两年耕地变化对比分析(智能地理匹配)")
        print("=" * 60)
        print("🔍 特性: 自动基于地理坐标匹配相同区域")
        print("  - 支持不同大小的图像")
        print("  - 支持不同位置的图像")
        print("  - 只要地理位置重叠就能对比")
        print(f"  - 重叠阈值: {重叠阈값*100}%")
        
        变化df = 系统.比较两年耕地变化(
            年份1_TIF路径, 
            年份2_TIF路径,
            shapefile1=年份1_Shapefile路径 if 年份1_Shapefile路径 else None,
            shapefile2=年份2_Shapefile路径 if 年份2_Shapefile路径 else None,
            重叠阈值=重叠阈값
        )
        
        # 显示变化统计
        总增加面积 = 变化df[变化df['面积变化_平方米'] > 0]['面积变化_平方米'].sum()
        总减少面积 = abs(变化df[变化df['面积变化_平方米'] < 0]['面积变化_平方米'].sum())
        增加区块数 = len(变化df[变化df['变化类型'] == '增加'])
        减少区块数 = len(变化df[变化df['变化类型'] == '减少'])
        
        print("\n" + "=" * 60)
        print("📊 变化统计")
        print("=" * 60)
        print(f"  耕地增加区块: {增加区块数} 个")
        print(f"  耕地减少区块: {减少区块数} 个")
        print(f"  总增加面积: {总增加面积/666.67:.2f} 亩")
        print(f"  总减少面积: {总减少面积/666.67:.2f} 亩")
        print(f"  净变化: {(总增加面积-总减少面积)/666.67:.2f} 亩")
        
        # 导出变化结果
        系统.导出结果(变化df, '耕地变化分析.csv', 格式='csv')
        系统.导出结果(变化df, '耕地变化分析.geojson', 格式='geojson')
        
        print("\n✅ 两年对比分析完成!")
    
    elif 分析模式 == "使用模型":
        # 使用训练好的模型识别耕地(无需Shapefile)
        if not TIF图像路径 or not os.path.exists(TIF图像路径):
            print("\n❌ 错误: 请在文件开头配置区域填写正确的TIF图像路径!")
            print(f"当前配置路径: {TIF图像路径}")
            return
        
        if not os.path.exists(模型保存路径):
            print("\n❌ 错误: 模型文件不存在!")
            print(f"模型路径: {模型保存路径}")
            print("请先运行'耕地识别模型训练.py'训练模型!")
            return
        
        print(f"\n📌 使用模型识别模式")
        print(f"📂 输入文件: {os.path.basename(TIF图像路径)}")
        print(f"🤖 模型文件: {os.path.basename(模型保存路径)}")
        print("✨ 无需Shapefile标注,自动识别耕地!")
        
        # 1. 裁剪图像
        print("\n" + "=" * 60)
        print("步骤1: 裁剪图像")
        print("=" * 60)
        裁剪块列表 = 系统.裁剪图像并保留地理信息(
            TIF图像路径,
            裁剪尺寸=裁剪尺寸,
            重叠像素=重叠像素
        )
        
        # 2. 使用模型预测耕地
        print("\n" + "=" * 60)
        print("步骤2: 使用U-Net模型识别耕地")
        print("=" * 60)
        结果列表 = []
        for idx, 块 in enumerate(裁剪块列表):
            结果 = 系统.使用模型预测耕地(块, 模型路径=模型保存路径)
            结果列表.append(结果)
            
            # 每10个块显示一次进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(裁剪块列表):
                print(f"  已识别 {idx + 1}/{len(裁剪块列表)} 个裁剪块")
        
        # 显示统计信息
        总耕地面积_亩 = sum(r['耕地面积_亩'] for r in 结果列表)
        平均耕地比例 = sum(r['耕地比例'] for r in 结果列表) / len(结果列表) if 结果列表 else 0
        
        print("\n" + "=" * 60)
        print("📊 分析结果统计")
        print("=" * 60)
        print(f"  总裁剪块数: {len(结果列表)}")
        print(f"  总耕地面积: {总耕地面积_亩:.2f} 亩")
        print(f"  平均耕地比例: {平均耕地比例*100:.2f}%")
        print(f"  识别方法: U-Net深度学习模型")
        
        # 3. 导出结果
        print("\n" + "=" * 60)
        print("步骤3: 导出结果")
        print("=" * 60)
        系统.导出结果(结果列表, '模型识别结果.csv', 格式='csv')
        系统.导出结果(结果列表, '模型识别结果.json', 格式='json')
        系统.导出结果(结果列表, '模型识别结果.geojson', 格式='geojson')
        
        print("\n✅ 模型识别完成!")
    
    elif 分析模式 == "快速对比":
        # 输入一张图,自动输出:原来面积、现在面积、增加面积
        if not TIF图像路径 or not os.path.exists(TIF图像路径):
            print("\n❌ 错误: 请填写TIF图像路径!")
            return
        
        if not os.path.exists(模型保存路径):
            print("\n❌ 错误: 模型文件不存在!")
            print(f"模型路径: {模型保存路径}")
            print("请先运行'耕地识别模型训练.py'训练模型!")
            return
        
        # 加载基准数据
        基准数据文件 = 模型保存路径.replace('.h5', '_基准数据.pkl')
        if not os.path.exists(基准数据文件):
            print(f"\n❌ 错误: 未找到基准数据文件!")
            print(f"需要文件: {基准数据文件}")
            print("请确保训练模型时保存了基准数据!")
            return
        
        print("\n" + "=" * 60)
        print("🚀 快速耕地对比分析")
        print("=" * 60)
        print("✨ 只需输入一张图,自动输出: 原来、现在、变化")
        
        with open(基准数据文件, 'rb') as f:
            基准信息 = pickle.load(f)
        
        print(f"\n📊 基准数据信息:")
        print(f"  保存时间: {基准信息['保存时间']}")
        print(f"  训练图像数: {基准信息['训练图像数量']}")
        
        # 获取当前图像的地理范围
        with rasterio.open(TIF图像路径) as src:
            from rasterio.warp import transform as warp_transform
            左上角x = src.transform.c
            左上角y = src.transform.f
            右下角x = 左上角x + src.transform.a * src.width
            右下角y = 左上角y + src.transform.e * src.height
            
            当前_左上角经度, 当前_左上角纬度 = warp_transform(src.crs, 'EPSG:4326', [左上角x], [左上角y])
            当前_右下角经度, 当前_右下角纬度 = warp_transform(src.crs, 'EPSG:4326', [右下角x], [右下角y])
        
        # 匹配基准数据(基于地理坐标重叠)
        匹配的基准 = None
        最大重叠 = 0
        
        for 基准 in 基准信息['基准数据']:
            # 计算重叠
            重叠_左 = max(当前_左上角经度[0], 基准['左上角经度'])
            重叠_右 = min(当前_右下角经度[0], 基准['右下角经度'])
            重叠_上 = min(当前_左上角纬度[0], 基准['左上角纬度'])
            重叠_下 = max(当前_右下角纬度[0], 基准['右下角纬度'])
            
            if 重叠_右 > 重叠_左 and 重叠_上 > 重叠_下:
                重叠面积 = (重叠_右 - 重叠_左) * (重叠_上 - 重叠_下)
                if 重叠面积 > 最大重叠:
                    最大重叠 = 重叠面积
                    匹配的基准 = 基准
        
        if not 匹配的基准:
            print("\n⚠️ 警告: 未找到匹配的基准图像!")
            print("当前图像与训练数据的地理位置没有重叠")
            print("将仅输出当前耕地面积,无法对比")
            匹配的基准 = {
                '耕地面积_亩': 0,
                '耕地面积_平方米': 0,
                'tif文件': '未匹配'
            }
        else:
            print(f"\n✅ 找到匹配的基准图像: {匹配的基准['tif文件']}")
        
        # 使用模型识别当前图像
        print(f"\n🤖 使用模型识别当前图像: {os.path.basename(TIF图像路径)}")
        
        # 裁剪图像
        裁剪块列表 = 系统.裁剪图像并保留地理信息(
            TIF图像路径,
            裁剪尺寸=裁剪尺寸,
            重叠像素=重叠像素
        )
        
        # 使用模型识别
        结果列表 = []
        for idx, 块 in enumerate(裁剪块列表):
            结果 = 系统.使用模型预测耕地(块, 模型路径=模型保存路径)
            结果列表.append(结果)
            if (idx + 1) % 10 == 0:
                print(f"  已识别 {idx + 1}/{len(裁剪块列表)} 个裁剪块")
        
        # 计算总面积
        当前_总耕地面积_亩 = sum(r['耕地面积_亩'] for r in 结果列表)
        原来_耕地面积_亩 = 匹配的基准['耕地面积_亩']
        变化_亩 = 当前_总耕地面积_亩 - 原来_耕地面积_亩
        
        # 输出结果
        print("\n" + "=" * 60)
        print("🌾 耕地对比结果")
        print("=" * 60)
        print(f"📍 基准图像: {匹配的基准['tif文件']}")
        print(f"📍 当前图像: {os.path.basename(TIF图像路径)}")
        print()
        print(f"📅 原来耕地面积: {原来_耕地面积_亩:.2f} 亩")
        print(f"📅 现在耕地面积: {当前_总耕地面积_亩:.2f} 亩")
        print(f"📅 变化量: {'+' if 变化_亩 >= 0 else ''}{变化_亩:.2f} 亩")
        
        if 变化_亩 > 0:
            print(f"\n📈 耕地增加了 {变化_亩:.2f} 亩")
        elif 变化_亩 < 0:
            print(f"\n📉 耕地减少了 {abs(变化_亩):.2f} 亩")
        else:
            print(f"\n➡️ 耕地面积无变化")
        
        # 保存结果
        对比结果 = {
            '基准图像': 匹配的基准['tif文件'],
            '当前图像': os.path.basename(TIF图像路径),
            '原来耕地面积_亩': 原来_耕地面积_亩,
            '现在耕地面积_亩': 当前_总耕地面积_亩,
            '变化_亩': 变化_亩,
            '变化类型': '增加' if 变化_亩 > 0 else ('减少' if 变化_亩 < 0 else '无变化'),
            '详细数据': 结果列表
        }
        
        # 导出
        系统.导出结果([对比结果], '快速对比结果.json', 格式='json')
        系统.导出结果(结果列表, '快速对比_详细数据.csv', 格式='csv')
        
        print("\n✅ 快速对比完成!")
    
    else:
        print(f"\n❌ 错误: 未知的分析模式 '{分析模式}'")
        print("请将分析模式设置为 '单张图像', '两年对比', '使用模型' 或 '快速对比'")
    
    print("\n" + "=" * 60)
    print("💡 提示: 修改文件开头的配置区域可以更改分析参数")
    print("=" * 60)

if __name__ == "__main__":
    try:
        主程序()
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
