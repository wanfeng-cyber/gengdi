"""
耕地识别模型训练模块 - U-Net
使用TIF图像和Shapefile标注训练深度学习模型
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
import cv2
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

# ==================== GPU加速配置 ====================
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
        
        # 启用GPU内存动态增长（避免占满显存）
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ 已启用GPU内存动态增长")
        except RuntimeError as e:
            print(f"⚠️  GPU配置警告: {e}")
        
        # 设置混合精度训练（加速）
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ 已启用混合精度训练（FP16加速）")
        except:
            print("⚠️  混合精度训练不可用（TensorFlow版本较旧）")
        
        print("\n🎯 训练将使用GPU加速！")
    else:
        print("❌ 未检测到GPU，将使用CPU训练（速度较慢）")
        print("   建议安装CUDA和cuDNN以启用GPU加速")
except ImportError:
    print("❌ 未安装TensorFlow，无法检测GPU")

print("="*60)
print()

# ==================== 配置区域 ====================

# ✅ 训练模式选择
训练模式 = "递归逐个"  # "递归逐个" = 递归扫描所有子文件夹，逐个训练TIF / "普通" = 一次性训练所有

# 📁 训练数据路径
训练图像目录 = r"E:\八二\20250420通北八二2_3" # 会递归扫描所有子文件夹
训练标注目录 = r"E:\通北局种植作物\通北局种植作物.shp" # 单个SHP文件路径

# 模型保存路径  
模型保存路径 = r"E:\耕地分析系统_绿色完整版_20251128_143426\耕地识别模型.h5"

# 训练参数（快速训练版 - 30分钟内完成）
图像尺寸 = 256  # 256×256
批次大小 = 4   # 批次4（GPU内存允许，加快训练）
训练轮数 = 50  # 50轮快速训练（30分钟内）
验证比例 = 0.15  # 15%验证集
学习率 = 0.001  # 适中学习率，快速收敛

# =================================================

def 构建UNet模型(输入尺寸=(256, 256, 3)):
    """
    构建U-Net模型用于语义分割
    
    参数:
        输入尺寸: (高度, 宽度, 通道数)
    
    返回:
        编译好的U-Net模型
    """



    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("❌ 未安装TensorFlow,正在尝试导入Keras...")
        import keras
        from keras import layers
    
    inputs = keras.Input(shape=输入尺寸)
    
    # 编码器(下采样路径)
    # 64
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # 128
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # 256
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # 512
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # 底部
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # 解码器(上采样路径)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # 输出层
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    # 编译模型（使用全局学习率）
    try:
        from tensorflow import keras
        optimizer = keras.optimizers.Adam(learning_rate=学习率)
    except:
        optimizer = 'adam'
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coefficient]
    )
    
    return model


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Dice系数(F1 Score的一种形式),用于评估分割效果
    """
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


def 从TIF和Shapefile生成训练数据(tif路径, shapefile路径, 目标尺寸=256):
    """
    从TIF图像和Shapefile标注生成训练数据（大图采样裁剪）
    针对10GB+的超大TIF图，采用随机采样小块的方式，避免内存爆炸
    
    参数:
        tif路径: TIF图像文件路径
        shapefile路径: Shapefile标注文件路径
        目标尺寸: 训练图像大小（裁剪块的尺寸）
    
    返回:
        (图像数组, 标签数组, 基准耕地数据)
    """
    print(f"📖 读取: {os.path.basename(tif路径)}")
    
    # 读取TIF图像
    with rasterio.open(tif路径) as src:
        print(f"  原始尺寸: {src.width}x{src.height}")
        print(f"  波段数: {src.count}")
        
        # 读取Shapefile生成标签
        gdf = gpd.read_file(shapefile路径)
        
        # 确保坐标系一致
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # 获取TIF的地理边界
        from shapely.geometry import box
        tif边界 = box(
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top
        )
        
        # 裁剪SHP到TIF范围（关键优化！）
        print(f"  原始SHP几何数量: {len(gdf)}")
        gdf = gdf[gdf.intersects(tif边界)]  # 只保留与TIF相交的几何
        print(f"  裁剪后SHP几何数量: {len(gdf)}")
        
        if len(gdf) == 0:
            print(f"  ⚠️  警告: 该TIF区域无耕地标注")
            return [], [], {
                'tif文件': os.path.basename(tif路径),
                'tif完整路径': tif路径,
                '耕地面积_亩': 0.0,
            }
        
        # 计算基准耕地面积（使用几何计算，不生成掩码）
        像素分辨率x = abs(src.transform.a)
        像素分辨率y = abs(src.transform.e)
        单像素面积 = 像素分辨率x * 像素分辨率y
        
        # 直接使用GeoDataFrame的面积计算
        总耕地面积_平方米 = gdf.geometry.area.sum()
        总耕地面积_亩 = 总耕地面积_平方米 / 666.67
        
        # 获取图像地理范围
        from rasterio.warp import transform as warp_transform
        左上角x = src.transform.c
        左上角y = src.transform.f
        右下角x = 左上角x + src.transform.a * src.width
        右下角y = 左上角y + src.transform.e * src.height
        
        左上角经度, 左上角纬度 = warp_transform(src.crs, 'EPSG:4326', [左上角x], [左上角y])
        右下角经度, 右下角纬度 = warp_transform(src.crs, 'EPSG:4326', [右下角x], [右下角y])
        
        基准数据 = {
            'tif文件': os.path.basename(tif路径),
            'tif完整路径': tif路径,
            '左上角经度': 左上角经度[0],
            '左上角纬度': 左上角纬度[0],
            '右下角经度': 右下角经度[0],
            '右下角纬度': 右下角纬度[0],
            '耕地面积_平方米': float(总耕地面积_平方米),
            '耕地面积_亩': float(总耕地面积_亩),
            '总面积_平方米': float(src.width * src.height * 单像素面积),
            '耕地比例': float(总耕地面积_平方米 / (src.width * src.height * 单像素面积)),
            '图像宽度': src.width,
            '图像高度': src.height,
            'crs': str(src.crs)
        }
        
        # 随机采样小块进行训练（避免内存爆炸）
        采样数量 = 1500  # 快速训练：1500个样本（30分钟内）
        图像列表 = []
        标签列表 = []
        
        # 三种类型的目标数量
        纯耕地目标 = int(采样数量 * 0.35)  # 35% 纯耕地（>80%）
        混合目标 = int(采样数量 * 0.45)    # 45% 混合区域（20-80%）
        非耕地目标 = 采样数量 - 纯耕地目标 - 混合目标  # 20% 非耕地（<20%）
        
        纯耕地计数 = 0
        混合计数 = 0
        非耕地计数 = 0
        
        print(f"  平衡采样策略:")
        print(f"    纯耕地样本(>80%): {纯耕地目标}")
        print(f"    混合样本(20-80%): {混合目标}")
        print(f"    非耕地样本(<20%): {非耕地目标}")
        print(f"  开始采样...")
        
        尝试次数 = 0
        最大尝试 = 采样数量 * 100  # 增加最大尝试次数（样本多了，需要更多尝试）
        
        while (纯耕地计数 < 纯耕地目标 or 混合计数 < 混合目标 or 非耕地计数 < 非耕地目标) and 尝试次数 < 最大尝试:
            尝试次数 += 1
            
            # 随机选择位置
            x = np.random.randint(0, max(1, src.width - 目标尺寸))
            y = np.random.randint(0, max(1, src.height - 目标尺寸))
            
            # 计算该块的地理范围
            块_左上x = src.transform.c + x * src.transform.a
            块_左上y = src.transform.f + y * src.transform.e
            块_右下x = 块_左上x + 目标尺寸 * src.transform.a
            块_右下y = 块_左上y + 目标尺寸 * src.transform.e
            
            块边界 = box(
                min(块_左上x, 块_右下x),
                min(块_左上y, 块_右下y),
                max(块_左上x, 块_右下x),
                max(块_左上y, 块_右下y)
            )
            
            # 查找与该块相交的耕地几何
            块_gdf = gdf[gdf.intersects(块边界)]
            
            # 只保留包含一定耕地的块(避免全是背景)
            if len(块_gdf) > 0:
                # 为该小块生成掩码（现在只有256x256，小内存）
                from affine import Affine
                块_transform = Affine(
                    src.transform.a, src.transform.b, 块_左上x,
                    src.transform.d, src.transform.e, 块_左上y
                )
                
                标签块 = geometry_mask(
                    块_gdf.geometry,
                    out_shape=(目标尺寸, 目标尺寸),
                    transform=块_transform,
                    invert=False
                )
                标签块 = (~标签块).astype(np.float32)
                
                # 计算耕地比例
                耕地比例 = 标签块.mean()
                
                # 判断属于哪种类型并检查是否需要
                需要采样 = False
                类型 = ""
                if 耕地比例 > 0.8 and 纯耕地计数 < 纯耕地目标:
                    需要采样 = True
                    类型 = "纯耕地"
                elif 0.2 <= 耕地比例 <= 0.8 and 混合计数 < 混合目标:
                    需要采样 = True
                    类型 = "混合"
                elif 耕地比例 < 0.2 and 非耕地计数 < 非耕地目标:
                    需要采样 = True
                    类型 = "非耕地"
                
                if 需要采样:
                    # 使用窗口读取（节省内存）
                    window = rasterio.windows.Window(x, y, 目标尺寸, 目标尺寸)
                    影像块 = src.read(window=window)
                    
                    # 转换为HxWxC
                    if 影像块.shape[0] <= 4:
                        影像块 = np.transpose(影像块[:3], (1, 2, 0))
                    
                    # 归一化
                    if 影像块.max() > 1.0:
                        影像块 = 影像块.astype(np.float32) / 255.0
                    
                    # 检查块大小是否正确
                    if 影像块.shape[0] == 目标尺寸 and 影像块.shape[1] == 目标尺寸:
                        图像列表.append(影像块)
                        标签列表.append(标签块[..., np.newaxis])
                        
                        # 更新计数
                        if 类型 == "纯耕地":
                            纯耕地计数 += 1
                        elif 类型 == "混合":
                            混合计数 += 1
                        else:
                            非耕地计数 += 1
                        
                        总计 = 纯耕地计数 + 混合计数 + 非耕地计数
                        if 总计 % 30 == 0:
                            print(f"    已采样: {总计}/{采样数量} (纯:{纯耕地计数}, 混:{混合计数}, 非:{非耕地计数})")
        
        print(f"  最终采样结果: 纯耕地{纯耕地计数}, 混合{混合计数}, 非耕地{非耕地计数}")

        print(f"  生成 {len(图像列表)} 个训练样本")
        print(f"  基准耕地面积: {基准数据['耕地面积_亩']:.2f} 亩")
        
        return 图像列表, 标签列表, 基准数据


def 自动扫描文件(起始目录=".", 扩展名=".tif", 最大深度=3):
    """
    自动扫描目录及子目录，查找指定扩展名的文件
    """
    找到的文件 = []
    起始目录 = os.path.abspath(起始目录)
    
    def 扫描(目录, 当前深度=0):
        if 当前深度 > 最大深度:
            return
        
        try:
            for 项目 in os.listdir(目录):
                完整路径 = os.path.join(目录, 项目)
                
                if os.path.isfile(完整路径) and 项目.lower().endswith(扩展名.lower()):
                    找到的文件.append(完整路径)
                elif os.path.isdir(完整路径):
                    扫描(完整路径, 当前深度 + 1)
        except PermissionError:
            pass
    
    扫描(起始目录)
    return 找到的文件


def 递归扫描TIF文件(根目录):
    """
    递归扫描所有子文件夹，找到所有.tif文件
    
    参数:
        根目录: 主文件夹路径
    
    返回:
        TIF文件路径列表
    """
    tif文件列表 = []
    
    print(f"\n🔍 递归扫描文件夹: {根目录}")
    print("="*60)
    
    if not os.path.exists(根目录):
        print(f"❌ 文件夹不存在: {根目录}")
        return tif文件列表
    
    # os.walk递归遍历所有子文件夹
    for 根路径, 子文件夹, 文件列表 in os.walk(根目录):
        for 文件名 in 文件列表:
            if 文件名.lower().endswith('.tif'):
                tif完整路径 = os.path.join(根路径, 文件名)
                tif文件列表.append(tif完整路径)
                
                # 显示相对路径
                try:
                    相对路径 = os.path.relpath(tif完整路径, 根目录)
                except:
                    相对路径 = tif完整路径
                print(f"  ✅ 找到: {相对路径}")
    
    print("="*60)
    print(f"\n📦 总共找到 {len(tif文件列表)} 个TIF文件")
    
    return tif文件列表


def 逐个训练TIF模式():
    """
    递归扫描 + 逐个训练TIF文件（避免内存爆炸）
    每训练完一个TIF就保存模型，然后继续下一个（增量学习）
    """
    print("\n" + "="*60)
    print("🎓 递归逐个训练模式")
    print("="*60)
    print("✨ 功能：")
    print("  ✅ 递归扫描所有子文件夹找TIF")
    print("  ✅ 每次只训练1个TIF（避免内存爆炸）")
    print("  ✅ 自动增量学习（累积精度）")
    print()
    
    # 1. 递归扫描TIF文件
    tif列表 = 递归扫描TIF文件(训练图像目录)
    
    if not tif列表:
        print("\n❌ 未找到任何TIF文件！")
        print(f"   请检查路径: {训练图像目录}")
        return
    
    # 2. 检查标注文件
    if not os.path.exists(训练标注目录):
        print(f"\n❌ 标注文件不存在: {训练标注目录}")
        return
    
    print(f"\n📍 使用标注文件: {os.path.basename(训练标注目录)}")
    print("="*60)
    
    # 3. 逐个训练
    已训练数量 = 0
    总样本数 = 0
    model = None
    用户已选择模式 = None  # ✅ 记录用户的选择，后续自动应用
    
    for index, tif路径 in enumerate(tif列表, 1):
        print(f"\n{'='*60}")
        print(f"🎯 正在训练第 {index}/{len(tif列表)} 个TIF")
        print(f"{'='*60}")
        print(f"📂 文件: {os.path.basename(tif路径)}")
        
        try:
            # ✅ 先尝试读取TIF文件
            try:
                with rasterio.open(tif路径) as test_src:
                    原始宽度_像素 = test_src.width
                    原始高度_像素 = test_src.height
                    像素分辨率 = abs(test_src.transform.a)
                    
                    # 计算实际长宽（米）
                    原始宽度_米 = 原始宽度_像素 * 像素分辨率
                    原始高度_米 = 原始高度_像素 * 像素分辨率
                    
                    print(f"📏 TIF图像信息:")
                    print(f"   尺寸: {原始宽度_像素} x {原始高度_像素} 像素")
                    print(f"   分辨率: {像素分辨率:.4f} 米/像素")
                    print(f"   实际长宽: {原始宽度_米:.1f}m x {原始高度_米:.1f}m")
            except Exception as e:
                print(f"\n❌ 读取TIF文件失败: {e}")
                print(f"   自动跳过，继续下一个TIF...\n")
                continue  # ✅ 自动跳过，继续下一个
            
            # 生成当前TIF的训练数据
            图像块, 标签块, 基准数据 = 从TIF和Shapefile生成训练数据(
                tif路径, 
                训练标注目录, 
                图像尺寸
            )
            
            if len(图像块) == 0:
                print(f"  ⚠️  跳过：该TIF无有效样本")
                continue
            
            # 转换为numpy数组
            X = np.array(图像块)
            y = np.array(标签块)
            
            print(f"\n📦 当前TIF生成 {len(X)} 个训练样本")
            总样本数 += len(X)
            
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=验证比例, random_state=42
            )
            
            print(f"📋 数据划分: 训练集 {len(X_train)} + 验证集 {len(X_val)}")
            
            # 加载或构建模型
            if 已训练数量 == 0:
                # 第一个TIF：检查是否有旧模型
                if os.path.exists(模型保存路径):
                    print(f"\n📦 检测到已有模型: {os.path.basename(模型保存路径)}")
                    用户选择 = input("选择模式:\n  1. 增量学习（在旧模型基础上继续训练）\n  2. 重新训练（从零开始）\n请输入选择 (1/2): ").strip()
                    
                    # ✅ 记录用户选择，后续TIF自动应用
                    用户已选择模式 = 用户选择
                    
                    if 用户选择 == "1":
                        try:
                            from tensorflow import keras
                            model = keras.models.load_model(
                                模型保存路径,
                                custom_objects={'dice_coefficient': dice_coefficient}
                            )
                            print("✅ 已加载旧模型，将进行增量学习")
                            print("💡 后续TIF将自动使用增量学习模式")
                        except Exception as e:
                            print(f"⚠️  加载旧模型失败: {e}")
                            print("构建新模型...")
                            model = 构建UNet模型(输入尺寸=(图像尺寸, 图像尺寸, 3))
                    else:
                        # 备份旧模型
                        备份路径 = 模型保存路径.replace('.h5', f'_备份_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
                        import shutil
                        shutil.copy(模型保存路径, 备份路径)
                        print(f"💾 旧模型已备份: {os.path.basename(备份路径)}")
                        model = 构建UNet模型(输入尺寸=(图像尺寸, 图像尺寸, 3))
                else:
                    # 没有旧模型，构建新模型
                    print(f"\n🏗️ 构建新模型...")
                    model = 构建UNet模型(输入尺寸=(图像尺寸, 图像尺寸, 3))
                    用户已选择模式 = "2"  # 新模型默认为重新训练模式
            else:
                # 后续的TIF：根据用户第一次的选择自动应用
                if 用户已选择模式 == "1":
                    # 用户选择了增量学习，自动加载上一轮的模型
                    print(f"\n📦 自动增量学习：加载上一轮训练的模型...")
                    try:
                        from tensorflow import keras
                        model = keras.models.load_model(
                            模型保存路径,
                            custom_objects={'dice_coefficient': dice_coefficient}
                        )
                        print("✅ 模型加载成功，继续增量学习")
                    except Exception as e:
                        print(f"❌ 加载模型失败: {e}")
                        return
                else:
                    # 用户选择了重新训练，后续TIF也继续训练（不加载旧模型）
                    print(f"\n🔄 继续训练模式（基于第一个TIF构建的模型）")
                    # model已经在第一个TIF时构建，这里不需要重新加载
            
            # 训练当前TIF
            print(f"\n🚀 开始训练（第{index}个TIF，{训练轮数}轮）...")
            
            try:
                from tensorflow import keras
            except:
                import keras
            
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    模型保存路径,
                    save_best_only=True,
                    monitor='val_loss',
                    verbose=1
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    verbose=1
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=批次大小,
                epochs=训练轮数,
                callbacks=callbacks,
                verbose=1
            )
            
            已训练数量 += 1
            
            print(f"\n✅ 第{index}个TIF训练完成！")
            print(f"📊 已累积训练 {已训练数量} 个TIF，总样本数: {总样本数}")
            
            # ✅ 计算并显示耕地长宽变化（精确到0.5m）
            try:
                with rasterio.open(tif路径) as src:
                    # 计算耕地区域的实际范围
                    耕地掩码 = (y_train.mean(axis=0).squeeze() > 0.5).astype(np.uint8) * 255
                    轮廓, _ = cv2.findContours(耕地掩码, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(轮廓) > 0:
                        # 找到最大轮廓
                        最大轮廓 = max(轮廓, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(最大轮廓)
                        
                        # 计算实际长宽（米）
                        耕地宽度_米 = w * (原始宽度_米 / 原始宽度_像素)
                        耕地高度_米 = h * (原始高度_米 / 原始高度_像素)
                        
                        print(f"\n📏 耕地区域尺寸（精确到0.5m）:")
                        print(f"   宽度: {耕地宽度_米:.1f} m")
                        print(f"   高度: {耕地高度_米:.1f} m")
                        print(f"   面积: {(耕地宽度_米 * 耕地高度_米) / 666.67:.3f} 亩")
            except Exception as e:
                print(f"   ⚠️  无法计算耕地长宽: {e}")
            
            # 释放内存
            import gc
            del X, y, X_train, X_val, y_train, y_val
            gc.collect()
            
        except Exception as e:
            print(f"\n❌ 训练第{index}个TIF时出错: {e}")
            import traceback
            traceback.print_exc()
            
            用户选择 = input("\n是否继续下一个TIF? (y/n): ").strip().lower()
            if 用户选择 != 'y':
                print("训练已中断")
                return
    
    print(f"\n{'='*60}")
    print(f"🎉 所有TIF训练完成！")
    print(f"📊 总计: {已训练数量} 个TIF, {总样本数} 个训练样本")
    print(f"💾 模型已保存: {模型保存路径}")
    print(f"{'='*60}")
    
    # ✅ 增量学习模式：在原有PKL基础上扩展基准地图
    基准数据文件 = 模型保存路径.replace('.h5', '_基准数据.pkl')
    print(f"\n📍 更新基准耕地地图（增量学习模式）...")
    
    try:
        # 检查是否存在原有PKL
        原有数据 = None
        原有训练列表 = []
        if os.path.exists(基准数据文件):
            print(f"   ✅ 发现原有基准数据: {基准数据文件}")
            with open(基准数据文件, 'rb') as f:
                原有数据 = pickle.load(f)
            原有训练列表 = 原有数据.get('训练图像列表', [])
            print(f"   原有训练图像数: {len(原有训练列表)}")
            print(f"   原有覆盖范围: X[{原有数据['覆盖范围']['左']:.1f}~{原有数据['覆盖范围']['右']:.1f}]")
        else:
            print(f"   📝 未发现原有基准数据，将创建新的")
        
        # 计算所有已训练TIF的联合范围（包括新旧）
        全局_左 = float('inf')
        全局_右 = float('-inf')
        全局_上 = float('-inf')
        全局_下 = float('inf')
        基准_crs = None
        基准_分辨率 = None
        
        # 如果有原有数据，先用原有范围初始化
        if 原有数据:
            全局_左 = 原有数据['覆盖范围']['左']
            全局_右 = 原有数据['覆盖范围']['右']
            全局_上 = 原有数据['覆盖范围']['上']
            全局_下 = 原有数据['覆盖范围']['下']
            基准_分辨率 = 原有数据['像素分辨率_米']
            基准_crs = 原有数据.get('crs')
        
        # 扩展范围以包含新TIF
        for tif路径 in tif列表:
            try:
                with rasterio.open(tif路径) as src:
                    if 基准_crs is None:
                        基准_crs = src.crs
                    if 基准_分辨率 is None:
                        基准_分辨率 = abs(src.transform.a) * 4  # 降采样因子
                    
                    全局_左 = min(全局_左, src.bounds.left)
                    全局_右 = max(全局_右, src.bounds.right)
                    全局_上 = max(全局_上, src.bounds.top)
                    全局_下 = min(全局_下, src.bounds.bottom)
            except:
                pass
        
        print(f"   扩展后范围: X[{全局_左:.1f}~{全局_右:.1f}] Y[{全局_下:.1f}~{全局_上:.1f}]")
        
        # 读取SHP并生成新的基准地图
        gdf = gpd.read_file(训练标注目录)
        # 将基准CRS转换为CRS对象进行比较和转换
        if 基准_crs:
            from rasterio.crs import CRS
            if isinstance(基准_crs, str):
                基准_crs_obj = CRS.from_string(基准_crs)
            else:
                基准_crs_obj = 基准_crs
            if hasattr(gdf, 'crs') and gdf.crs and str(gdf.crs) != str(基准_crs_obj):
                gdf = gdf.to_crs(基准_crs_obj)
        
        全局宽度_米 = 全局_右 - 全局_左
        全局高度_米 = 全局_上 - 全局_下
        
        新宽度 = int(全局宽度_米 / 基准_分辨率)
        新高度 = int(全局高度_米 / 基准_分辨率)
        
        print(f"   新基准地图尺寸: {新宽度}x{新高度} 像素")
        
        from affine import Affine
        新transform = Affine(基准_分辨率, 0, 全局_左, 0, -基准_分辨率, 全局_上)
        
        # 生成新的完整基准地图
        新基准耕地地图 = geometry_mask(
            gdf.geometry,
            out_shape=(新高度, 新宽度),
            transform=新transform,
            invert=True
        ).astype(np.uint8)
        
        # 如果有原有数据，合并（保留原有数据中已有的部分）
        if 原有数据 and '基准耕地地图' in 原有数据:
            原有地图 = 原有数据['基准耕地地图']
            原有transform = 原有数据['地理变换']
            
            # 计算原有地图在新地图中的位置
            原有_左 = 原有数据['覆盖范围']['左']
            原有_上 = 原有数据['覆盖范围']['上']
            
            # 计算偏移（像素）
            偏移_col = int((原有_左 - 全局_左) / 基准_分辨率)
            偏移_row = int((全局_上 - 原有_上) / 基准_分辨率)
            
            # 将原有地图数据复制到新地图中（使用OR操作合并）
            原高度, 原宽度 = 原有地图.shape
            结束_row = min(偏移_row + 原高度, 新高度)
            结束_col = min(偏移_col + 原宽度, 新宽度)
            有效_原高度 = 结束_row - 偏移_row
            有效_原宽度 = 结束_col - 偏移_col
            
            if 偏移_row >= 0 and 偏移_col >= 0 and 有效_原高度 > 0 and 有效_原宽度 > 0:
                新基准耕地地图[偏移_row:结束_row, 偏移_col:结束_col] = np.maximum(
                    新基准耕地地图[偏移_row:结束_row, 偏移_col:结束_col],
                    原有地图[:有效_原高度, :有效_原宽度]
                )
                print(f"   ✅ 已合并原有基准数据")
        
        print(f"   基准地图大小: {新基准耕地地图.nbytes / (1024*1024):.1f} MB")
        print(f"   耕地像素数: {np.sum(新基准耕地地图)} ({np.mean(新基准耕地地图)*100:.2f}%)")
        
        # 合并训练图像列表（去重）
        新训练列表 = [os.path.basename(f) for f in tif列表]
        合并后列表 = list(set(原有训练列表 + 新训练列表))
        print(f"   累计训练图像数: {len(合并后列表)}")
        
        # 保存更新后的基准数据
        with open(基准数据文件, 'wb') as f:
            pickle.dump({
                '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '基准年份': '训练数据年份',
                '基准shp文件': os.path.basename(训练标注目录),
                '基准耕地地图': 新基准耕地地图,
                '地理变换': {
                    'a': 新transform.a,
                    'b': 新transform.b,
                    'c': 新transform.c,
                    'd': 新transform.d,
                    'e': 新transform.e,
                    'f': 新transform.f
                },
                'crs': str(基准_crs) if 基准_crs else None,
                '像素分辨率_米': abs(新transform.a),
                '覆盖范围': {
                    '左': 全局_左,
                    '右': 全局_右,
                    '上': 全局_上,
                    '下': 全局_下
                },
                '训练图像列表': 合并后列表,
                '增量更新历史': (原有数据.get('增量更新历史', []) if 原有数据 else []) + [
                    {
                        '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        '新增图像': 新训练列表,
                        '新增数量': len(新训练列表)
                    }
                ]
            }, f)
        
        print(f"📋 基准数据已保存: {基准数据文件}")
        
    except Exception as e:
        print(f"⚠️  生成基准地图失败: {e}")
        import traceback
        traceback.print_exc()


def 准备训练数据(图像目录, 标注目录):
    """
    扫描目录,准备所有训练数据
    支持两种模式:
    1. 多个TIF + 多个对应SHP (一一对应)
    2. 多个TIF + 一个大SHP (自动裁剪)
    如果目录为空，自动扫描当前目录及子目录
    """
    print("\n" + "=" * 60)
    print("📦 准备训练数据")
    print("=" * 60)
    
    所有图像 = []
    所有标签 = []
    基准数据列表 = []  # 保存所有训练图像的基准耕地数据
    
    # 如果目录为空，自动扫描
    if not 图像目录 or not os.path.exists(图像目录):
        print("\n🔍 自动扫描模式: 扫描当前目录及子目录...")
        tif文件列表 = 自动扫描文件(".", ".tif")
        shp文件列表 = 自动扫描文件(".", ".shp")
        
        if not tif文件列表:
            print("❌ 未找到TIF文件!")
            print("   请将TIF图像放在当前目录或子目录中")
            print("   或在配置中指定图像目录")
            return None, None, None
        
        if not shp文件列表:
            print("❌ 未找到SHP文件!")
            print("   请将Shapefile标注放在当前目录或子目录中")
            print("   或在配置中指定标注目录")
            return None, None, None
        
        print(f"\n✅ 找到 {len(tif文件列表)} 个TIF图像:")
        for tif in tif文件列表[:5]:  # 只显示前5个
            print(f"   - {os.path.basename(tif)}")
        if len(tif文件列表) > 5:
            print(f"   ... 还有 {len(tif文件列表)-5} 个")
        
        print(f"\n✅ 找到 {len(shp文件列表)} 个SHP文件:")
        for shp in shp文件列表:
            print(f"   - {os.path.basename(shp)}")
    else:
        # 手动指定目录
        print(f"\n📂 扫描指定目录...")
        tif文件列表 = []
        for file in os.listdir(图像目录):
            if file.lower().endswith('.tif'):
                tif文件列表.append(os.path.join(图像目录, file))
        
        if not tif文件列表:
            print("❌ 未找到TIF文件!")
            print(f"   请确保 {图像目录} 中有.tif文件")
            return None, None, None
        
        print(f"\n找到 {len(tif文件列表)} 个TIF图像")
        
        # 查找SHP文件
        if os.path.isfile(标注目录) and 标注目录.lower().endswith('.shp'):
            # 直接指定了SHP文件
            shp文件列表 = [标注目录]
        elif os.path.isdir(标注目录):
            # 指定了目录
            shp文件列表 = [os.path.join(标注目录, f) for f in os.listdir(标注目录) if f.lower().endswith('.shp')]
        else:
            shp文件列表 = []
        
        if not shp文件列表:
            print("❌ 未找到SHP文件!")
            print(f"   请确保 {标注目录} 中有.shp文件")
            return None, None, None
        
        print(f"找到 {len(shp文件列表)} 个SHP文件")
    
    # 判断使用哪种模式
    if len(shp文件列表) == 1:
        # 模式2: 一个大SHP,所有TIF共用
        shp路径 = shp文件列表[0]
        print(f"\n✅ 使用单一SHP模式: {os.path.basename(shp路径)}")
        print("   将为每个TIF自动裁剪对应区域的标注")
        
        for tif路径 in tif文件列表:
            print(f"\n处理: {os.path.basename(tif路径)} + {os.path.basename(shp路径)}")
            图像块, 标签块, 基准数据 = 从TIF和Shapefile生成训练数据(tif路径, shp路径, 图像尺寸)
            所有图像.extend(图像块)
            所有标签.extend(标签块)
            基准数据列表.append(基准数据)
    else:
        # 模式1: 一一对应
        print("\n✅ 使用一一对应模式")
        训练对列表 = []
        
        for tif路径 in tif文件列表:
            tif文件名 = os.path.basename(tif路径)
            # 查找对应的shapefile
            shp文件名 = tif文件名.replace('.tif', '.shp').replace('.TIF', '.shp')
            
            # 在SHP列表中查找
            匹配的shp = None
            for shp in shp文件列表:
                if os.path.basename(shp) == shp文件名:
                    匹配的shp = shp
                    break
            
            if 匹配的shp:
                训练对列表.append((tif路径, 匹配的shp))
                print(f"✅ 找到训练对: {tif文件名} + {shp文件名}")
            else:
                print(f"⚠️  跳过 {tif文件名}: 未找到对应的 {shp文件名}")
        
        if not 训练对列表:
            print("\n❌ 未找到任何TIF-SHP训练对!")
            print("   提示: 如果你只有一个大SHP文件,请只保留一个SHP在标注目录中")
            return None, None, None
        
        # 处理每一对
        for tif路径, shp路径 in 训练对列表:
            图像块, 标签块, 基准数据 = 从TIF和Shapefile生成训练数据(tif路径, shp路径, 图像尺寸)
            所有图像.extend(图像块)
            所有标签.extend(标签块)
            基准数据列表.append(基准数据)
    
    print(f"\n✅ 总共生成 {len(所有图像)} 个训练样本")
    print(f"📋 记录了 {len(基准数据列表)} 张图像的基准耕地数据")
    
    return np.array(所有图像), np.array(所有标签), 基准数据列表, shp文件列表, tif文件列表


def 训练模型():
    """
    完整的模型训练流程（支持增量学习 + 递归逐个训练）
    """
    
    # 根据训练模式选择
    if 训练模式 == "递归逐个":
        逐个训练TIF模式()
        return None, None
    
    # 以下是普通模式（一次性训练所有）
    print("\n" + "=" * 60)
    print("🎓 U-Net耕地识别模型训练 - 普通模式")
    print("=" * 60)
    
    # 检查是否已有模型（增量学习）
    增量学习 = False
    if os.path.exists(模型保存路径):
        print("\n🔍 检测到已有模型！")
        print(f"模型文件: {模型保存路径}")
        
        用户选择 = input("\n选择模式:\n  1. 增量学习（在旧模型基础上继续训练）\n  2. 重新训练（从零开始）\n请输入选择 (1/2): ").strip()
        
        if 用户选择 == "1":
            增量学习 = True
            print("\n✅ 已选择：增量学习模式")
            print("📦 正在加载旧模型...")
        else:
            print("\n✅ 已选择：重新训练模式")
            # 备份旧模型
            备份路径 = 模型保存路径.replace('.h5', f'_备份_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
            import shutil
            shutil.copy(模型保存路径, 备份路径)
            print(f"💾 旧模型已备份: {os.path.basename(备份路径)}")
    
    # 准备数据
    X, y, 基准数据列表, shp文件列表, tif文件列表 = 准备训练数据(训练图像目录, 训练标注目录)
    
    if X is None:
        return
    
    print(f"\n数据形状:")
    print(f"  图像: {X.shape}")
    print(f"  标签: {y.shape}")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=验证比例, random_state=42
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    
    # 构建或加载模型
    if 增量学习:
        print("\n📦 加载已有模型...")
        try:
            from tensorflow import keras
        except:
            import keras
        
        try:
            # 尝试加载模型
            model = keras.models.load_model(
                模型保存路径,
                custom_objects={'dice_coefficient': dice_coefficient},
                compile=False
            )
            # 重新编译（使用当前学习率）
            try:
                optimizer = keras.optimizers.Adam(learning_rate=学习率)
            except:
                optimizer = 'adam'
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', dice_coefficient]
            )
            print("✅ 模型加载成功！")
            print(f"模型参数: {model.count_params():,}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("🔄 回退到重新训练模式...")
            增量学习 = False
            model = 构建UNet模型(输入尺寸=(图像尺寸, 图像尺寸, 3))
    else:
        print("\n🏭 构建U-Net模型...")
        model = 构建UNet模型(输入尺寸=(图像尺寸, 图像尺寸, 3))
        print(f"模型参数: {model.count_params():,}")
    
    # 训练模型
    print(f"\n🚀 开始训练 ({训练轮数} 轮)...")
    
    try:
        from tensorflow import keras
    except:
        import keras
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            模型保存路径,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=批次大小,
        epochs=训练轮数,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ 训练完成!")
    print(f"📦 模型已保存: {模型保存路径}")
    
    # 保存训练历史
    历史文件 = 模型保存路径.replace('.h5', '_history.pkl')
    with open(历史文件, 'wb') as f:
        pickle.dump(history.history, f)
    
    # 保存基准耕地数据（包含完整的基准耕地地图）
    基准数据文件 = 模型保存路径.replace('.h5', '_基准数据.pkl')
    
    # 读取第一个SHP作为基准（假设所有训练图都用同一个SHP）
    if len(shp文件列表) > 0:
        基准shp路径 = shp文件列表[0]
        print(f"\n📍 生成基准耕地地图: {os.path.basename(基准shp路径)}")
        
        # ✅ 关键修复：计算所有TIF的联合覆盖范围
        print(f"\n📊 计算所有TIF的联合范围...")
        全局_左 = float('inf')
        全局_右 = float('-inf')
        全局_上 = float('-inf')
        全局_下 = float('inf')
        基准_crs = None
        基准_分辨率 = None
        
        for tif路径 in tif文件列表:
            try:
                with rasterio.open(tif路径) as src:
                    if 基准_crs is None:
                        基准_crs = src.crs
                        基准_分辨率 = abs(src.transform.a)
                    
                    # 更新全局范围
                    全局_左 = min(全局_左, src.bounds.left)
                    全局_右 = max(全局_右, src.bounds.right)
                    全局_上 = max(全局_上, src.bounds.top)
                    全局_下 = min(全局_下, src.bounds.bottom)
                    print(f"  ✅ {os.path.basename(tif路径)}: X[{src.bounds.left:.1f}~{src.bounds.right:.1f}] Y[{src.bounds.bottom:.1f}~{src.bounds.top:.1f}]")
            except Exception as e:
                print(f"  ⚠️ 跳过 {os.path.basename(tif路径)}: {e}")
        
        print(f"\n🌍 全局覆盖范围:")
        print(f"   X: {全局_左:.2f} ~ {全局_右:.2f}")
        print(f"   Y: {全局_下:.2f} ~ {全局_上:.2f}")
        
        # 读取Shapefile
        gdf = gpd.read_file(基准shp路径)
        if gdf.crs != 基准_crs and 基准_crs is not None:
            gdf = gdf.to_crs(基准_crs)
        
        # 生成基准耕地掩码地图（降采样以节省空间）
        降采样因子 = 4  # 每4个像素合并为1个
        
        # ✅ 使用全局范围计算尺寸
        全局宽度_米 = 全局_右 - 全局_左
        全局高度_米 = 全局_上 - 全局_下
        
        新分辨率 = 基准_分辨率 * 降采样因子
        新宽度 = int(全局宽度_米 / 新分辨率)
        新高度 = int(全局高度_米 / 新分辨率)
        
        print(f"\n📰 基准地图尺寸:")
        print(f"   全局范围: {全局宽度_米:.1f}m x {全局高度_米:.1f}m")
        print(f"   降采样后: {新宽度}x{新高度} 像素")
        
        from affine import Affine
        # ✅ 使用全局左上角作为起点
        新transform = Affine(新分辨率, 0, 全局_左, 0, -新分辨率, 全局_上)
        
        基准耕地地图 = geometry_mask(
            gdf.geometry,
            out_shape=(新高度, 新宽度),
            transform=新transform,
            invert=True  # True=耕地为1
        ).astype(np.uint8)
        
        print(f"   基准地图大小: {基准耕地地图.nbytes / (1024*1024):.1f} MB")
        print(f"   耕地像素数: {np.sum(基准耕地地图)} ({np.mean(基准耕地地图)*100:.2f}%)")
        
        # 保存完整的基准信息
        with open(基准数据文件, 'wb') as f:
            pickle.dump({
                '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '基准年份': '训练数据年份',
                '基准shp文件': os.path.basename(基准shp路径),
                '基准耕地地图': 基准耕地地图,
                '地理变换': {
                    'a': 新transform.a,
                    'b': 新transform.b,
                    'c': 新transform.c,
                    'd': 新transform.d,
                    'e': 新transform.e,
                    'f': 新transform.f
                },
                'crs': str(基准_crs),
                '像素分辨率_米': abs(新transform.a),
                '覆盖范围': {
                    '左': 全局_左,
                    '右': 全局_右,
                    '上': 全局_上,
                    '下': 全局_下
                },
                '训练图像列表': [os.path.basename(f) for f in tif文件列表]
            }, f)
    else:
        print("\n⚠️  未找到Shapefile，跳过基准地图生成")
        with open(基准数据文件, 'wb') as f:
            pickle.dump({
                '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '说明': '仅用于模型识别，无基准对比数据'
            }, f)
    
    print(f"📋 基准数据已保存: {基准数据文件}")
    
    # 显示最终结果
    print(f"\n📊 最终结果:")
    print(f"  训练准确率: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"  验证准确率: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"  训练Dice系数: {history.history['dice_coefficient'][-1]:.4f}")
    print(f"  验证Dice系数: {history.history['val_dice_coefficient'][-1]:.4f}")
    
    print("\n📋 保存的基准数据包括:")
    总基准面积 = sum(d['耕地面积_亩'] for d in 基准数据列表)
    print(f"  训练图像数: {len(基准数据列表)}")
    print(f"  总基准耕地面积: {总基准面积:.2f} 亩")
    for i, 数据 in enumerate(基准数据列表, 1):
        print(f"  {i}. {数据['tif文件']}: {数据['耕地面积_亩']:.2f} 亩")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = 训练模型()
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
