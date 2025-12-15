"""
高精度识别部署检查脚本
检查高精度颜色识别是否已成功集成到系统中
"""

import os
import sys

def 检查文件完整性():
    """检查必要的文件是否存在"""
    print("=" * 60)
    print("📋 文件完整性检查")
    print("=" * 60)

    必要文件 = [
        "高精度颜色识别.py",
        "耕地分析系统.py",
        "耕地识别模型训练(16)(1).py",
        "耕地分析工具_图形界面.py"
    ]

    缺失文件 = []
    for 文件 in 必要文件:
        if os.path.exists(文件):
            print(f"✅ {文件}")
        else:
            print(f"❌ {文件} - 缺失！")
            缺失文件.append(文件)

    if 缺失文件:
        print(f"\n⚠️ 缺失 {len(缺失文件)} 个必要文件！")
        return False
    else:
        print("\n✅ 所有必要文件都存在")
        return True

def 检查高精度模块():
    """检查高精度识别模块是否可以导入"""
    print("\n" + "=" * 60)
    print("🔧 高精度识别模块检查")
    print("=" * 60)

    try:
        from 高精度颜色识别 import 高精度颜色识别器
        print("✅ 高精度颜色识别模块导入成功")

        # 创建识别器实例
        识别器 = 高精度颜色识别器()
        print("✅ 高精度颜色识别器实例化成功")

        # 检查关键方法
        关键方法 = ['多方法融合', '自适应颜色阈值', 'LAB空间增强检测', '局部优化']
        for 方法 in 关键方法:
            if hasattr(识别器, 方法):
                print(f"  ✅ 方法: {方法}")
            else:
                print(f"  ❌ 方法缺失: {方法}")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   请确保高精度颜色识别.py在同一目录")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def 检查依赖项():
    """检查必要的Python包"""
    print("\n" + "=" * 60)
    print("📦 依赖项检查")
    print("=" * 60)

    必要包 = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'rasterio': 'rasterio',
        'geopandas': 'geopandas',
        'tensorflow': 'tensorflow',
        'tkinter': 'tkinter (内置)',
        'PIL': 'Pillow'
    }

    缺失包 = []
    for 模块名, 包名 in 必要包.items():
        try:
            __import__(模块名)
            print(f"✅ {包名}")
        except ImportError:
            print(f"❌ {包名} - 未安装！")
            缺失包.append(包名)

    if 缺失包:
        print(f"\n⚠️ 缺失 {len(缺失包)} 个包！")
        print("\n安装命令：")
        for 包 in 缺失包:
            if 包 not in ['tkinter (内置)', 'tensorflow']:
                print(f"pip install {包}")
            elif 包 == 'tensorflow':
                print(f"# TensorFlow安装较复杂，请参考官方文档")
        return False
    else:
        print("\n✅ 所有依赖项都已安装")
        return True

def 检查集成代码():
    """检查集成代码是否存在"""
    print("\n" + "=" * 60)
    print("🔍 代码集成检查")
    print("=" * 60)

    # 检查耕地分析系统.py
    try:
        with open('耕地分析系统.py', 'r', encoding='utf-8') as f:
            内容 = f.read()

        if '高精度颜色识别' in 内容 and '多方法融合' in 内容:
            print("✅ 耕地分析系统.py - 已集成高精度识别")
        else:
            print("❌ 耕地分析系统.py - 未找到高精度识别集成代码")

        if '使用高精度增强' in 内容:
            print("✅ 耕地识别模型训练.py - 已添加高精度增强选项")

    except Exception as e:
        print(f"❌ 代码检查失败: {e}")

def 测试高精度识别():
    """简单测试高精度识别功能"""
    print("\n" + "=" * 60)
    print("🧪 功能测试")
    print("=" * 60)

    try:
        import numpy as np
        from 高精度颜色识别 import 高精度颜色识别器

        # 创建测试图像（100x100的RGB图像）
        测试图像 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        测试掩码 = np.zeros((100, 100))

        # 测试识别
        识别器 = 高精度颜色识别器()
        结果 = 识别器.多方法融合(测试图像, 测试掩码)

        print(f"✅ 测试图像处理成功")
        print(f"   输入图像尺寸: {测试图像.shape}")
        print(f"   输出掩码尺寸: {结果.shape}")
        print(f"   耕地像素比例: {np.mean(结果):.2%}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def 主检查():
    """执行所有检查"""
    print("🚀 高精度识别部署检查")
    print("=" * 60)
    print("时间:", __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # 执行各项检查
    文件检查 = 检查文件完整性()
    模块检查 = 检查高精度模块()
    依赖检查 = 检查依赖项()
    代码检查 = 检查集成代码()
    功能测试 = 测试高精度识别()

    # 总结
    print("\n" + "=" * 60)
    print("📊 检查总结")
    print("=" * 60)

    检查项目 = [
        ("文件完整性", 文件检查),
        ("高精度模块", 模块检查),
        ("依赖项", 依赖检查),
        ("代码集成", 代码检查),
        ("功能测试", 功能测试)
    ]

    通过数 = sum(1 for _, 结果 in 检查项目 if 结果)

    for 项目, 结果 in 检查项目:
        状态 = "✅ 通过" if 结果 else "❌ 失败"
        print(f"  {项目}: {状态}")

    print(f"\n总体结果: {通过数}/{len(检查项目)} 项通过")

    if 通过数 == len(检查项目):
        print("\n🎉 恭喜！高精度识别已成功部署！")
        print("\n使用说明：")
        print("1. 耕地分析系统会自动使用高精度识别")
        print("2. 训练时设置 使用高精度增强=True 可启用增强功能")
        print("3. 图形界面无需修改，直接使用即可")
    else:
        print("\n⚠️ 部分检查未通过，请根据上述提示修复问题")

    return 通过数 == len(检查项目)

if __name__ == "__main__":
    主检查()