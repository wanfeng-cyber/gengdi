"""
自动坐标系修复示例
展示如何在耕地分析前自动处理坐标系问题
"""

import os
import numpy as np
import rasterio
from 坐标系处理模块 import 坐标系处理器, 预处理图像坐标系
from 耕地分析系统 import 耕地分析系统
from tqdm import tqdm
import tempfile

class 智能耕地分析器(耕地分析系统):
    """
    带有自动坐标系处理的耕地分析器
    """

    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp(prefix="耕地分析_")
        print(f"✅ 创建临时目录: {self.temp_dir}")

    def 智能比较两年数据(self, 去年图像路径, 今年图像路径, 模型路径=None):
        """
        智能比较两年数据，自动处理坐标系问题

        参数:
            去年图像路径: 去年图像路径
            今年图像路径: 今年图像路径
            模型路径: 模型路径

        返回:
            dict: 分析结果
        """
        print("=" * 60)
        print("🌍 智能耕地分析（自动坐标系处理）")
        print("=" * 60)

        # 1. 检查坐标系一致性
        print("\n1️⃣ 检查坐标系一致性...")
        检查结果 = 坐标系处理器.检查坐标系一致性(去年图像路径, 今年图像路径)

        print(f"   去年图像: {os.path.basename(去年图像路径)}")
        print(f"   坐标系: {检查结果['文件1']['crs']}")
        print(f"   单位: {检查结果['文件1']['单位']}")

        print(f"\n   今年图像: {os.path.basename(今年图像路径)}")
        print(f"   坐标系: {检查结果['文件2']['crs']}")
        print(f"   单位: {检查结果['文件2']['单位']}")

        # 2. 处理坐标系差异
        实际今年路径 = 今年图像路径
        if not 检查结果['一致']:
            print(f"\n❌ 检测到坐标系不匹配！")
            print(f"   问题: {检查结果['问题描述']}")

            print(f"\n2️⃣ 自动转换坐标系...")
            # 将今年图像转换到去年的坐标系
            转换后路径 = 坐标系处理器.自动转换坐标系(
                今年图像路径,
                检查结果['文件1']['crs'],
                self.temp_dir
            )

            if 转换后路径 and 转换后路径 != 今年图像路径:
                实际今年路径 = 转换后路径
                print(f"   ✅ 已转换坐标系: {os.path.basename(实际今年路径)}")
            else:
                print(f"   ⚠️ 坐标系转换失败，将使用原图像")
        else:
            print(f"\n✅ 坐标系一致，无需转换")

        # 3. 分析去年数据
        print(f"\n3️⃣ 分析去年耕地...")
        去年结果 = self.使用模型预测耕地_大图(
            去年图像路径,
            模型路径=模型路径,
            快速模式=True
        )

        # 4. 分析今年数据
        print(f"\n4️⃣ 分析今年耕地...")
        今年结果 = self.使用模型预测耕地_大图(
            实际今年路径,
            模型路径=模型路径,
            快速模式=True
        )

        # 5. 计算变化
        print(f"\n5️⃣ 计算耕地变化...")
        面积变化 = 今年结果['耕地面积_亩'] - 去年结果['耕地面积_亩']
        变化百分比 = abs(面积变化 / 去年结果['耕地面积_亩'] * 100) if 去年结果['耕地面积_亩'] > 0 else 0

        # 6. 生成报告
        print(f"\n" + "=" * 60)
        print(f"📊 分析结果报告")
        print(f"=" * 60)

        print(f"\n📌 数据来源:")
        print(f"   去年: {os.path.basename(去年图像路径)}")
        print(f"   今年: {os.path.basename(今年图像路径)}")
        if 实际今年路径 != 今年图像路径:
            print(f"   (已转换坐标系)")

        print(f"\n📏 耕地面积:")
        print(f"   去年: {去年结果['耕地面积_亩']:.3f} 亩")
        print(f"   今年: {今年结果['耕地面积_亩']:.3f} 亩")
        print(f"   变化: {面积变化:+.3f} 亩 ({变化百分比:+.1f}%)")

        # 分析变化原因
        if 变化百分比 > 200:
            print(f"\n⚠️ 注意：面积变化过大 ({变化百分比:.1f}%)")
            print(f"   可能原因：")
            print(f"   1. 坐标系转换问题")
            print(f"   2. 图像覆盖范围不同")
            print(f"   3. 实际耕地变化")
        elif abs(面积变化) > 0.1:
            if 面积变化 > 0:
                print(f"\n📈 耕地面积增加了 {面积变化:.3f} 亩")
                print(f"   可能是开垦了新耕地或边界扩展")
            else:
                print(f"\n📉 耕地面积减少了 {abs(面积变化):.3f} 亩")
                print(f"   可能是建设占用或退耕还林")
        else:
            print(f"\n📊 耕地面积基本稳定")

        # 返回结果
        return {
            '去年结果': 去年结果,
            '今年结果': 今年结果,
            '面积变化': 面积变化,
            '变化百分比': 变化百分比,
            '坐标系转换': 实际今年路径 != 今年图像路径,
            '转换后路径': 实际今年路径 if 实际今年路径 != 今年图像路径 else None
        }

    def 清理临时文件(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"✅ 已清理临时文件: {self.temp_dir}")


def 使用示例():
    """使用示例"""
    # 创建分析器
    分析器 = 智能耕地分析器()

    # 配置路径
    去年图像 = "path/to/去年图像.tif"
    今年图像 = "path/to/今年图像.tif"
    模型路径 = "path/to/模型.h5"

    try:
        # 执行分析
        结果 = 分析器.智能比较两年数据(去年图像, 今年图像, 模型路径)

        # 保存结果
        print(f"\n💾 分析完成！")

    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")

    finally:
        # 清理临时文件
        分析器.清理临时文件()


# 批量处理示例
def 批量处理目录(去年目录, 今年目录, 模型路径):
    """
    批量处理两个目录中的图像

    参数:
        去年目录: 去年图像目录
        今年目录: 今年图像目录
        模型路径: 模型路径
    """
    分析器 = 智能耕地分析器()

    # 获取文件列表
    去年文件 = [f for f in os.listdir(去年目录) if f.endswith(('.tif', '.tiff'))]
    今年文件 = [f for f in os.listdir(今年目录) if f.endswith(('.tif', '.tiff'))]

    print(f"发现去年图像: {len(去年文件)} 个")
    print(f"发现今年图像: {len(今年文件)} 个")

    # 批量处理
    results = []
    for 去年_path in tqdm(去年文件, desc="处理中"):
        去年_full = os.path.join(去年目录, 去年_path)

        # 查找对应的今年图像（假设文件名相似）
        今年_path = None
        for 今年 in 今年文件:
            if 今年.split('.')[0] in 去年_path or 去年_path.split('.')[0] in 今年:
                今年_path = os.path.join(今年目录, 今年)
                break

        if 今年_path:
            try:
                result = 分析器.智能比较两年数据(去年_full, 今年_path, 模型路径)
                results.append({
                    '去年文件': 去年_path,
                    '今年文件': os.path.basename(今年_path),
                    '面积变化': result['面积变化'],
                    '变化百分比': result['变化百分比']
                })
            except Exception as e:
                print(f"处理失败 {去年_path}: {str(e)}")

    # 保存批量结果
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("批量分析结果.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ 批量结果已保存到: 批量分析结果.csv")

    # 清理
    分析器.清理临时文件()


if __name__ == "__main__":
    print("智能耕地分析器 - 自动坐标系处理")
    print("=" * 60)

    # 运行示例
    使用示例()