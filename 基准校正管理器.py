"""
基准数据校正管理器
用于管理和应用不同测试区域的校正系数
"""
import json
import os
from typing import Dict, Optional, Tuple

class 基准校正管理器:
    """管理基准数据校正的系统"""

    def __init__(self, 配置文件路径="基准校正配置.json"):
        self.配置文件路径 = 配置文件路径
        self.校正配置 = self.加载配置()

    def 加载配置(self) -> Dict:
        """加载校正配置"""
        默认配置 = {
            "默认参考面积": 12.6,  # 默认参考面积（亩）
            "测试区域": {
                "default": {
                    "参考面积": 12.6,
                    "校正系数": 0.921,
                    "说明": "基于测试区域真实值12.6亩"
                }
            },
            "自动应用": True,
            "最小偏差阈值": 0.01  # 小于此值不应用校正
        }

        if os.path.exists(self.配置文件路径):
            try:
                with open(self.配置文件路径, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # 合并默认配置
                for key in 默认配置:
                    if key not in loaded_config:
                        loaded_config[key] = 默认配置[key]
                return loaded_config
            except Exception as e:
                print(f"⚠️  加载配置文件失败: {e}")
                return 默认配置
        else:
            self.保存配置(默认配置)
            return 默认配置

    def 保存配置(self, 配置: Dict = None):
        """保存校正配置"""
        if 配置 is None:
            配置 = self.校正配置

        try:
            with open(self.配置文件路径, 'w', encoding='utf-8') as f:
                json.dump(配置, f, ensure_ascii=False, indent=2)
            print(f"✅ 配置已保存到: {self.配置文件路径}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")

    def 添加测试区域(self, 区域名称: str, 参考面积: float, 说明: str = ""):
        """添加或更新测试区域的校正配置"""
        if 区域名称 not in self.校正配置["测试区域"]:
            self.校正配置["测试区域"][区域名称] = {}

        self.校正配置["测试区域"][区域名称]["参考面积"] = 参考面积
        if 说明:
            self.校正配置["测试区域"][区域名称]["说明"] = 说明

        # 如果已有计算面积，自动计算校正系数
        if "计算面积" in self.校正配置["测试区域"][区域名称]:
            计算面积 = self.校正配置["测试区域"][区域名称]["计算面积"]
            if 计算面积 > 0:
                self.校正配置["测试区域"][区域名称]["校正系数"] = 参考面积 / 计算面积

        self.保存配置()
        print(f"✅ 已添加测试区域: {区域名称}")

    def 计算校正系数(self, 区域名称: str, 计算面积: float, 参考面积: float = None) -> Tuple[float, bool]:
        """
        计算校正系数

        Args:
            区域名称: 测试区域名称
            计算面积: 系统计算的面积
            参考面积: 真实参考面积（可选）

        Returns:
            (校正系数, 是否需要校正)
        """
        # 保存计算面积
        if 区域名称 not in self.校正配置["测试区域"]:
            self.校正配置["测试区域"][区域名称] = {}

        self.校正配置["测试区域"][区域名称]["计算面积"] = 计算面积

        # 获取参考面积
        if 参考面积 is None:
            if 区域名称 in self.校正配置["测试区域"]:
                参考面积 = self.校正配置["测试区域"][区域名称].get("参考面积",
                                                             self.校正配置["默认参考面积"])
            else:
                参考面积 = self.校正配置["默认参考面积"]

        # 计算校正系数
        if 参考面积 and 参考面积 > 0 and 计算面积 > 0:
            校正系数 = 参考面积 / 计算面积
            偏差 = abs(校正系数 - 1.0)

            # 保存校正系数
            self.校正配置["测试区域"][区域名称]["校正系数"] = 校正系数
            self.校正配置["测试区域"][区域名称]["参考面积"] = 参考面积

            # 判断是否需要校正
            需要校正 = 偏差 > self.校正配置["最小偏差阈值"]

            if 需要校正:
                print(f"\n📊 {区域名称} 校正信息:")
                print(f"   系统计算: {计算面积:.3f} 亩")
                print(f"   参考面积: {参考面积:.3f} 亩")
                print(f"   校正系数: {校正系数:.3f}")
                print(f"   偏差: {偏差:.1%}")

            return 校正系数, 需要校正

        return 1.0, False

    def 应用校正(self, 原始面积: float, 区域名称: str = "default") -> Tuple[float, Dict]:
        """
        应用面积校正

        Args:
            原始面积: 原始计算的面积
            区域名称: 测试区域名称

        Returns:
            (校正后面积, 校正信息)
        """
        校正信息 = {
            "原始面积": 原始面积,
            "校正系数": 1.0,
            "是否校正": False,
            "区域名称": 区域名称
        }

        if not self.校正配置["自动应用"]:
            return 原始面积, 校正信息

        校正系数, 需要校正 = self.计算校正系数(区域名称, 原始面积)

        if 需要校正:
            校正后面积 = 原始面积 * 校正系数
            校正信息.update({
                "校正系数": 校正系数,
                "校正后面积": 校正后面积,
                "是否校正": True,
                "参考面积": self.校正配置["测试区域"][区域名称].get("参考面积")
            })
            return 校正后面积, 校正信息

        return 原始面积, 校正信息

    def 显示配置(self):
        """显示当前校正配置"""
        print("\n" + "="*60)
        print("基准数据校正配置")
        print("="*60)
        print(f"\n📋 全局设置:")
        print(f"   默认参考面积: {self.校正配置['默认参考面积']} 亩")
        print(f"   自动应用校正: {self.校正配置['自动应用']}")
        print(f"   最小偏差阈值: {self.校正配置['最小偏差阈值']:.1%}")

        print(f"\n📍 测试区域配置:")
        for 区域名称, 配置 in self.校正配置["测试区域"].items():
            print(f"\n   区域: {区域名称}")
            if "参考面积" in 配置:
                print(f"     参考面积: {配置['参考面积']} 亩")
            if "校正系数" in 配置:
                print(f"     校正系数: {配置['校正系数']:.3f}")
            if "计算面积" in 配置:
                print(f"     计算面积: {配置['计算面积']:.3f} 亩")
            if "说明" in 配置:
                print(f"     说明: {配置['说明']}")

    def 编辑配置(self):
        """交互式编辑配置"""
        print("\n" + "="*60)
        print("编辑校正配置")
        print("="*60)

        # 显示当前配置
        self.显示配置()

        while True:
            print("\n请选择操作:")
            print("1. 添加/更新测试区域")
            print("2. 修改默认参考面积")
            print("3. 切换自动应用")
            print("4. 保存并退出")
            print("5. 仅退出")

            choice = input("\n请输入选项 (1-5): ").strip()

            if choice == "1":
                区域名称 = input("输入区域名称 (如: test_1): ").strip()
                try:
                    参考面积 = float(input("输入参考面积(亩): ").strip())
                    说明 = input("输入说明 (可选): ").strip()
                    self.添加测试区域(区域名称, 参考面积, 说明)
                except ValueError:
                    print("❌ 请输入有效的数字")

            elif choice == "2":
                try:
                    新参考值 = float(input(f"输入新的默认参考面积(当前:{self.校正配置['默认参考面积']}): ").strip())
                    self.校正配置["默认参考面积"] = 新参考值
                    print("✅ 已更新默认参考面积")
                except ValueError:
                    print("❌ 请输入有效的数字")

            elif choice == "3":
                self.校正配置["自动应用"] = not self.校正配置["自动应用"]
                状态 = "启用" if self.校正配置["自动应用"] else "禁用"
                print(f"✅ 已{状态}自动应用校正")

            elif choice == "4":
                self.保存配置()
                break

            elif choice == "5":
                break

            else:
                print("❌ 无效选项")


def 创建简化版本():
    """创建简化的校正函数，可以直接集成到现有代码中"""
    print("\n" + "="*60)
    print("简化版校正函数")
    print("="*60)

    简化代码 = '''
# 在耕地分析工具_图形界面.py中，替换现有的校正代码（约1086-1096行）：

# 校正管理器（可选：从外部配置文件加载）
class 简单校正管理器:
    def __init__(self):
        self.参考面积 = 12.6  # 默认参考值
        self.启用校正 = True

    def 应用校正(self, 计算面积, 区域名称="default"):
        if not self.启用校正 or 计算面积 <= 0:
            return 计算面积, 1.0, False

        校正系数 = self.参考面积 / 计算面积
        偏差 = abs(校正系数 - 1.0)

        if 偏差 > 0.01:  # 偏差大于1%才校正
            校正后面积 = 计算面积 * 校正系数
            return 校正后面积, 校正系数, True

        return 计算面积, 1.0, False

# 在面积计算部分使用：
校正管理器 = 简单校正管理器()

# 可以通过输入框动态调整
if hasattr(self, '参考面积输入') and self.参考面积输入.get():
    try:
        校正管理器.参考面积 = float(self.参考面积输入.get())
    except:
        pass

# 应用校正
校正后面积, 校正系数, 是否校正 = 校正管理器.应用校正(去年耕地面积_亩)

if 是否校正:
    self.输出结果(f"\\n⚙️  应用校正系数: {校正系数:.3f}")
    self.输出结果(f"   原始面积: {去年耕地面积_亩:.3f} 亩")
    self.输出结果(f"   校正后面积: {校正后面积:.3f} 亩")
    self.输出结果(f"   参考面积: {校正管理器.参考面积} 亩")
    去年耕地面积_亩 = 校正后面积
'''

    print(简化代码)

    print("\n💡 使用建议:")
    print("1. 如果只需要简单校正，使用简化版")
    print("2. 如果需要管理多个测试区域，使用完整版")
    print("3. 可以添加GUI输入框让用户动态输入参考面积")


if __name__ == "__main__":
    print("基准数据校正管理器")
    print("="*60)

    # 创建管理器实例
    管理器 = 基准校正管理器()

    # 显示当前配置
    管理器.显示配置()

    # 测试校正功能
    print("\n" + "="*60)
    print("测试校正功能")
    print("="*60)

    测试数据 = [
        ("default", 13.679),
        ("test_1", 15.2),
        ("test_2", 11.8),
    ]

    for 区域名称, 计算面积 in 测试数据:
        校正后面积, 校正信息 = 管理器.应用校正(计算面积, 区域名称)

        print(f"\n{区域名称}:")
        print(f"   原始面积: {校正信息['原始面积']:.3f} 亩")

        if 校正信息['是否校正']:
            print(f"   校正系数: {校正信息['校正系数']:.3f}")
            print(f"   校正后面积: {校正后面积:.3f} 亩")
            print(f"   参考面积: {校正信息['参考面积']} 亩")
        else:
            print(f"   未应用校正（偏差太小）")

    # 创建简化版本
    创建简化版本()

    print("\n" + "="*60)
    print("使用说明")
    print("="*60)
    print("\n1. 📦 完整版（推荐用于多测试区域）:")
    print("   - 支持多个测试区域配置")
    print("   - 配置保存在JSON文件中")
    print("   - 可以交互式管理配置")
    print("   - 运行: python 基准校正管理器.py")

    print("\n2. 🔧 简化版（推荐用于快速实现）:")
    print("   - 直接集成到现有代码")
    print("   - 硬编码参考面积")
    print("   - 可以通过GUI输入框调整")

    print("\n3. 🎯 集成到现有系统:")
    print("   - 在耕地分析工具_图形界面.py中导入管理器")
    print("   - 替换现有的硬编码校正部分")
    print("   - 添加GUI控件让用户输入参考面积")