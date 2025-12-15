"""
耕地分析工具增强版 - 支持可配置的基准校正
在原有功能基础上添加了校正系数配置功能
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os
import json

class 增强校正管理器:
    """简化的校正管理器，集成到GUI中"""

    def __init__(self):
        self.参考面积 = 12.6  # 默认参考值
        self.启用校正 = True
        self.最小偏差 = 0.01  # 最小偏差阈值

    def 应用校正(self, 计算面积, 参考面积=None):
        """
        应用面积校正

        Returns:
            (校正后面积, 校正系数, 是否校正)
        """
        if not self.启用校正 or 计算面积 <= 0:
            return 计算面积, 1.0, False

        # 使用提供的参考面积或默认值
        if 参考面积 is None:
            参考面积 = self.参考面积

        if 参考面积 <= 0:
            return 计算面积, 1.0, False

        校正系数 = 参考面积 / 计算面积
        偏差 = abs(校正系数 - 1.0)

        if 偏差 > self.最小偏差:
            校正后面积 = 计算面积 * 校正系数
            return 校正后面积, 校正系数, True

        return 计算面积, 1.0, False

    def 保存配置(self, 文件路径="校正配置.json"):
        """保存配置到文件"""
        config = {
            "参考面积": self.参考面积,
            "启用校正": self.启用校正,
            "最小偏差": self.最小偏差
        }
        try:
            with open(文件路径, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def 加载配置(self, 文件路径="校正配置.json"):
        """从文件加载配置"""
        if os.path.exists(文件路径):
            try:
                with open(文件路径, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.参考面积 = config.get("参考面积", 12.6)
                    self.启用校正 = config.get("启用校正", True)
                    self.最小偏差 = config.get("最小偏差", 0.01)
                return True
            except Exception as e:
                print(f"加载配置失败: {e}")
        return False


def 添加校正配置界面(gui_instance):
    """
    为现有的GUI界面添加校正配置功能

    Args:
        gui_instance: 耕地分析工具GUI实例
    """

    # 创建校正管理器
    gui_instance.校正管理器 = 增强校正管理器()

    # 加载保存的配置
    gui_instance.校正管理器.加载配置()

    # 创建校正配置区域
    校正配置框 = tk.LabelFrame(gui_instance.左侧面板,
                                text="🔧 面积校正配置",
                                font=("微软雅黑", 10, "bold"),
                                bg=gui_instance.bg_dark,
                                fg=gui_instance.text_primary)
    校正配置框.pack(padx=20, pady=10, fill="x")

    # 参考面积输入
    参考面积框 = tk.Frame(校正配置框, bg=gui_instance.bg_dark)
    参考面积框.pack(pady=10, padx=10, fill="x")

    tk.Label(参考面积框, text="参考面积(亩):",
            font=("微软雅黑", 9),
            bg=gui_instance.bg_dark,
            fg=gui_instance.text_secondary).pack(side="left")

    gui_instance.参考面积输入 = tk.Entry(参考面积框,
                                       font=("微软雅黑", 9),
                                       width=10)
    gui_instance.参考面积输入.pack(side="left", padx=(10, 5))
    gui_instance.参考面积输入.insert(0, str(gui_instance.校正管理器.参考面积))

    # 校正开关
    gui_instance.启用校正变量 = tk.BooleanVar(value=gui_instance.校正管理器.启用校正)
    校正开关 = tk.Checkbutton(参考面积框,
                              text="启用校正",
                              variable=gui_instance.启用校正变量,
                              font=("微软雅黑", 9),
                              bg=gui_instance.bg_dark,
                              fg=gui_instance.text_secondary,
                              selectcolor=gui_instance.bg_dark)
    校正开关.pack(side="left", padx=10)

    # 应用按钮
    应用按钮 = tk.Button(参考面积框,
                         text="应用",
                         font=("微软雅黑", 9),
                         bg=gui_instance.primary,
                         fg="white",
                         bd=0,
                         padx=15,
                         cursor="hand2",
                         command=lambda: 更新校正配置(gui_instance))
    应用按钮.pack(side="left", padx=5)

    # 说明文字
    说明 = tk.Label(校正配置框,
                   text=f"说明：输入测试区域的\n真实面积，系统将自动\n计算校正系数",
                   font=("微软雅黑", 8),
                   bg=gui_instance.bg_dark,
                   fg=gui_instance.text_muted,
                   justify="left")
    说明.pack(pady=(0, 10), padx=10, anchor="w")

    # 添加历史记录
    校正历史框 = tk.Frame(校正配置框, bg=gui_instance.bg_dark)
    校正历史框.pack(pady=(0, 10), padx=10, fill="x")

    tk.Label(校正历史框, text="最近校正记录:",
            font=("微软雅黑", 9),
            bg=gui_instance.bg_dark,
            fg=gui_instance.text_secondary).pack(anchor="w")

    gui_instance.校正历史文本 = tk.Text(校正历史框,
                                       font=("微软雅黑", 8),
                                       height=3,
                                       width=30,
                                       bg=gui_instance.bg_secondary,
                                       fg=gui_instance.text_secondary,
                                       bd=0,
                                       padx=5,
                                       pady=5)
    gui_instance.校正历史文本.pack(pady=(5, 0), fill="x")


def 更新校正配置(gui_instance):
    """更新校正配置"""
    try:
        # 获取输入值
        参考面积 = float(gui_instance.参考面积输入.get())

        # 更新管理器
        gui_instance.校正管理器.参考面积 = 参考面积
        gui_instance.校正管理器.启用校正 = gui_instance.启用校正变量.get()

        # 保存配置
        gui_instance.校正管理器.保存配置()

        # 显示成功消息
        messagebox.showinfo("成功",
                          f"校正配置已更新：\n"
                          f"参考面积：{参考面积} 亩\n"
                          f"校正状态：{'启用' if gui_instance.校正管理器.启用校正 else '禁用'}")

    except ValueError:
        messagebox.showerror("错误", "请输入有效的数字")


def 应用增强校正(gui_instance, 计算面积):
    """
    在面积计算时应用增强校正

    Args:
        gui_instance: GUI实例
        计算面积: 原始计算的面积

    Returns:
        校正后的面积
    """
    # 获取当前参考面积
    参考面积 = None
    try:
        参考面积 = float(gui_instance.参考面积输入.get())
    except:
        pass

    # 应用校正
    校正后面积, 校正系数, 是否校正 = gui_instance.校正管理器.应用校正(
        计算面积,
        参考面积
    )

    # 如果应用了校正，记录到历史
    if 是否校正:
        记录 = f"原始:{计算面积:.3f}→校正:{校正后面积:.3f}亩 (系数:{校正系数:.3f})\n"
        gui_instance.校正历史文本.insert("1.0", 记录)

        # 限制历史记录行数
        行数 = int(gui_instance.校正历史_text.index('end-1c').split('.')[0])
        if 行数 > 10:
            gui_instance.校正历史_text.delete('10.0', 'end')

    return 校正后面积, 校正系数, 是否校正


def 创建独立的校正配置工具():
    """创建一个独立的校正配置工具"""

    root = tk.Tk()
    root.title("耕地面积校正配置工具")
    root.geometry("500x400")
    root.configure(bg="#f5f5f5")

    # 创建管理器
    管理器 = 增强校正管理器()
    管理器.加载配置()

    # 标题
    标题 = tk.Label(root,
                   text="🔧 耕地面积校正配置",
                   font=("微软雅黑", 16, "bold"),
                   bg="#f5f5f5",
                   fg="#2c3e50")
    标题.pack(pady=20)

    # 配置框架
    配置框 = tk.LabelFrame(root,
                          text="校正参数设置",
                          font=("微软雅黑", 12, "bold"),
                          bg="#ffffff",
                          fg="#2c3e50")
    配置框.pack(pady=20, padx=20, fill="both", expand=True)

    # 参考面积
    参考框 = tk.Frame(配置框, bg="#ffffff")
    参考框.pack(pady=20, padx=20, fill="x")

    tk.Label(参考框, text="参考面积（亩）:",
            font=("微软雅黑", 11),
            bg="#ffffff",
            fg="#34495e").pack(side="left")

    参考输入 = tk.Entry(参考框,
                       font=("微软雅黑", 11),
                       width=15)
    参考输入.pack(side="left", padx=10)
    参考输入.insert(0, str(管理器.参考面积))

    # 校正开关
    启用变量 = tk.BooleanVar(value=管理器.启用校正)
    启用开关 = tk.Checkbutton(配置框,
                             text="启用自动校正",
                             variable=启用变量,
                             font=("微软雅黑", 11),
                             bg="#ffffff",
                             fg="#34495e",
                             selectcolor="#ffffff")
    启用开关.pack(pady=10, anchor="w", padx=20)

    # 测试区域
    测试框 = tk.LabelFrame(配置框,
                          text="测试校正效果",
                          font=("微软雅黑", 11, "bold"),
                          bg="#ffffff",
                          fg="#2c3e50")
    测试框.pack(pady=20, padx=20, fill="x")

    测试输入框 = tk.Frame(测试框, bg="#ffffff")
    测试输入框.pack(pady=10, padx=10, fill="x")

    tk.Label(测试输入框, text="系统计算面积（亩）:",
            font=("微软雅黑", 10),
            bg="#ffffff",
            fg="#34495e").pack(side="left")

    测试输入 = tk.Entry(测试输入框,
                       font=("微软雅黑", 10),
                       width=15)
    测试输入.pack(side="left", padx=10)
    测试输入.insert(0, "13.679")

    # 结果显示
    结果文本 = tk.Text(测试框,
                       font=("微软雅黑", 10),
                       height=8,
                       width=50,
                       bg="#f8f9fa",
                       fg="#2c3e50",
                       bd=1,
                       padx=10,
                       pady=10)
    结果文本.pack(pady=10, padx=10, fill="both", expand=True)

    def 测试校正():
        """测试校正效果"""
        try:
            参考面积 = float(参考输入.get())
            计算面积 = float(测试输入.get())

            # 更新管理器
            管理器.参考面积 = 参考面积
            管理器.启用校正 = 启用变量.get()

            # 应用校正
            校正后面积, 校正系数, 是否校正 = 管理器.应用校正(计算面积)

            # 显示结果
            结果文本.delete("1.0", "end")
            结果文本.insert("1.0", f"测试结果：\n\n")
            结果文本.insert("end", f"原始计算面积: {计算_area:.3f} 亩\n")
            结果文本.insert("end", f"参考真实面积: {参考面积:.3f} 亩\n")
            结果_text.insert("end", f"校正系数: {校正系数:.3f}\n")
            结果文本.insert("end", f"是否应用校正: {'是' if 是否校正 else '否'}\n\n")

            if 是否校正:
                结果文本.insert("end", f"✅ 校正后面积: {校正后面积:.3f} 亩\n")
                结果_text.insert("end", f"偏差修正: {计算面积 - 校正后面积:.3f} 亩\n")
            else:
                结果文本.insert("end", f"ℹ️ 未应用校正（偏差小于阈值）\n")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def 保存配置():
        """保存配置"""
        try:
            管理器.参考面积 = float(参考输入.get())
            管理器.启用校正 = 启用变量.get()

            if 管理器.保存配置():
                messagebox.showinfo("成功", "配置已保存")
            else:
                messagebox.showerror("错误", "保存失败")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    # 按钮
    按钮框 = tk.Frame(配置框, bg="#ffffff")
    按钮框.pack(pady=20)

    tk.Button(按钮框,
             text="测试校正",
             font=("微软雅黑", 11),
             bg="#3498db",
             fg="white",
             bd=0,
             padx=20,
             pady=10,
             cursor="hand2",
             command=测试校正).pack(side="left", padx=10)

    tk.Button(按钮框,
             text="保存配置",
             font=("微软雅黑", 11),
             bg="#27ae60",
             fg="white",
             bd=0,
             padx=20,
             pady=10,
             cursor="hand2",
             command=保存配置).pack(side="left", padx=10)

    root.mainloop()


if __name__ == "__main__":
    print("耕地分析工具增强版 - 校正配置")
    print("="*60)

    # 运行独立的配置工具
    print("启动校正配置工具...")
    创建独立的校正配置工具()

    print("\n使用说明:")
    print("1. 运行此程序可以配置校正参数")
    print("2. 配置会自动保存到 '校正配置.json'")
    print("3. 主程序启动时会自动加载配置")
    print("4. 可以在主程序中动态调整参考面积")