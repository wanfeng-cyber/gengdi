"""
项目进度管理工具
自动记录和管理项目进度
"""

import os
import json
from datetime import datetime

class ProgressManager:
    """项目进度管理器"""

    def __init__(self):
        self.progress_file = "docs/PROGRESS.md"

    def 记录任务(self, 任务描述, 状态, 关键文件=None, 详细说明=""):
        """
        记录新任务到进度文件

        参数：
        - 任务描述：任务内容
        - 状态：已完成/进行中/待开始
        - 关键文件：涉及的文件列表
        - 详细说明：额外说明
        """
        时间戳 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        日期 = datetime.now().strftime("%Y-%m-%d")

        # 构建任务记录
        记录 = f"""
### 任务：{任务描述}
- **时间戳**：{时间戳}
- **任务描述**：{任务描述}
- **状态**：{状态} {'✅' if 状态 == '已完成' else '🔄' if 状态 == '进行中' else '⏳'}
"""

        if 关键文件:
            记录 += f"- **关键代码/文件**：\n"
            for 文件 in 关键文件:
                记录 += f"  - `{文件}`\n"

        if 详细说明:
            记录 += f"- **详细说明**：{详细说明}\n"

        # 检查是否需要添加新的日期标题
        if self._需要新日期标题(日期):
            日期标题 = f"\n---\n\n## {日期}\n\n"
            记录 = 日期标题 + 记录

        # 追加到文件
        with open(self.progress_file, 'a', encoding='utf-8') as f:
            f.write(记录 + '\n')

        print(f"✅ 任务已记录到 {self.progress_file}")

    def _需要新日期标题(self, 日期):
        """检查是否需要添加新的日期标题"""
        if not os.path.exists(self.progress_file):
            return True

        with open(self.progress_file, 'r', encoding='utf-8') as f:
            内容 = f.read()

        # 检查是否已经有今天的日期标题
        return f"## {日期}" not in 内容

    def 读取进度(self):
        """读取当前进度"""
        if not os.path.exists(self.progress_file):
            print("⚠️ 进度文件不存在")
            return None

        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return f.read()

    def 获取最新任务(self, 数量=5):
        """获取最新的几条任务"""
        内容 = self.读取进度()
        if not 内容:
            return []

        # 简单解析，返回最近的任务
        任务列表 = []
        for line in 内容.split('\n'):
            if '### 任务：' in line:
                任务列表.append(line)

        return 任务列表[-数量:]

    def 标记任务完成(self, 任务关键词):
        """标记任务为已完成（需要手动编辑）"""
        print(f"请手动在 {self.progress_file} 中将包含 '{任务关键词}' 的任务状态更新为 '已完成 ✅'")


# 创建全局实例
进度管理器 = ProgressManager()


def 记录进度(任务描述, 状态="进行中", 关键文件=None, 详细说明=""):
    """
    快捷函数：记录项目进度

    使用示例：
    记录进度("修复了一个bug", "已完成", ["file.py"], "修复了变量名错误")
    """
    进度管理器.记录任务(任务描述, 状态, 关键文件, 详细说明)


# 使用示例
if __name__ == "__main__":
    # 记录一个示例任务
    记录进度(
        "创建进度管理机制",
        "已完成",
        ["docs/PROGRESS.md", "progress_manager.py"],
        "实现了自动记录项目进度的功能"
    )

    # 读取并显示最新进度
    print("\n📊 最新项目进度：")
    最新任务 = 进度管理器.获取最新任务(3)
    for 任务 in 最新任务:
        print(f"  - {任务}")