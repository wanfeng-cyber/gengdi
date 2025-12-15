"""
运行图形界面并测试
"""
import subprocess
import sys

print("=" * 60)
print("运行图形界面程序进行测试")
print("=" * 60)

print("\n正在启动: 耕地分析工具_图形界面.py")
print("\n请按以下步骤测试：")
print("1. 选择今年的TIF图像（jinnian.tif）")
print("2. 观察识别结果是否准确")
print("3. 检查相同图像的识别是否一致")
print("4. 查看不同图像的识别差异")
print("\n按回车键继续...")
input()

try:
    # 使用subprocess运行图形界面
    subprocess.run([sys.executable, "耕地分析工具_图形界面.py"], check=True)
except Exception as e:
    print(f"\n运行失败: {e}")
    print("\n可能的原因：")
    print("1. 图形界面需要额外的库（tkinter）")
    print("2. 当前环境配置不正确")
    print("3. 文件路径错误")

print("\n测试完成！")