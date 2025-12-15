"""
检测图形界面程序是否正确加载
"""
import sys
import os

print("=" * 60)
print("检测耕地分析工具_图形界面.py")
print("=" * 60)

# 检查文件
文件路径 = "C:\\Users\\jiao\\Desktop\\python\\耕地分析工具_图形界面.py"
if not os.path.exists(文件路径):
    print(f"❌ 文件不存在: {文件路径}")
    exit()

print(f"✅ 文件存在: {文件路径}")

# 读取文件并检查关键代码
print("\n检查关键代码...")
with open(文件路径, 'r', encoding='utf-8') as f:
    内容 = f.read()

# 检查是否有自动转换代码
if "正在自动转换坐标系" in 内容:
    print("✅ 找到自动转换代码")
else:
    print("❌ 没有找到自动转换代码")

# 检查是否有旧代码
if "为避免图像变形，未进行坐标转换" in 内容:
    print("❌ 仍有旧代码存在")
else:
    print("✅ 旧代码已清除")

# 检查关键函数
if "def 执行分析" in 内容:
    print("✅ 找到执行分析函数")

    # 提取执行分析函数的部分代码
    import re
    模式 = r'def 执行分析.*?(?=def |\nclass |\Z)'
    匹配 = re.search(模式, 内容, re.DOTALL)
    if 匹配:
        函数代码 = 匹配.group(0)
        print(f"\n执行分析函数长度: {len(函数代码)} 字符")

        # 检查函数中的关键部分
        if "from rasterio.warp import reproject" in 函数代码:
            print("✅ 包含rasterio.warp导入")

        # 查找坐标转换部分
        转换开始 = 函数代码.find("正在自动转换坐标系")
        if 转换开始 > -1:
            # 提取转换相关代码片段
            片段 = 函数代码[转换开始-100:转换开始+500]
            print("\n坐标转换代码片段:")
            print("-" * 40)
            print(片段[:200])
            print("...")
            print("-" * 40)

print("\n" + "=" * 60)
print("诊断完成！")
print("\n建议：")
print("1. 如果显示仍有旧代码，说明文件未正确保存")
print("2. 尝试手动删除旧代码行")
print("3. 或直接运行快速转换.py预处理图像")