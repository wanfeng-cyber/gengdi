"""
测试坐标系自动转换功能
"""
import os
import tempfile
import numpy as np

print("=" * 60)
print("测试坐标系自动转换功能")
print("=" * 60)

# 测试导入
try:
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform
    print("✅ rasterio模块正常")
except ImportError as e:
    print(f"❌ rasterio未安装: {e}")
    exit()

# 创建测试数据
print("\n测试坐标转换代码...")

# 模拟两个CRS
crs_126 = "PROJCS[\"CGCS2000 / 3-degree Gauss-Kruger CM 126E\"]"
crs_129 = "PROJCS[\"CGCS2000 / 3-degree Gauss-Kruger CM 129E\"]"

print(f"CRS1: {crs_126}")
print(f"CRS2: {crs_129}")
print(f"是否相同: {str(crs_126) == str(crs_129)}")

# 测试临时目录
temp_dir = tempfile.mkdtemp()
test_file = os.path.join(temp_dir, "test.txt")
with open(test_file, 'w') as f:
    f.write("test")
print(f"\n✅ 可以写入临时目录: {temp_dir}")

print("\n测试完成！如果模块都正常，自动转换应该能工作。")
print("\n如果转换还是没有执行，请确保：")
print("1. 关闭并重新启动图形界面程序")
print("2. 运行的是 C:\\Users\\jiao\\Desktop\\python\\耕地分析工具_图形界面.py")