"""
增强的耕地识别模块
专门用于识别新增耕地和改进识别精度
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans

def 增强耕地识别(今年影像, 去年掩码, 新增耕地检测=True):
    """
    增强版耕地识别，专门检测新增耕地

    Args:
        今年影像: 今年的影像数据
        去年掩码: 去年的耕地掩码
        新增耕地检测: 是否启用新增耕地检测

    Returns:
        增强后的耕地掩码
    """

    # 1. 标准识别流程
    标准掩码 = 标准颜色识别(今年影像)

    if not 新增耕地检测 or 去年掩码 is None:
        return 标准掩码

    # 2. 新增耕地检测
    print("🔍 检测新增耕地区域...")

    # 找出去年不是耕地，但今年可能是耕地的区域
    非去年耕地 = 去年掩码 < 0.5

    # 在这些区域进行更宽松的颜色识别
    新增掩码 = 宽松颜色识别(今年影像, 非去年耕地)

    # 3. 合并结果
    增强掩码 = np.maximum(标准掩码, 新增掩码)

    # 4. 后处理：去除小的噪声
    增强掩码 = 后处理(增强掩码)

    return 增强掩码

def 标准颜色识别(影像):
    """标准的耕地颜色识别"""
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(影像, cv2.COLOR_RGB2HSV)

    # 绿色范围（植被）
    绿色下限 = np.array([35, 40, 40])
    绿色上限 = np.array([85, 255, 255])
    绿色掩码 = cv2.inRange(hsv, 绿色下限, 绿色上限)

    # 棕色范围（裸土）
    棕色下限 = np.array([8, 30, 30])
    棕色上限 = np.array([25, 255, 255])
    棕色掩码 = cv2.inRange(hsv, 棕色下限, 棕色上限)

    # 合并掩码
    耕地掩码 = cv2.bitwise_or(绿色掩码, 棕色掩码)

    # 归一化到0-1
    return 耕地掩码.astype(np.float32) / 255.0

def 宽松颜色识别(影像, 感兴趣区域):
    """用于检测新增耕地的宽松颜色识别"""

    # 创建掩码，只处理感兴趣的区域
    掩码 = (感兴趣区域 * 255).astype(np.uint8)

    # 转换到LAB色彩空间（更好的颜色区分）
    lab = cv2.cvtColor(影像, cv2.COLOR_RGB2LAB)

    # 使用K-means聚类识别耕地颜色
    pixels = lab[感兴趣区域].reshape(-1, 3)

    # 如果像素太少，返回空掩码
    if len(pixels) < 100:
        return np.zeros_like(影像[:,:,0])

    # K-means聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # 找到最像耕地的聚类（绿色或棕色）
    耕地掩码 = np.zeros_like(pixels[:,0])

    # LAB中的绿色和棕色特征
    for i in range(3):
        cluster_center = kmeans.cluster_centers_[i]
        # A通道：绿色为负，棕色为正
        # B通道：蓝色为负，黄色为正
        if -20 < cluster_center[1] < 30 and 10 < cluster_center[2] < 50:
            # 可能是耕地（绿色或棕黄色）
            耕地掩码[labels == i] = 1

    # 重塑为图像尺寸
    结果 = np.zeros_like(影像[:,:,0])
    结果[感兴趣区域] = 耕地掩码

    return 结果

def 后处理(掩码):
    """后处理：去除噪声和填充空洞"""

    # 二值化
    二值 = (掩码 > 0.5).astype(np.uint8)

    # 形态学操作
    kernel = np.ones((3,3), np.uint8)

    # 填充小空洞
    二值 = cv2.morphologyEx(二值, cv2.MORPH_CLOSE, kernel)

    # 去除小的噪声区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(二值, 8, cv2.CV_32S)

    # 保留面积大于阈值的区域
    min_area = 100  # 最小面积阈值
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            二值[labels == i] = 0

    # 转换回浮点
    return 二值.astype(np.float32)

def 边界扩展检测(今年影像, 去年掩码, 扩展像素=5):
    """
    检测去年耕地边界的扩展

    Args:
        今年影像: 今年的影像
        去年掩码: 去年的耕地掩码
        扩展像素: 边界扩展的像素数

    Returns:
        边界扩展的耕地掩码
    """
    # 找出去年耕地的边界
    kernel = np.ones((扩展像素*2+1, 扩展像素*2+1), np.uint8)
    扩展区域 = cv2.dilate(去年掩码, kernel)
    边界区域 = 扩展区域 - 去年掩码

    # 只在边界区域进行检测
    if np.sum(边界区域) == 0:
        return 去年掩码

    # 使用更敏感的颜色检测
    边界掩码 = 敏感颜色检测(今年影像, 边界区域 > 0.1)

    # 合并原有耕地和边界扩展
    结果 = np.maximum(去年掩码, 边界掩码)

    return 结果

def 敏感颜色检测(影像, 感兴趣区域):
    """
    在边界区域使用更敏感的颜色检测
    """
    # 转换到HSV
    hsv = cv2.cvtColor(影像, cv2.COLOR_RGB2HSV)

    # 更宽的颜色范围
    绿色下限 = np.array([30, 30, 30])
    绿色上限 = np.array([90, 255, 255])

    棕色下限 = np.array([5, 20, 20])
    棕色上限 = np.array([30, 255, 255])

    绿色掩码 = cv2.inRange(hsv, 绿色下限, 绿色上限)
    棕色掩码 = cv2.inRange(hsv, 棕色下限, 棕色上限)

    合并掩码 = cv2.bitwise_or(绿色掩码, 棕色掩码)

    # 应用感兴趣区域
    结果 = np.zeros_like(合并掩码)
    区域掩码 = (感兴趣区域 * 255).astype(np.uint8)
    结果 = cv2.bitwise_and(合并掩码, 区域掩码)

    return 结果.astype(np.float32) / 255.0

def 多尺度检测(影像, 尺度列表=[0.8, 1.0, 1.2]):
    """
    多尺度检测，提高识别鲁棒性
    """
    掩码列表 = []

    for 尺度 in 尺度列表:
        # 缩放图像
        if 尺度 != 1.0:
            h, w = 影像.shape[:2]
            new_h, new_w = int(h * 尺度), int(w * 尺度)
            缩放影像 = cv2.resize(影像, (new_w, new_h))
            缩放掩码 = 标准颜色识别(缩放影像)

            # 缩放回原始尺寸
            掩码 = cv2.resize(缩放掩码, (w, h))
        else:
            掩码 = 标准颜色识别(影像)

        掩码列表.append(掩码)

    # 平均所有尺度的结果
    平均掩码 = np.mean(掩码列表, axis=0)

    return (平均掩码 > 0.3).astype(np.float32)


# 创建一个修改建议
print("""
增强耕地识别建议
================

问题：今年的耕地面积应该比去年多，但系统计算的结果反而少了

可能的原因：
1. 系统过于依赖去年的掩码，没有识别新增耕地
2. 颜色识别阈值过于严格
3. 只处理边界变化，遗漏全新区域

建议的解决方案：

1. 在耕地分析系统中增加"新增耕地检测"模式：
   - 对非去年耕地区域使用更宽松的颜色识别
   - 使用K-means聚类自动学习耕地颜色特征

2. 改进边界检测：
   - 扩展去年耕地边界3-5像素进行重点检测
   - 使用更敏感的颜色阈值

3. 多尺度验证：
   - 在不同缩放比例下进行识别
   - 合并多个结果提高准确性

4. 后处理优化：
   - 填充小空洞
   - 去除噪声但保留小的新增耕地

修改建议：
在调用系统.使用模型预测耕地_大图时，可以添加一个参数：
- 检测新增耕地=True
- 边界扩展检测=True

这样可以提高新增耕地的识别率。
""")