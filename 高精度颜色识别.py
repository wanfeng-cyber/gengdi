"""
高精度颜色识别模块
专门针对5-10厘米分辨率的无人机图像优化
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
# 使用OpenCV替代skimage

class 高精度颜色识别器:
    """针对高分辨率图像的颜色识别器"""

    def __init__(self):
        # 根据图像分辨率动态调整参数
        self.参数表 = {
            '超高分辨率': {  # < 0.1米/像素
                'gaussian_sigma': 1.0,
                '形态学核大小': 2,
                '边缘平滑度': 0.5,
                '颜色聚类数': 8
            },
            '高分辨率': {  # 0.1-0.2米/像素
                'gaussian_sigma': 1.5,
                '形态学核大小': 3,
                '边缘平滑度': 1.0,
                '颜色聚类数': 6
            },
            '中高分辨率': {  # 0.2-0.5米/像素
                'gaussian_sigma': 2.0,
                '形态学核大小': 4,
                '边缘平滑度': 1.5,
                '颜色聚类数': 5
            }
        }

    def 获取参数(self, 分辨率):
        """根据分辨率获取参数"""
        if 分辨率 < 0.1:
            return self.参数表['超高分辨率']
        elif 分辨率 < 0.2:
            return self.参数表['高分辨率']
        else:
            return self.参数表['中高分辨率']

    def 预处理(self, 图像):
        """
        高精度预处理
        """
        # 1. 轻微的高斯模糊去噪（保留边缘）
        params = self.获取参数(0.1)  # 假设是超高分辨率
        模糊图像 = cv2.GaussianBlur(图像, (5,5), params['gaussian_sigma'])

        # 2. 转换到多个颜色空间
        RGB = 图像
        HSV = cv2.cvtColor(图像, cv2.COLOR_RGB2HSV)
        LAB = cv2.cvtColor(图像, cv2.COLOR_RGB2LAB)

        return RGB, HSV, LAB, 模糊图像

    def 自适应颜色阈值(self, 块, 去年块掩码=None):
        """
        自适应的颜色阈值识别
        根据局部统计特征动态调整阈值
        """
        h, w = 块.shape[:2]

        # 确保输入是uint8格式
        if 块.max() <= 1.0:
            块 = (块 * 255).astype(np.uint8)
        else:
            块 = 块.astype(np.uint8)

        # 转换到HSV
        HSV = cv2.cvtColor(块, cv2.COLOR_RGB2HSV)
        H, S, V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2]

        # 直接在RGB空间计算
        R, G, B = 块[:,:,0].astype(np.float32), 块[:,:,1].astype(np.float32), 块[:,:,2].astype(np.float32)

        # 计算块的统计特征
        if 去年块掩码 is not None and np.any(去年块掩码 > 0.5):
            # 有去年数据，基于去年耕地区域统计
            耕地掩码 = 去年块掩码 > 0.5
            耕地R = R[耕地掩码]
            耕地G = G[耕地掩码]
            耕地B = B[耕地掩码]

            if len(耕地R) > 10:
                # 计算耕地区域的颜色统计
                R_mean = np.mean(耕地R)
                G_mean = np.mean(耕地G)
                B_mean = np.mean(耕地B)

                # 动态阈值（基于去年数据）
                绿色强度 = np.mean(G_mean - R_mean)
                棕色强度 = np.mean(R_mean - B_mean)

                绿色阈值 = max(15, 绿色强度 * 0.5)
                棕色阈值 = max(10, 棕色强度 * 0.5)

                亮度_mean = (R_mean + G_mean + B_mean) / 3
                最小亮度 = max(30, 亮度_mean * 0.3)
                最大亮度 = min(220, 亮度_mean * 1.7)
            else:
                # 默认值
                绿色阈值 = 25
                棕色阈值 = 20
                最小亮度 = 40
                最大亮度 = 200
        else:
            # 无去年数据，使用更严格的经验值（减少识别面积）
            绿色阈值 = 35  # 从30提高到35
            棕色阈值 = 30  # 从25提高到30
            最小亮度 = 60  # 从50提高到60
            最大亮度 = 190  # 从200降低到190

        # 1. 绿色检测（更严格的条件）
        # 绿色特征：G明显大于R和B
        green_diff_rg = G - R
        green_diff_bg = G - B
        mask_green = (green_diff_rg > 绿色阈值) & (green_diff_bg > 绿色阈值) & (G > 80)  # 从60提高到80

        # 2. 棕色检测（耕地、裸土）
        # 棕色特征：R > G > B，或 R > B 且色调偏暖
        brown_diff_rb = R - B
        brown_diff_rg = np.maximum(0, R - G)
        mask_brown = (brown_diff_rb > 棕色阈值) & (brown_diff_rg > 10) & (R > 80) & (B < R * 0.7)  # 提高要求

        # 3. 黄色检测（干草、成熟作物）
        # 黄色特征：R和G都高，B较低
        mask_yellow = (R > 120) & (G > 120) & (B < 100) & (np.abs(R - G) < 25)  # 提高要求

        # 4. 排除非耕地
        亮度 = (R + G + B) / 3

        # 排除过暗（阴影、水体）
        mask_too_dark = 亮度 < 最小亮度

        # 排除过亮（建筑物、白云）
        mask_too_bright = 亮度 > 最大亮度

        # 排除水体（蓝色特征）
        blue_dominant = (B > R * 1.3) & (B > G * 1.3) & (B > 80)  # 更严格

        # 排除纯灰色（道路、岩石）
        rg_diff = np.abs(R - G)
        gb_diff = np.abs(G - B)
        rb_diff = np.abs(R - B)
        gray_like = (rg_diff < 10) & (gb_diff < 10) & (rb_diff < 10)  # 更严格

        # 额外排除：白色或浅色（建筑物、道路）
        white_like = (R > 200) & (G > 200) & (B > 200)

        # 合并所有耕地掩码
        mask_farmland = (mask_green | mask_brown | mask_yellow) & \
                       ~mask_too_dark & ~mask_too_bright & \
                       ~blue_dominant & ~gray_like & \
                       ~white_like

        return mask_farmland.astype(float)

    def 局部优化(self, 块, 初始掩码):
        """
        局部优化掩码（使用OpenCV实现）
        """
        # 1. 形态学操作
        params = self.获取参数(0.1)
        核大小 = params['形态学核大小']

        # 创建形态学核
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (核大小, 核大小))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (核大小+1, 核大小+1))

        # 开运算去除小噪声
        mask_opened = cv2.morphologyEx(初始掩码.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)

        # 闭运算填充小孔洞
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

        # 2. 边缘平滑
        mask_smooth = cv2.GaussianBlur(mask_closed.astype(float),
                                     (2*int(params['边缘平滑度']*2)+1,
                                      2*int(params['边缘平滑度']*2)+1),
                                     params['边缘平滑度'])

        # 3. 阈值二值化（使用OTSU）
        _, mask_final = cv2.threshold((mask_smooth * 255).astype(np.uint8),
                                     0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return mask_final.astype(float) / 255.0

    def 智能颜色聚类(self, 块):
        """
        使用K-means聚类识别耕地颜色
        """
        # 确保输入是正确格式
        if 块.max() <= 1.0:
            块 = (块 * 255).astype(np.uint8)
        else:
            块 = 块.astype(np.uint8)

        # 将图像reshape为像素数组
        像素数组 = 块.reshape((-1, 3)).astype(np.float32)

        # 定义聚类数
        params = self.获取参数(0.1)
        k = params['颜色聚类数']

        # K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        标签 = kmeans.fit_predict(像素数组)
        中心点 = kmeans.cluster_centers_

        # 识别耕地聚类（绿色或棕色）
        耕地聚类 = []
        for i in range(k):
            center = 中心点[i]
            # 直接在RGB空间判断，避免OpenCV转换错误
            r, g, b = center[0], center[1], center[2]

            # 判断是否为耕地颜色
            # 绿色：g > r*1.1 且 g > b*1.1
            is_green = (g > r * 1.1) and (g > b * 1.1) and (g > 50)

            # 棕色：r > g*1.1 且 r > b*1.0（红褐色）
            is_brown = (r > g * 1.1) and (r > b * 1.0) and (r > 60)

            # 黄色（干草）：r > b*1.2 且 g > b*1.2
            is_yellow = (r > b * 1.2) and (g > b * 1.2) and (r > 80 and g > 80)

            # 排除过暗或过亮
            not_too_dark = (r + g + b) > 60
            not_too_bright = (r + g + b) < 600

            if (is_green or is_brown or is_yellow) and not_too_dark and not_too_bright:
                耕地聚类.append(i)

        # 创建掩码
        掩码 = np.zeros(len(标签))
        for 聚类ID in 耕地聚类:
            掩码[标签 == 聚类ID] = 1

        return 掩码.reshape(块.shape[:2])

    def 多方法融合(self, 块, 去年块掩码=None):
        """
        融合多种识别方法
        """
        # 确保输入格式正确
        if 块.max() <= 1.0:
            块 = (块 * 255).astype(np.uint8)
        else:
            块 = 块.astype(np.uint8)

        # 方法1：自适应阈值（主要方法）
        mask1 = self.自适应颜色阈值(块, 去年块掩码)

        # 方法2：颜色聚类（辅助方法）
        try:
            mask2 = self.智能颜色聚类(块)
        except Exception as e:
            # print(f"  ⚠️ 颜色聚类失败，使用备用方法: {str(e)}")
            mask2 = mask1.copy()  # 使用方法1的结果作为备用

        # 方法3：基于LAB空间的增强检测（保守方法）
        try:
            mask3 = self.LAB空间增强检测(块)
        except Exception as e:
            # print(f"  ⚠️ LAB空间检测失败，使用备用方法: {str(e)}")
            mask3 = np.zeros_like(mask1)  # 使用空掩码作为备用

        # 严格的融合策略
        if 去年块掩码 is not None and np.any(去年块掩码 > 0.5):
            # 有去年数据时，使用保守策略
            # 要求至少两个方法都同意才认为是耕地
            agree_12 = (mask1 > 0.5) & (mask2 > 0.5)
            agree_13 = (mask1 > 0.5) & (mask3 > 0.5)
            agree_23 = (mask2 > 0.5) & (mask3 > 0.5)

            # 至少两个方法同意
            融合掩码 = (agree_12 | agree_13 | agree_23).astype(float)

            # 在去年的耕地区域，稍微放宽条件（信任单一方法）
            去年耕地 = 去年块掩码 > 0.5
            融合掩码[去年耕地] = np.maximum(融合掩码[去年耕地], mask1[去年耕地])
        else:
            # 无去年数据时，使用更严格的策略
            # 要求主要方法（自适应）且至少一个辅助方法同意
            主方法确认 = mask1 > 0.5
            辅助确认 = (mask2 > 0.5) | (mask3 > 0.5)
            融合掩码 = (主方法确认 & 辅助确认).astype(float)

        # 局部优化（轻微的形态学处理）
        try:
            最终掩码 = self.局部优化(块, 融合掩码)
        except Exception as e:
            # print(f"  ⚠️ 局部优化失败，使用融合结果: {str(e)}")
            最终掩码 = 融合掩码

        return 最终掩码

    def LAB空间增强检测(self, 块):
        """
        在LAB空间进行增强检测
        """
        LAB = cv2.cvtColor(块, cv2.COLOR_RGB2LAB)
        L, A, B = LAB[:,:,0], LAB[:,:,1], LAB[:,:,2]

        # 1. 绿色植被检测（高A值表示偏红，低A值表示偏绿）
        green_a = A < np.mean(A) - np.std(A)

        # 2. 黄色/棕色检测（高B值表示偏黄）
        yellow_b = B > np.mean(B) + np.std(A)

        # 3. 亮度调整（避免过暗或过亮）
        bright = (L > 50) & (L < 200)

        mask = (green_a | yellow_b) & bright

        return mask.astype(float)


def 使用示例():
    """
    在耕地分析系统中使用高精度识别的示例
    """
    print("""
在耕地分析系统.py中替换颜色识别部分的代码：

1. 在文件开头添加：
from 高精度颜色识别 import 高精度颜色识别器

2. 创建识别器实例：
识别器 = 高精度颜色识别器()

3. 替换原来的颜色识别代码（第1084-1117行）：
# 使用高精度颜色识别
今年疑似耕地 = 识别器.多方法融合(块, 去年块掩码)

4. 这样可以显著提高5-10厘米分辨率图像的识别精度
    """)


if __name__ == "__main__":
    使用示例()