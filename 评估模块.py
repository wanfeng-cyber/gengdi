"""
耕地识别系统评估模块
包含多种评估指标：RMSE、MAE、R²、IoU、F1分数等
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from datetime import datetime
import json

class 耕地评估器:
    """耕地识别结果评估器"""

    def __init__(self, 像素分辨率=1.0):
        """
        初始化评估器

        参数:
            像素分辨率: 每个像素代表的平方米数
        """
        self.像素分辨率 = 像素分辨率
        self.评估结果 = {}

    def 计算混淆矩阵(self, 预测结果, 真实标签):
        """
        计算混淆矩阵

        返回:
            TP, TN, FP, FN
        """
        # 确保输入是二值化的
        pred = (预测结果 > 0.5).astype(int)
        true = (真实标签 > 0.5).astype(int)

        # 计算混淆矩阵
        TP = np.sum((pred == 1) & (true == 1))
        TN = np.sum((pred == 0) & (true == 0))
        FP = np.sum((pred == 1) & (true == 0))
        FN = np.sum((pred == 0) & (true == 1))

        return TP, TN, FP, FN

    def 计算基础指标(self, 预测结果, 真实标签):
        """计算精确率、召回率、F1分数、准确率"""
        TP, TN, FP, FN = self.计算混淆矩阵(预测结果, 真实标签)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        return {
            '精确率(Precision)': precision,
            '召回率(Recall)': recall,
            'F1分数(F1-Score)': f1,
            '准确率(Accuracy)': accuracy,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }

    def 计算IoU(self, 预测结果, 真实标签):
        """
        计算IoU（交并比）
        """
        pred = (预测结果 > 0.5).astype(int)
        true = (真实标签 > 0.5).astype(int)

        intersection = np.sum(pred & true)
        union = np.sum(pred | true)

        iou = intersection / union if union > 0 else 0

        return iou

    def 计算Dice系数(self, 预测结果, 真实标签):
        """
        计算Dice系数
        """
        pred = (预测结果 > 0.5).astype(int)
        true = (真实标签 > 0.5).astype(int)

        intersection = np.sum(pred & true)
        dice = 2 * intersection / (np.sum(pred) + np.sum(true)) if (np.sum(pred) + np.sum(true)) > 0 else 0

        return dice

    def 计算面积指标(self, 预测结果, 真实标签):
        """
        计算面积相关指标
        """
        pred_pixels = np.sum(预测结果 > 0.5)
        true_pixels = np.sum(真实标签 > 0.5)

        pred_area = pred_pixels * self.像素分辨率  # 平方米
        true_area = true_pixels * self.像素分辨率  # 平方米

        # 转换为亩
        pred_area_mu = pred_area / 666.67
        true_area_mu = true_area / 666.67

        # 计算误差
        abs_error = abs(pred_area - true_area)
        rel_error = abs_error / true_area if true_area > 0 else 0

        return {
            '预测面积_平方米': pred_area,
            '真实面积_平方米': true_area,
            '预测面积_亩': pred_area_mu,
            '真实面积_亩': true_area_mu,
            '绝对误差_平方米': abs_error,
            '相对误差': rel_error,
            '像素差异': pred_pixels - true_pixels
        }

    def 计算回归指标(self, 预测结果, 真实标签):
        """
        计算回归评估指标（RMSE、MAE、R²等）
        将分割任务转换为回归问题评估
        """
        # 展平数组
        pred_flat = 预测结果.flatten()
        true_flat = 真实标签.flatten()

        # 计算各种回归指标
        mse = mean_squared_error(true_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_flat, pred_flat)
        r2 = r2_score(true_flat, pred_flat)

        # 计算自定义指标
        mape = np.mean(np.abs((true_flat - pred_flat) / (true_flat + 1e-8))) * 100  # MAPE

        return {
            'RMSE': rmse,
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'MAPE(%)': mape,
            '预测均值': np.mean(pred_flat),
            '真实均值': np.mean(true_flat),
            '预测标准差': np.std(pred_flat),
            '真实标准差': np.std(true_flat)
        }

    def 计算斑块级别指标(self, 预测结果, 真实标签, min_size=10):
        """
        计算斑块级别的指标
        """
        from scipy import ndimage

        # 标记连通区域
        pred_labeled, pred_num = ndimage.label(预测结果 > 0.5)
        true_labeled, true_num = ndimage.label(真实标签 > 0.5)

        # 计算斑块统计
        pred_sizes = [np.sum(pred_labeled == i) for i in range(1, pred_num + 1)]
        true_sizes = [np.sum(true_labeled == i) for i in range(1, true_num + 1)]

        # 过滤小斑块
        pred_sizes = [s for s in pred_sizes if s >= min_size]
        true_sizes = [s for s in true_sizes if s >= min_size]

        return {
            '预测斑块数': len(pred_sizes),
            '真实斑块数': len(true_sizes),
            '斑块数差异': len(pred_sizes) - len(true_sizes),
            '平均预测斑块大小': np.mean(pred_sizes) if pred_sizes else 0,
            '平均真实斑块大小': np.mean(true_sizes) if true_sizes else 0,
            '最大预测斑块': max(pred_sizes) if pred_sizes else 0,
            '最大真实斑块': max(true_sizes) if true_sizes else 0
        }

    def 全面评估(self, 预测结果, 真实标签):
        """
        执行全面评估
        """
        print("=" * 60)
        print("🔍 耕地识别系统评估报告")
        print("=" * 60)
        print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"图像尺寸: {预测结果.shape}")
        print(f"像素分辨率: {self.像素分辨率} 平方米/像素")
        print("-" * 60)

        # 1. 基础分类指标
        print("\n1. 分类准确度指标")
        print("-" * 40)
        基础指标 = self.计算基础指标(预测结果, 真实标签)
        for 指标, 值 in 基础指标.items():
            if 指标 in ['TP', 'TN', 'FP', 'FN']:
                print(f"  {指标}: {值:,}")
            else:
                print(f"  {指标}: {值:.4f}")

        # 2. IoU和Dice
        print("\n2. 分割质量指标")
        print("-" * 40)
        iou = self.计算IoU(预测结果, 真实标签)
        dice = self.计算Dice系数(预测结果, 真实标签)
        print(f"  IoU (交并比): {iou:.4f}")
        print(f"  Dice系数: {dice:.4f}")

        # 3. 面积指标
        print("\n3. 面积评估")
        print("-" * 40)
        面积指标 = self.计算面积指标(预测结果, 真实标签)
        print(f"  真实面积: {面积指标['真实面积_亩']:.2f} 亩")
        print(f"  预测面积: {面积指标['预测面积_亩']:.2f} 亩")
        print(f"  绝对误差: {面积指标['绝对误差_平方米']:.0f} 平方米")
        print(f"  相对误差: {面积指标['相对误差']*100:.2f}%")

        # 4. 回归指标
        print("\n4. 回归评估指标")
        print("-" * 40)
        回归指标 = self.计算回归指标(预测结果, 真实标签)
        print(f"  RMSE (均方根误差): {回归指标['RMSE']:.4f}")
        print(f"  MAE (平均绝对误差): {回归指标['MAE']:.4f}")
        print(f"  R² (决定系数): {回归指标['R²']:.4f}")
        print(f"  MAPE (平均绝对百分比误差): {回归指标['MAPE(%)']:.2f}%")

        # 5. 斑块指标
        print("\n5. 斑块级别分析")
        print("-" * 40)
        斑块指标 = self.计算斑块级别指标(预测结果, 真实标签)
        print(f"  真实斑块数: {斑块指标['真实斑块数']}")
        print(f"  预测斑块数: {斑块指标['预测斑块数']}")
        print(f"  斑块数差异: {斑块指标['斑块数差异']}")
        print(f"  平均斑块大小差异: {斑块指标['平均预测斑块大小'] - 斑块指标['平均真实斑块大小']:.1f} 像素")

        # 6. 总体评价
        print("\n6. 总体评价")
        print("-" * 40)
        if iou > 0.75:
            print("  🌟 识别质量: 优秀")
        elif iou > 0.65:
            print("  👍 识别质量: 良好")
        elif iou > 0.50:
            print("  👌 识别质量: 可接受")
        else:
            print("  ⚠️  识别质量: 需要改进")

        if 面积指标['相对误差'] < 0.05:
            print("  📊 面积精度: 优秀 (<5%)")
        elif 面积指标['相对误差'] < 0.10:
            print("  📊 面积精度: 良好 (<10%)")
        elif 面积指标['相对误差'] < 0.20:
            print("  📊 面积精度: 可接受 (<20%)")
        else:
            print("  📊 面积精度: 需要改进 (>20%)")

        # 保存评估结果
        self.评估结果 = {
            '基础指标': 基础指标,
            'IoU': iou,
            'Dice': dice,
            '面积指标': 面积指标,
            '回归指标': 回归指标,
            '斑块指标': 斑块指标
        }

        print("\n" + "=" * 60)

        return self.评估结果

    def 可视化对比(self, 预测结果, 真实标签, 保存路径=None):
        """
        生成可视化对比图
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('耕地识别评估可视化', fontsize=16)

        # 1. 原始图像（如果有）
        axes[0, 0].imshow(真实标签, cmap='gray')
        axes[0, 0].set_title('真实标签 (Ground Truth)')
        axes[0, 0].axis('off')

        # 2. 预测结果
        axes[0, 1].imshow(预测结果, cmap='gray')
        axes[0, 1].set_title('预测结果 (Prediction)')
        axes[0, 1].axis('off')

        # 3. 叠加显示
        叠加 = np.zeros((真实标签.shape[0], 真实标签.shape[1], 3))
        叠加[:,:,1] = 真实标签  # 真实-绿色
        叠加[:,:,0] = 预测结果  # 预测-红色
        axes[0, 2].imshow(叠加)
        axes[0, 2].set_title('叠加显示 (红=预测, 绿=真实)')
        axes[0, 2].axis('off')

        # 4. 正确预测（TP+TN）
        pred_binary = (预测结果 > 0.5).astype(int)
        true_binary = (真实标签 > 0.5).astype(int)
        correct = (pred_binary == true_binary).astype(int)
        axes[1, 0].imshow(correct, cmap='gray')
        axes[1, 0].set_title('正确预测 (白色=正确)')
        axes[1, 0].axis('off')

        # 5. 误报（FP）
        fp = ((pred_binary == 1) & (true_binary == 0)).astype(int)
        axes[1, 1].imshow(fp, cmap='Reds')
        axes[1, 1].set_title('误报 (False Positive)')
        axes[1, 1].axis('off')

        # 6. 漏报（FN）
        fn = ((pred_binary == 0) & (true_binary == 1)).astype(int)
        axes[1, 2].imshow(fn, cmap='Blues')
        axes[1, 2].set_title('漏报 (False Negative)')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if 保存路径:
            plt.savefig(保存路径, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存至: {保存路径}")

        plt.show()

    def 保存评估报告(self, 文件路径):
        """
        保存评估报告到JSON文件
        """
        if not self.评估结果:
            print("警告: 没有评估结果可保存")
            return

        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        report = {
            '评估时间': datetime.now().isoformat(),
            '像素分辨率': self.像素分辨率,
            '评估结果': convert_types(self.评估结果)
        }

        with open(文件路径, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"评估报告已保存至: {文件路径}")


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    真实标签 = np.random.choice([0, 1], size=(256, 256), p=[0.7, 0.3])

    # 添加一些噪声创建预测结果
    预测结果 = 真实标签.astype(float) + np.random.normal(0, 0.2, (256, 256))
    预测结果 = np.clip(预测结果, 0, 1)

    # 创建评估器
    评估器 = 耕地评估器(像素分辨率=0.5*0.5)  # 假设每个像素0.25平方米

    # 执行评估
    结果 = 评估器.全面评估(预测结果, 真实标签)

    # 生成可视化
    评估器.可视化对比(预测结果, 真实标签, "评估可视化.png")

    # 保存报告
    评估器.保存评估报告("评估报告.json")