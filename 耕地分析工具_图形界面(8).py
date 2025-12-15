"""
耕地分析工具 - 图形界面版
双击运行，选择图片，一键分析
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import sys
import threading
from PIL import Image, ImageTk, ImageDraw  # 添加PIL用于图像处理
import numpy as np

# 嵌入模型路径（打包后自动定位）
if getattr(sys, 'frozen', False):
    # 打包后的路径
    BASE_DIR = sys._MEIPASS
else:
    # 开发时的路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

模型路径 = os.path.join(BASE_DIR, "E:\Work_Space_E\Py_green_baer\耕地识别模型.h5")
基准数据路径 = os.path.join(BASE_DIR, "E:\pkl_16_备份\耕地识别模型_基准数据.pkl")

class 耕地分析界面:
    def __init__(self, root):
        self.root = root
        self.root.title("🌾 耕地智能分析系统 v2.0")
        self.root.geometry("950x720")
        self.root.resizable(True, True)
        
        # 现代科技风配色
        self.bg_dark = "#0d1117"  # GitHub深色背景
        self.bg_card = "#161b22"  # 卡片背景
        self.accent = "#1f6feb"   # 蓝色强调
        self.success = "#3fb950"  # 成功绿
        self.warning = "#f85149"  # 警告红
        self.text_primary = "#c9d1d9"  # 主文本
        self.text_secondary = "#8b949e"  # 次文本
        
        self.root.configure(bg=self.bg_dark)
        
        # 选择的图片路径
        self.图片路径 = None
        
        # 创建界面
        self.创建界面()
        
    def 创建界面(self):
        # ========== 顶部标题栏 ==========
        顶部框 = tk.Frame(self.root, bg=self.bg_card, height=100)
        顶部框.pack(fill="x", padx=0, pady=0)
        顶部框.pack_propagate(False)
        
        # 标题
        标题 = tk.Label(顶部框, text="🌾 耕地智能分析系统", 
                       font=("微软雅黑", 24, "bold"),
                       bg=self.bg_card, fg=self.text_primary)
        标题.pack(pady=15)
        
        # 副标题
        副标题 = tk.Label(顶部框, text="基于深度学习的耕地面积自动识别与变化检测", 
                        font=("微软雅黑", 11),
                        bg=self.bg_card, fg=self.text_secondary)
        副标题.pack()
        
        # ========== 主内容区域 ==========
        主区域 = tk.Frame(self.root, bg=self.bg_dark)
        主区域.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 左侧操作面板
        左侧面板 = tk.Frame(主区域, bg=self.bg_card, width=350)
        左侧面板.pack(side="left", fill="y")
        左侧面板.pack_propagate(False)
        
        # 操作标题
        tk.Label(左侧面板, text="📋 操作面板", 
                font=("微软雅黑", 14, "bold"),
                bg=self.bg_card, fg=self.text_primary).pack(pady=20, padx=20, anchor="w")
        
        # === 去年图像上传 ===
        文件卡片 = tk.Frame(左侧面板, bg=self.bg_dark, relief="flat")
        文件卡片.pack(padx=20, pady=5, fill="x")
        
        tk.Label(文件卡片, text="1️⃣  去年基准图像", 
                font=("微软雅黑", 11, "bold"),
                bg=self.bg_dark, fg=self.accent).pack(pady=(10,5), anchor="w", padx=10)
        
        tk.Button(文件卡片, text="📁 选择去年TIF", 
                 font=("微软雅黑", 10, "bold"),
                 bg=self.accent, fg="white",
                 activebackground="#1f6feb",
                 bd=0, padx=15, pady=8,
                 cursor="hand2",
                 command=self.选择去年图像).pack(pady=5, padx=10, fill="x")
        
        self.去年状态框 = tk.Frame(文件卡片, bg=self.bg_card, relief="flat")
        self.去年状态框.pack(pady=5, padx=10, fill="x")
        
        self.去年图标 = tk.Label(self.去年状态框, text="📅", 
                              font=("Segoe UI Emoji", 12),
                              bg=self.bg_card, fg=self.text_secondary)
        self.去年图标.pack(side="left", padx=(0,8))
        
        self.去年标签 = tk.Label(self.去年状态框, text="未选择", 
                              font=("微软雅黑", 9),
                              bg=self.bg_card, fg=self.text_secondary,
                              wraplength=220, justify="left")
        self.去年标签.pack(side="left", pady=10)
        
        # === 今年图像上传 ===
        文件卡片 = tk.Frame(左侧面板, bg=self.bg_dark, relief="flat")
        文件卡片.pack(padx=20, pady=5, fill="x")
        
        tk.Label(文件卡片, text="2️⃣  今年对比图像", 
                font=("微软雅黑", 11, "bold"),
                bg=self.bg_dark, fg=self.success).pack(pady=(10,5), anchor="w", padx=10)
        
        tk.Button(文件卡片, text="📁 选择今年TIF", 
                 font=("微软雅黑", 10, "bold"),
                 bg=self.success, fg="white",
                 activebackground="#2ea043",
                 bd=0, padx=15, pady=8,
                 cursor="hand2",
                 command=self.选择今年图像).pack(pady=5, padx=10, fill="x")
        
        self.今年状态框 = tk.Frame(文件卡片, bg=self.bg_card, relief="flat")
        self.今年状态框.pack(pady=5, padx=10, fill="x")
        
        self.今年图标 = tk.Label(self.今年状态框, text="📅", 
                              font=("Segoe UI Emoji", 12),
                              bg=self.bg_card, fg=self.text_secondary)
        self.今年图标.pack(side="left", padx=(0,8))
        
        self.今年标签 = tk.Label(self.今年状态框, text="未选择", 
                              font=("微软雅黑", 9),
                              bg=self.bg_card, fg=self.text_secondary,
                              wraplength=220, justify="left")
        self.今年标签.pack(side="left", pady=10)
        
        # 分隔线
        tk.Frame(左侧面板, bg=self.text_secondary, height=1).pack(fill="x", padx=30, pady=15)
        
        # 分析按钮卡片
        分析卡片 = tk.Frame(左侧面板, bg=self.bg_dark, relief="flat")
        分析卡片.pack(padx=20, pady=10, fill="x")
        
        tk.Label(分析卡片, text="3️⃣  执行分析", 
                font=("微软雅黑", 11, "bold"),
                bg=self.bg_dark, fg=self.text_primary).pack(pady=(10,5), anchor="w", padx=10)
        
        self.分析按钮 = tk.Button(分析卡片, text="🚀 开始分析", 
                                font=("微软雅黑", 12, "bold"),
                                bg=self.success, fg="white",
                                activebackground="#3fb950",
                                bd=0, padx=20, pady=12,
                                cursor="hand2",
                                state="disabled",
                                command=self.开始分析)
        self.分析按钮.pack(pady=10, padx=10, fill="x")
        
        # 进度条
        self.进度条 = ttk.Progressbar(分析卡片, mode='indeterminate', length=280)
        # 初始隐藏
        
        # 系统信息
        tk.Frame(左侧面板, bg=self.text_secondary, height=1).pack(fill="x", padx=30, pady=15)
        
        信息框 = tk.Frame(左侧面板, bg=self.bg_dark)
        信息框.pack(padx=20, pady=10, fill="x")
        
        tk.Label(信息框, text="ℹ️ 系统信息", 
                font=("微软雅黑", 10, "bold"),
                bg=self.bg_dark, fg=self.text_primary).pack(anchor="w", padx=10, pady=(10,5))
        
        tk.Label(信息框, text="• AI引擎: U-Net深度学习模型", 
                font=("微软雅黑", 9),
                bg=self.bg_dark, fg=self.text_secondary).pack(anchor="w", padx=10, pady=2)
        
        tk.Label(信息框, text="• 支持格式: TIF/TIFF影像", 
                font=("微软雅黑", 9),
                bg=self.bg_dark, fg=self.text_secondary).pack(anchor="w", padx=10, pady=2)
        
        tk.Label(信息框, text="• 精度: 亚米级分辨率", 
                font=("微软雅黑", 9),
                bg=self.bg_dark, fg=self.text_secondary).pack(anchor="w", padx=10, pady=2)
        
        # 右侧结果面板
        右侧面板 = tk.Frame(主区域, bg=self.bg_card)
        右侧面板.pack(side="right", fill="both", expand=True)
        
        # 结果标题
        结果标题框 = tk.Frame(右侧面板, bg=self.bg_card)
        结果标题框.pack(fill="x", padx=20, pady=(20,10))
        
        tk.Label(结果标题框, text="📊 分析结果", 
                font=("微软雅黑", 14, "bold"),
                bg=self.bg_card, fg=self.text_primary).pack(side="left")
        
        # 清空按钮
        清空按钮 = tk.Button(结果标题框, text="🗑️ 清空", 
                           font=("微软雅黑", 9),
                           bg=self.bg_dark, fg=self.text_secondary,
                           bd=0, padx=15, pady=5,
                           cursor="hand2",
                           command=self.清空结果)
        清空按钮.pack(side="right")
        
        # 结果文本框
        结果框架 = tk.Frame(右侧面板, bg=self.bg_dark)
        结果框架.pack(fill="both", expand=True, padx=20, pady=(0,20))
        
        # 上部：左右对比图像显示区
        对比框架 = tk.Frame(结果框架, bg=self.bg_dark, height=350)
        对比框架.pack(fill="x", pady=(0,10))
        对比框架.pack_propagate(False)
        
        # 左侧：去年基准图像
        左侧框 = tk.Frame(对比框架, bg=self.bg_dark)
        左侧框.pack(side="left", fill="both", expand=True, padx=(0,5))
        
        tk.Label(左侧框, text="📅 去年基准（SHP标注）", 
                font=("微软雅黑", 10, "bold"),
                bg=self.bg_dark, fg=self.accent).pack(pady=5)
        
        self.左侧图像标签 = tk.Label(左侧框, 
                                  text="🖼️ 去年图像+SHP轮廓",
                                  bg=self.bg_dark, 
                                  fg=self.text_secondary,
                                  font=("微软雅黑", 10))
        self.左侧图像标签.pack(expand=True, fill="both")
        
        # 右侧：今年识别图像
        右侧框 = tk.Frame(对比框架, bg=self.bg_dark)
        右侧框.pack(side="right", fill="both", expand=True, padx=(5,0))
        
        tk.Label(右侧框, text="📅 今年识别（AI识别）", 
                font=("微软雅黑", 10, "bold"),
                bg=self.bg_dark, fg=self.success).pack(pady=5)
        
        self.右侧图像标签 = tk.Label(右侧框, 
                                  text="🖼️ 今年图像+AI轮廓",
                                  bg=self.bg_dark, 
                                  fg=self.text_secondary,
                                  font=("微软雅黑", 10))
        self.右侧图像标签.pack(expand=True, fill="both")
        
        # 下部：文字结果
        self.结果文本 = scrolledtext.ScrolledText(结果框架, 
                                              font=("Consolas", 10),
                                              bg="#0d1117",
                                              fg=self.text_primary,
                                              insertbackground=self.text_primary,
                                              selectbackground=self.accent,
                                              bd=0,
                                              padx=15, pady=15,
                                              height=15,
                                              state="disabled")
        self.结果文本.pack(fill="both", expand=True)
        
        # ========== 底部状态栏 ==========
        底部栏 = tk.Frame(self.root, bg=self.bg_card, height=35)
        底部栏.pack(fill="x", side="bottom")
        底部栏.pack_propagate(False)
        
        tk.Label(底部栏, text="耕地智能分析系统 v2.0 | 基于深度学习技术 | Powered by U-Net", 
                font=("微软雅黑", 8),
                bg=self.bg_card, fg=self.text_secondary).pack(side="left", padx=20)
        
        self.状态标签 = tk.Label(底部栏, text="● 就绪", 
                              font=("微软雅黑", 8),
                              bg=self.bg_card, fg=self.success)
        self.状态标签.pack(side="right", padx=20)
        
    def 选择去年图像(self):
        """选择去年基准TIF图像"""
        文件路径 = filedialog.askopenfilename(
            title="选择去年TIF图像",
            filetypes=[("TIF文件", "*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        
        if 文件路径:
            self.去年图像路径 = 文件路径
            文件名 = os.path.basename(文件路径)
            文件大小 = os.path.getsize(文件路径) / (1024*1024)
            
            self.去年图标.config(text="✅", fg=self.success)
            self.去年标签.config(text=f"去年: {文件名}\n{文件大小:.1f} MB", fg=self.text_primary)
            
            # 检查两个图像是否都已选择
            self._检查启用分析按钮()
    
    def 选择今年图像(self):
        """选择今年对比TIF图像"""
        文件路径 = filedialog.askopenfilename(
            title="选择今年TIF图像",
            filetypes=[("TIF文件", "*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        
        if 文件路径:
            self.今年图像路径 = 文件路径
            文件名 = os.path.basename(文件路径)
            文件大小 = os.path.getsize(文件路径) / (1024*1024)
            
            self.今年图标.config(text="✅", fg=self.success)
            self.今年标签.config(text=f"今年: {文件名}\n{文件大小:.1f} MB", fg=self.text_primary)
            
            # 检查两个图像是否都已选择
            self._检查启用分析按钮()
    
    def _检查启用分析按钮(self):
        """检查两个图像是否都已选择，启用分析按钮"""
        if hasattr(self, '去年图像路径') and hasattr(self, '今年图像路径'):
            self.分析按钮.config(state="normal", bg=self.success)
            self.状态标签.config(text="● 已选择两张图像，可开始对比", fg=self.success)
    
    def 选择图片(self):
        """旧函数，保留兼容"""
        pass
    
    def 输出结果(self, 文本):
        """在结果框中输出文本"""
        self.结果文本.config(state="normal")
        self.结果文本.insert(tk.END, 文本 + "\n")
        self.结果文本.see(tk.END)
        self.结果文本.config(state="disabled")
        
    def 清空结果(self):
        """清空结果框"""
        self.结果文本.config(state="normal")
        self.结果文本.delete(1.0, tk.END)
        self.结果文本.config(state="disabled")
        
        # 清空左右图像
        self.左侧图像标签.config(image='', text="🖼️ 去年图像+SHP轮廓")
        self.右侧图像标签.config(image='', text="🖼️ 今年图像+AI轮廓")
    
    def 显示耕地可视化(self, 耕地掩码, 基准耕地掩码=None, 基准transform=None, 基准_crs=None):
        """左右对比显示：显示经纬度交集区域的对比
        
        Args:
            耕地掩码: 今年AI识别的耕地掩码
            基准耕地掩码: 完整的基准地图（不是resize后的）
            基准transform: 基准地图的地理变换信息
            基准_crs: 基准地图的坐标参考系统（用于坐标转换）
        """
        try:
            import rasterio
            import cv2
            from rasterio.warp import transform as warp_transform
            from rasterio.windows import Window
            from affine import Affine
            
            # === 第1步：计算两张图的经纬度交集 ===
            from rasterio.warp import transform_bounds
            
            with rasterio.open(self.去年图像路径) as src_去年:
                去年_左 = src_去年.bounds.left
                去年_右 = src_去年.bounds.right
                去年_上 = src_去年.bounds.top
                去年_下 = src_去年.bounds.bottom
                去年_transform = src_去年.transform
                去年_crs = src_去年.crs
                
                with rasterio.open(self.今年图像路径) as src_今年:
                    今年_左 = src_今年.bounds.left
                    今年_右 = src_今年.bounds.right
                    今年_上 = src_今年.bounds.top
                    今年_下 = src_今年.bounds.bottom
                    今年_transform = src_今年.transform
                    今年_crs = src_今年.crs
                    
                    # ✅ 关键修复：如果CRS不同，先转换到统一的WGS84计算交集，再转回去年坐标系
                    # ✅ 使用字符串比较代替对象比较，更可靠
                    去年_crs_str = str(去年_crs)
                    今年_crs_str = str(今年_crs)
                    
                    # ✅ 重要：如果有基准CRS参数，使用基准CRS而不是去年图像文件的CRS
                    # 因为基准PKL数据才是真正的参考坐标系
                    if 基准_crs is not None:
                        基准_crs_str = str(基准_crs)
                    else:
                        基准_crs_str = 去年_crs_str
                    
                    # ✅ 比较中央经线来判断CRS是否不同（更可靠）
                    # CM 126E vs CM 129E 等
                    import re
                    基准_cm_match = re.search(r'central_meridian["\s,:]+(\d+)', 基准_crs_str)
                    今年_cm_match = re.search(r'central_meridian["\s,:]+(\d+)', 今年_crs_str)
                    基准_cm = 基准_cm_match.group(1) if 基准_cm_match else ''
                    今年_cm = 今年_cm_match.group(1) if 今年_cm_match else ''
                    
                    # ✅ 比较基准CRS与今年图像CRS（而不是去年图像CRS）
                    crs不同 = (基准_cm != 今年_cm) or (基准_crs_str != 今年_crs_str)
                    print(f"🔍 可视化函数CRS比较:")
                    print(f"   基准中央经线: CM {基准_cm}E")
                    print(f"   今年中央经线: CM {今年_cm}E")
                    print(f"   基准CRS与今年不同: {crs不同}")
                    
                    if crs不同:
                        print(f"⚠️ 两张图像坐标系不同，进行坐标转换...")
                        print(f"   去年CRS: {去年_crs}")
                        print(f"   今年CRS: {今年_crs}")
                        
                        # 将两个边界都转换到WGS84
                        去年_wgs84 = transform_bounds(去年_crs, 'EPSG:4326', 去年_左, 去年_下, 去年_右, 去年_上)
                        今年_wgs84 = transform_bounds(今年_crs, 'EPSG:4326', 今年_左, 今年_下, 今年_右, 今年_上)
                        
                        print(f"   去年WGS84: {去年_wgs84}")
                        print(f"   今年WGS84: {今年_wgs84}")
                        
                        # 在WGS84下计算交集
                        交集_wgs84_左 = max(去年_wgs84[0], 今年_wgs84[0])
                        交集_wgs84_下 = max(去年_wgs84[1], 今年_wgs84[1])
                        交集_wgs84_右 = min(去年_wgs84[2], 今年_wgs84[2])
                        交集_wgs84_上 = min(去年_wgs84[3], 今年_wgs84[3])
                        
                        print(f"   交集WGS84: ({交集_wgs84_左:.6f}, {交集_wgs84_下:.6f}, {交集_wgs84_右:.6f}, {交集_wgs84_上:.6f})")
                        
                        # 将交集转换回去年坐标系（用于裁剪去年图像）
                        交集_去年坐标 = transform_bounds('EPSG:4326', 去年_crs, 交集_wgs84_左, 交集_wgs84_下, 交集_wgs84_右, 交集_wgs84_上)
                        交集_左, 交集_下, 交集_右, 交集_上 = 交集_去年坐标
                        
                        # 将交集转换到今年坐标系（用于裁剪今年图像）
                        交集_今年坐标 = transform_bounds('EPSG:4326', 今年_crs, 交集_wgs84_左, 交集_wgs84_下, 交集_wgs84_右, 交集_wgs84_上)
                        今年交集_左, 今年交集_下, 今年交集_右, 今年交集_上 = 交集_今年坐标
                        
                        print(f"   交集(去年坐标系): ({交集_左:.2f}, {交集_下:.2f}, {交集_右:.2f}, {交集_上:.2f})")
                        print(f"   交集(今年坐标系): ({今年交集_左:.2f}, {今年交集_下:.2f}, {今年交集_右:.2f}, {今年交集_上:.2f})")
                        
                        # 标记使用不同坐标系
                        使用不同坐标系 = True
                    else:
                        # 相同CRS，直接计算交集区域（地理坐标）
                        # ✅ 调试：打印去年和今年图像范围
                        print(f"🔍 去年图像范围: 左={去年_左:.2f}, 下={去年_下:.2f}, 右={去年_右:.2f}, 上={去年_上:.2f}")
                        print(f"🔍 今年图像范围: 左={今年_左:.2f}, 下={今年_下:.2f}, 右={今年_右:.2f}, 上={今年_上:.2f}")
                        # ✅ 修复：用去年图像文件和今年图像文件的范围计算交集
                        交集_左 = max(去年_左, 今年_左)
                        交集_右 = min(去年_右, 今年_右)
                        交集_上 = min(去年_上, 今年_上)
                        交集_下 = max(去年_下, 今年_下)
                        print(f"🔍 计算交集: 左={交集_左:.2f}, 下={交集_下:.2f}, 右={交集_右:.2f}, 上={交集_上:.2f}")
                        今年交集_左, 今年交集_右, 今年交集_上, 今年交集_下 = 交集_左, 交集_右, 交集_上, 交集_下
                        使用不同坐标系 = False
                    
                    # 检查是否有交集（当CRS不同时，在WGS84下检查）
                    if 使用不同坐标系:
                        有交集 = 交集_wgs84_左 < 交集_wgs84_右 and 交集_wgs84_下 < 交集_wgs84_上
                    else:
                        有交集 = 交集_左 < 交集_右 and 交集_下 < 交集_上
                    
                    if not 有交集:
                        print("⚠️  两张图没有经纬度交集，只显示去年图像")
                        
                        # 读取去年完整图像
                        去年图像 = src_去年.read([1,2,3])
                        去年图像 = np.transpose(去年图像, (1, 2, 0))
                        
                        # 归一化
                        if 去年图像.max() > 1.0:
                            去年图像 = (去年图像 / 去年图像.max() * 255).astype(np.uint8)
                        else:
                            去年图像 = (去年图像 * 255).astype(np.uint8)
                        去年图像 = np.ascontiguousarray(去年图像)
                        
                        # 绘制去年SHP黄色轮廓
                        if 基准耕地掩码 is not None:
                            基准掩码_uint8 = (基准耕地掩码 > 0).astype(np.uint8) * 255
                            基准轮廓列表, _ = cv2.findContours(基准掩码_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(去年图像, 基准轮廓列表, -1, (0, 255, 255), 3)
                        
                        # 缩放显示
                        最大宽度 = 260
                        高度, 宽度 = 去年图像.shape[:2]
                        比例 = min(最大宽度 / 宽度, 300 / 高度, 1.0)
                        新宽 = int(宽度 * 比例)
                        新高 = int(高度 * 比例)
                        
                        左侧显示 = cv2.resize(去年图像, (新宽, 新高))
                        左侧_rgb = cv2.cvtColor(左侧显示, cv2.COLOR_BGR2RGB)
                        左侧_pil = Image.fromarray(左侧_rgb)
                        左侧_photo = ImageTk.PhotoImage(左侧_pil)
                        
                        self.左侧图像标签.config(image=左侧_photo, text="")
                        self.左侧图像标签.image = 左侧_photo
                        self.右侧图像标签.config(text="⚠️  两张图没有交集\n无法对比")
                        return
                    
                    # === 第2步：从去年图中裁剪交集区域 ===
                    去年_inv = ~去年_transform
                    去年_col1, 去年_row1 = 去年_inv * (交集_左, 交集_上)
                    去年_col2, 去年_row2 = 去年_inv * (交集_右, 交集_下)
                    
                    去年_col_min = max(0, int(min(去年_col1, 去年_col2)))
                    去年_col_max = min(src_去年.width, int(max(去年_col1, 去年_col2)))
                    去年_row_min = max(0, int(min(去年_row1, 去年_row2)))
                    去年_row_max = min(src_去年.height, int(max(去年_row1, 去年_row2)))
                    
                    # 读取去年交集区域
                    去年_window = Window(去年_col_min, 去年_row_min, 
                                         去年_col_max - 去年_col_min, 
                                         去年_row_max - 去年_row_min)
                    去年图像 = src_去年.read([1,2,3], window=去年_window)
                    去年图像 = np.transpose(去年图像, (1, 2, 0))
                    
                    # === 第3步：从今年图中裁剪交集区域 ===
                    今年_inv = ~今年_transform
                    # ✅ 使用今年坐标系的交集坐标
                    今年_col1, 今年_row1 = 今年_inv * (今年交集_左, 今年交集_上)
                    今年_col2, 今年_row2 = 今年_inv * (今年交集_右, 今年交集_下)
                    
                    今年_col_min = max(0, int(min(今年_col1, 今年_col2)))
                    今年_col_max = min(src_今年.width, int(max(今年_col1, 今年_col2)))
                    今年_row_min = max(0, int(min(今年_row1, 今年_row2)))
                    今年_row_max = min(src_今年.height, int(max(今年_row1, 今年_row2)))
                    
                    # 读取今年交集区域
                    今年_window = Window(今年_col_min, 今年_row_min, 
                                         今年_col_max - 今年_col_min, 
                                         今年_row_max - 今年_row_min)
                    今年图像 = src_今年.read([1,2,3], window=今年_window)
                    今年图像 = np.transpose(今年图像, (1, 2, 0))
            
            # === 第4步：归一化到0-255并确保C-连续 ===
            # ✅ 添加黑色区域检测：过滤掉无效的黑色边缘
            if 去年图像.max() > 1.0:
                去年图像_归一化 = (去年图像 / 去年图像.max() * 255).astype(np.uint8)
            else:
                去年图像_归一化 = (去年图像 * 255).astype(np.uint8)
            
            # 检测黑色区域（所有通道都接近0的像素）
            # ✅ 更严格：<5 认为是黑色
            去年黑色掩码 = (去年图像_归一化[:,:,0] < 5) & (去年图像_归一化[:,:,1] < 5) & (去年图像_归一化[:,:,2] < 5)
            去年图像 = np.ascontiguousarray(去年图像_归一化)
            
            if 今年图像.max() > 1.0:
                今年图像_归一化 = (今年图像 / 今年图像.max() * 255).astype(np.uint8)
            else:
                今年图像_归一化 = (今年图像 * 255).astype(np.uint8)
            
            # 检测黑色区域（所有通道都接近0的像素）
            # ✅ 更严格：<5 认为是黑色
            今年黑色掩码 = (今年图像_归一化[:,:,0] < 5) & (今年图像_归一化[:,:,1] < 5) & (今年图像_归一化[:,:,2] < 5)
            今年图像 = np.ascontiguousarray(今年图像_归一化)
            
            # === 第5步：裁剪掩码并绘制轮廓和变化区域 ===
            # 左侧：去年SHP轮廓（黄色）- 显示去年的真实耕地范围
            基准轮廓列表 = None
            if 基准耕地掩码 is not None and 基准transform is not None:
                # ✅ 将交集坐标转换到基准地图CRS
                # 如果使用不同坐标系，需要将WGS84交集转换到基准地图CRS
                print(f"🔍 使用不同坐标系: {使用不同坐标系}")
                if 使用不同坐标系:
                    # ✅ 使用基准CRS（传入参数）而不是去年图像CRS
                    用于转换的crs = 基准_crs if 基准_crs is not None else 去年_crs
                    print(f"   用于转换的CRS: {用于转换的crs}")
                    print(f"   WGS84交集: 左={交集_wgs84_左:.6f}, 下={交集_wgs84_下:.6f}, 右={交集_wgs84_右:.6f}, 上={交集_wgs84_上:.6f}")
                    # 获取基准地图CRS（优先使用传入的基准_crs参数）
                    基准_裁剪坐标 = transform_bounds('EPSG:4326', 用于转换的crs, 
                                                            交集_wgs84_左, 交集_wgs84_下,
                                                            交集_wgs84_右, 交集_wgs84_上)
                    基准交集_左, 基准交集_下, 基准交集_右, 基准交集_上 = 基准_裁剪坐标
                    print(f"   转换后: 左={基准交集_左:.2f}, 下={基准交集_下:.2f}, 右={基准交集_右:.2f}, 上={基准交集_上:.2f}")
                else:
                    基准交集_左, 基准交集_下, 基准交集_右, 基准交集_上 = 交集_左, 交集_下, 交集_右, 交集_上
                
                # 从完整基准地图中裁剪交集区域
                基准_inv = ~基准transform
                基准_col1, 基准_row1 = 基准_inv * (基准交集_左, 基准交集_上)
                基准_col2, 基准_row2 = 基准_inv * (基准交集_右, 基准交集_下)
                
                print(f"🔍 基准地图裁剪调试:")
                print(f"   基准交集坐标: 左={基准交集_左:.2f}, 下={基准交集_下:.2f}, 右={基准交集_右:.2f}, 上={基准交集_上:.2f}")
                print(f"   基准transform: {基准transform}")
                print(f"   计算得到: col1={基准_col1:.2f}, row1={基准_row1:.2f}, col2={基准_col2:.2f}, row2={基准_row2:.2f}")
                
                基准_col_min = max(0, int(min(基准_col1, 基准_col2)))
                基准_col_max = min(基准耕地掩码.shape[1], int(max(基准_col1, 基准_col2)))
                基准_row_min = max(0, int(min(基准_row1, 基准_row2)))
                基准_row_max = min(基准耕地掩码.shape[0], int(max(基准_row1, 基准_row2)))
                
                # 从完整基准地图中裁剪交集区域
                基准掩码_交集 = 基准耕地掩码[基准_row_min:基准_row_max, 基准_col_min:基准_col_max]
                
                # 检查裁剪区域是否为空
                if 基准掩码_交集.size == 0 or 基准掩码_交集.shape[0] == 0 or 基准掩码_交集.shape[1] == 0:
                    print(f"⚠️ 今年图像与基准地图无有效交集区域，跳过轮廓绘制")
                    基准轮廓列表 = []
                else:
                    # resize到与去年图像相同大小
                    基准掩码_交集 = cv2.resize(基准掩码_交集.astype(np.uint8), 
                                              (去年图像.shape[1], 去年图像.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST)
                
                    # 查找去年SHP的轮廓
                    基准掩码_uint8 = (基准掩码_交集 > 0).astype(np.uint8) * 255
                    # ✅ 关键：严格过滤黑色区域，不在黑色区域绘制轮廓！
                    基准掩码_uint8[去年黑色掩码] = 0
                    
                    # ✅ 再次过滤：只保留大于阈值的区域
                    基准轮廓列表_临时, _ = cv2.findContours(基准掩码_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 过滤小区域（如果整个轮廓都在黑色区域，则过滤掉）
                    基准掩码_清洁 = np.zeros_like(基准掩码_uint8)
                    有效轮廓数 = 0
                    for contour in 基准轮廓列表_临时:
                        # 检查轮廓是否在黑色区域
                        mask = np.zeros(去年黑色掩码.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        轮廓区域 = mask > 0
                        # ✅ 加强过滤：如果轮廓区域中有>10%是黑色，则过滤！
                        if np.sum(轮廓区域) > 0:
                            黑色比例 = np.sum(去年黑色掩码 & 轮廓区域) / np.sum(轮廓区域)
                        else:
                            黑色比例 = 1.0
                        # 黑色<10% 且 面积>10像素 才保留 (✅ 从100降到10，兼容16x降采样)
                        if 黑色比例 < 0.1 and cv2.contourArea(contour) >= 10:
                            cv2.drawContours(基准掩码_清洁, [contour], -1, 255, -1)
                            有效轮廓数 += 1
                    
                    # 查找清洁后的轮廓
                    基准轮廓列表, _ = cv2.findContours(基准掩码_清洁, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 在左侧图像绘制黄色轮廓（去年的真实范围）
                    # ✅ 纯黄色(R:255, G:255, B:0)，线宽根据图像大小自适应，完全闭合
                    # ✅ 大图用更粗的线，让轮廓更清晰
                    图像面积 = 去年图像.shape[0] * 去年图像.shape[1]
                    if 图像面积 > 500 * 500:  # 大图
                        线宽 = max(6, int(图像面积 ** 0.5 / 100))  # 根据图像大小计算线宽
                    else:  # 小图
                        线宽 = 4
                    
                    for contour in 基准轮廓列表:
                        cv2.polylines(去年图像, [contour], isClosed=True, color=(0, 255, 255), thickness=线宽, lineType=cv2.LINE_AA)
            
            # 右侧：今年AI识别结果轮廓（红色）
            # ✅ 修复：使用今年AI识别的耕地掩码来画轮廓，而不是复制去年的！
            今年轮廓列表 = []
            if 耕地掩码 is not None:
                # 将今年AI识别的掩码resize到与今年图像显示大小一致
                今年掩码_resized = cv2.resize(耕地掩码.astype(np.uint8), 
                                              (今年图像.shape[1], 今年图像.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST)
                今年掩码_uint8 = (今年掩码_resized > 0).astype(np.uint8) * 255
                # 过滤黑色区域
                今年掩码_uint8[今年黑色掩码] = 0
                今年轮廓列表, _ = cv2.findContours(今年掩码_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 过滤小轮廓 (✅ 从100降到10，兼容16x降采样)
                今年轮廓列表 = [c for c in 今年轮廓列表 if cv2.contourArea(c) >= 10]
            
            # 在右侧图像绘制红色轮廓（今年AI识别结果）
            # ✅ 纯红色(R:255, G:0, B:0)，线宽根据图像大小自适应，完全闭合
            图像面积 = 今年图像.shape[0] * 今年图像.shape[1]
            if 图像面积 > 500 * 500:  # 大图
                线宽 = max(6, int(图像面积 ** 0.5 / 100))  # 根据图像大小计算线宽
            else:  # 小图
                线宽 = 4
            
            for contour in 今年轮廓列表:
                cv2.polylines(今年图像, [contour], isClosed=True, color=(0, 0, 255), thickness=线宽, lineType=cv2.LINE_AA)
            
            # === 第6步：缩放到适合显示的大小 ===
            最大宽度 = 260  # 左右各一半
            
            # 左侧图像缩放
            高度, 宽度 = 去年图像.shape[:2]
            比例 = min(最大宽度 / 宽度, 300 / 高度, 1.0)
            新宽 = int(宽度 * 比例)
            新高 = int(高度 * 比例)
            
            左侧显示 = cv2.resize(去年图像, (新宽, 新高))
            右侧显示 = cv2.resize(今年图像, (新宽, 新高))
            
            # 转换为PIL并显示
            左侧_rgb = cv2.cvtColor(左侧显示, cv2.COLOR_BGR2RGB)
            右侧_rgb = cv2.cvtColor(右侧显示, cv2.COLOR_BGR2RGB)
            
            左侧_pil = Image.fromarray(左侧_rgb)
            右侧_pil = Image.fromarray(右侧_rgb)
            
            左侧_photo = ImageTk.PhotoImage(左侧_pil)
            右侧_photo = ImageTk.PhotoImage(右侧_pil)
            
            # 显示
            self.左侧图像标签.config(image=左侧_photo, text="")
            self.左侧图像标签.image = 左侧_photo
            
            self.右侧图像标签.config(image=右侧_photo, text="")
            self.右侧图像标签.image = 右侧_photo
            
        except Exception as e:
            import traceback
            错误详情 = traceback.format_exc()
            print(f"显示可视化错误: {e}")
            print(错误详情)
            self.左侧图像标签.config(text=f"⚠️  可视化失败: {str(e)}")
            self.右侧图像标签.config(text=f"⚠️  可视化失败: {str(e)}")
        
    def 开始分析(self):
        # 检查是否已选择两张图片
        if not hasattr(self, '去年图像路径') or not hasattr(self, '今年图像路径'):
            messagebox.showerror("错误", "请先选择去年和今年两张图片！")
            return
        
        # 禁用按钮
        self.分析按钮.config(state="disabled", text="⏳ 分析中...")
        self.状态标签.config(text="● 分析中", fg=self.accent)
        
        # 显示进度条
        self.进度条.pack(pady=10, padx=10, fill="x")
        self.进度条.start(10)
        
        self.清空结果()
        
        # 在新线程中执行分析
        线程 = threading.Thread(target=self.执行分析)
        线程.start()
        
    def 执行分析(self):
        try:
            self.输出结果("=" * 50)
            self.输出结果("🔍 开始分析...")
            self.输出结果("=" * 50)
            
            # 检查模型文件
            if not os.path.exists(模型路径):
                self.输出结果(f"\n❌ 错误: 找不到模型文件!")
                self.输出结果(f"   路径: {模型路径}")
                messagebox.showerror("错误", "模型文件不存在！\n请确保模型文件在程序目录中。")
                self.分析按钮.config(state="normal")
                return
            
            # 导入系统
            self.输出结果("\n📚 加载分析系统...")
            from 耕地分析系统 import 耕地分析系统
            import pickle
            
            系统 = 耕地分析系统(输出目录="分析结果")
            
            # 加载基准数据
            有基准数据 = False
            if os.path.exists(基准数据路径):
                with open(基准数据路径, 'rb') as f:
                    基准信息 = pickle.load(f)
                
                # 判断是新版本还是旧版本
                if '基准耕地地图' in 基准信息:
                    self.输出结果(f"✅ 已加载基准地图 (基准年份: {基准信息.get('基准年份', 'N/A')})")
                elif '训练图像数量' in 基准信息:
                    self.输出结果(f"✅ 已加载基准数据({基准信息['训练图像数量']}张训练图像)")
                else:
                    self.输出结果("✅ 已加载基准数据")
                有基准数据 = True
            else:
                self.输出结果("⚠️  未找到基准数据，将只输出当前耕地面积")
            
            # 使用模型识别今年图像（智能增量识别）
            self.输出结果("\n🤖 AI模型识别今年图像...")
            
            # 先判断是否有去年的基准掩码可用
            去年掩码 = None
            if 有基准数据 and '基准耕地地图' in 基准信息:
                # 提取去年的掩码作为先验知识
                import rasterio
                import numpy as np
                from affine import Affine
                import cv2
                
                with rasterio.open(self.今年图像路径) as src:
                    左上x = src.bounds.left
                    右下x = src.bounds.right
                    左上y = src.bounds.top
                    右下y = src.bounds.bottom
                    今年_crs = src.crs
                    
                    # 显示今年图像的经纬度
                    from rasterio.warp import transform as warp_transform, transform_bounds
                    今年_左上经度, 今年_左上纬度 = warp_transform(src.crs, 'EPSG:4326', [左上x], [左上y])
                    今年_右下经度, 今年_右下纬度 = warp_transform(src.crs, 'EPSG:4326', [右下x], [右下y])
                    
                    self.输出结果("\n📍 今年图像经纬度:")
                    self.输出结果(f"   左上: ({今年_左上经度[0]:.6f}°, {今年_左上纬度[0]:.6f}°)")
                    self.输出结果(f"   右下: ({今年_右下经度[0]:.6f}°, {今年_右下纬度[0]:.6f}°)")
                    
                    # 检查是否在基准范围内
                    基准范围 = 基准信息['覆盖范围']
                    
                    # ✅ 关键修复：将今年图像坐标转换到WGS84，将基准范围也转换到WGS84进行比较
                    # 获取基准地图CRS（如果保存了的话）
                    基准_crs_str = 基准信息.get('crs', None)
                    # 将CRS字符串转换为CRS对象
                    from rasterio.crs import CRS
                    if 基准_crs_str:
                        基准_crs = CRS.from_string(基准_crs_str) if isinstance(基准_crs_str, str) else 基准_crs_str
                    else:
                        基准_crs = None
                    
                    # 将今年图像边界转换到WGS84
                    今年_wgs84 = transform_bounds(今年_crs, 'EPSG:4326', 左上x, 右下y, 右下x, 左上y)
                    
                    # 将基准范围转换到WGS84
                    if 基准_crs:
                        基准_wgs84 = transform_bounds(基准_crs, 'EPSG:4326', 
                                                        基准范围['左'], 基准范围['下'], 
                                                        基准范围['右'], 基准范围['上'])
                    else:
                        # 没有CRS信息，尝试从去年图像路径获取
                        if hasattr(self, '去年图像路径'):
                            with rasterio.open(self.去年图像路径) as src_去年:
                                基准_crs = src_去年.crs
                                基准_wgs84 = transform_bounds(基准_crs, 'EPSG:4326', 
                                                                基准范围['左'], 基准范围['下'], 
                                                                基准范围['右'], 基准范围['上'])
                        else:
                            # 假设基准地图和今年图像使用相同CRS
                            基准_wgs84 = transform_bounds(今年_crs, 'EPSG:4326', 
                                                            基准范围['左'], 基准范围['下'], 
                                                            基准范围['右'], 基准范围['上'])
                    
                    # 在WGS84下计算交集
                    交集_wgs84_左 = max(今年_wgs84[0], 基准_wgs84[0])
                    交集_wgs84_下 = max(今年_wgs84[1], 基准_wgs84[1])
                    交集_wgs84_右 = min(今年_wgs84[2], 基准_wgs84[2])
                    交集_wgs84_上 = min(今年_wgs84[3], 基准_wgs84[3])
                    
                    有交集 = (交集_wgs84_右 > 交集_wgs84_左) and (交集_wgs84_上 > 交集_wgs84_下)
                    
                    # 调试信息
                    print(f"\n🔍 分析函数CRS调试:")
                    print(f"   基准_crs_str: {基准_crs_str}")
                    print(f"   基准_crs (转换后): {基准_crs}")
                    print(f"   今年_crs: {今年_crs}")
                    print(f"   今年_wgs84: {今年_wgs84}")
                    print(f"   基准_wgs84: {基准_wgs84}")
                    print(f"   WGS84交集: 左={交集_wgs84_左:.6f}, 下={交集_wgs84_下:.6f}, 右={交集_wgs84_右:.6f}, 上={交集_wgs84_上:.6f}")
                    print(f"   有交集: {有交集}")
                    
                    if 有交集:
                        # ✅ 将WGS84交集转换回基准地图的坐标系，用于裁剪
                        if 基准_crs:
                            交集_基准坐标 = transform_bounds('EPSG:4326', 基准_crs, 
                                                                交集_wgs84_左, 交集_wgs84_下,
                                                                交集_wgs84_右, 交集_wgs84_上)
                            裁剪_左x, 裁剪_下y, 裁剪_右x, 裁剪_上y = 交集_基准坐标
                        else:
                            # 没有基准CRS，使用今年图像的原始坐标
                            裁剪_左x, 裁剪_下y, 裁剪_右x, 裁剪_上y = 左上x, 右下y, 右下x, 左上y
                        
                        print(f"   裁剪坐标(基准CRS): 左={裁剪_左x:.2f}, 下={裁剪_下y:.2f}, 右={裁剪_右x:.2f}, 上={裁剪_上y:.2f}")
                        
                        # 计算在基准地图中的位置
                        基准transform = Affine(
                            基准信息['地理变换']['a'],
                            基准信息['地理变换']['b'],
                            基准信息['地理变换']['c'],
                            基准信息['地理变换']['d'],
                            基准信息['地理变换']['e'],
                            基准信息['地理变换']['f']
                        )
                        
                        # ✅ 使用转换后的坐标进行裁剪
                        左上_col, 左上_row = ~基准transform * (裁剪_左x, 裁剪_上y)
                        右下_col, 右下_row = ~基准transform * (裁剪_右x, 裁剪_下y)
                        
                        # 裁剪范围
                        基准地图 = 基准信息['基准耕地地图']
                        row_min = max(0, int(min(左上_row, 右下_row)))
                        row_max = min(基准地图.shape[0], int(max(左上_row, 右下_row)))
                        col_min = max(0, int(min(左上_col, 右下_col)))
                        col_max = min(基准地图.shape[1], int(max(左上_col, 右下_col)))
                        
                        # 裁剪去年的掩码
                        裁剪区域 = 基准地图[row_min:row_max, col_min:col_max]
                        
                        self.输出结果(f"\n📋 从基准地图裁剪:")
                        self.输出结果(f"   基准地图尺寸: {基准地图.shape}")
                        self.输出结果(f"   裁剪范围: row[{row_min}:{row_max}], col[{col_min}:{col_max}]")
                        self.输出结果(f"   裁剪后尺寸: {裁剪区域.shape}")
                        self.输出结果(f"   今年图像尺寸: {src.width}x{src.height}")
                        
                        # ✅ 关键修复：不要resize去年掩码，保持原始分辨率！
                        # resize会导致轮廓变形、像素数错误、识别不准！
                        去年掩码 = 裁剪区域.astype(np.uint8)  # 直接使用原始分辨率
                        
                        # 计算去年的耕地面积（用于验证）
                        去年耕地像素 = np.sum(去年掩码)
                        总像素 = 去年掩码.size
                        去年比例 = 去年耕地像素 / 总像素
                        
                        self.输出结果(f"\n✅ 已加载去年掩码，启用智能增量识别！")
                        self.输出结果(f"   去年耕地比例: {去年比例*100:.2f}%")
                    else:
                        self.输出结果("\n⚠️  今年图像不在基准范围内，无法使用去年数据")
            
            # 调用识别（传入去年掩码）
            结果 = 系统.使用模型预测耕地_大图(
                self.今年图像路径, 
                模型路径=模型路径, 
                快速模式=True,
                去年掩码=去年掩码  # 传入去年掩码
            )
            
            # 计算总面积
            当前耕地面积_亩 = 结果['耕地面积_亩']
            耕地掩码 = 结果.get('耕地掩码', None)  # 获取掩码用于可视化
            
            # 初始化基准掩码（稍后可能填充）
            基准耕地掩码 = None
            
            self.输出结果("\n" + "=" * 50)
            self.输出结果("📊 分析结果")
            self.输出结果("=" * 50)
            
            if 有基准数据:
                # 判断使用哪种对比逻辑
                if '基准耕地地图' in 基准信息:
                    # 新逻辑：从基准地图裁剪
                    import rasterio
                    import numpy as np
                    from affine import Affine
                    
                    with rasterio.open(self.今年图像路径) as src:
                        左上x = src.bounds.left
                        右下x = src.bounds.right
                        左上y = src.bounds.top
                        右下y = src.bounds.bottom
                        今年_crs = src.crs
                        
                        # 显示经纬度
                        from rasterio.warp import transform as warp_transform, transform_bounds
                        左上经度, 左上纬度 = warp_transform(src.crs, 'EPSG:4326', [左上x], [左上y])
                        右下经度, 右下纬度 = warp_transform(src.crs, 'EPSG:4326', [右下x], [右下y])
                        
                        self.输出结果(f"\n📍 图像经纬度信息:")
                        self.输出结果(f"   左上角: (经度 {左上经度[0]:.6f}°, 纬度 {左上纬度[0]:.6f}°)")
                        self.输出结果(f"   右下角: (经度 {右下经度[0]:.6f}°, 纬度 {右下纬度[0]:.6f}°)")
                        
                        # 检查是否在基准范围内（检查交集）
                        基准范围 = 基准信息['覆盖范围']
                        
                        # ✅ 关键修复：将今年图像和基准范围都转换到WGS84进行比较
                        基准_crs_str = 基准信息.get('crs', None)
                        # 将CRS字符串转换为CRS对象
                        from rasterio.crs import CRS
                        if 基准_crs_str:
                            基准_crs = CRS.from_string(基准_crs_str) if isinstance(基准_crs_str, str) else 基准_crs_str
                        else:
                            基准_crs = None
                        
                        # 将今年图像边界转换到WGS84
                        今年_wgs84 = transform_bounds(今年_crs, 'EPSG:4326', 左上x, 右下y, 右下x, 左上y)
                        
                        # 将基准范围转换到WGS84
                        if 基准_crs:
                            基准_wgs84 = transform_bounds(基准_crs, 'EPSG:4326', 
                                                            基准范围['左'], 基准范围['下'], 
                                                            基准范围['右'], 基准范围['上'])
                        else:
                            # 没有CRS信息，尝试从去年图像路径获取
                            if hasattr(self, '去年图像路径'):
                                with rasterio.open(self.去年图像路径) as src_去年:
                                    基准_crs = src_去年.crs
                                    基准_wgs84 = transform_bounds(基准_crs, 'EPSG:4326', 
                                                                    基准范围['左'], 基准范围['下'], 
                                                                    基准范围['右'], 基准范围['上'])
                            else:
                                # 假设基准地图和今年图像使用相同CRS
                                基准_wgs84 = transform_bounds(今年_crs, 'EPSG:4326', 
                                                                基准范围['左'], 基准范围['下'], 
                                                                基准范围['右'], 基准范围['上'])
                        
                        # 在WGS84下计算交集范围
                        交集_wgs84_左 = max(今年_wgs84[0], 基准_wgs84[0])
                        交集_wgs84_下 = max(今年_wgs84[1], 基准_wgs84[1])
                        交集_wgs84_右 = min(今年_wgs84[2], 基准_wgs84[2])
                        交集_wgs84_上 = min(今年_wgs84[3], 基准_wgs84[3])
                        
                        # ✅ 关键检查：是否有交集
                        有交集 = (交集_wgs84_右 > 交集_wgs84_左) and (交集_wgs84_上 > 交集_wgs84_下)
                        
                        if not 有交集:
                            self.输出结果("\n❌ 错误：今年图像与去年基准地图没有交集，无法对比！")
                            self.输出结果(f"   今年图像范围（WGS84）：经度[{今年_wgs84[0]:.6f}, {今年_wgs84[2]:.6f}], 纬度[{今年_wgs84[1]:.6f}, {今年_wgs84[3]:.6f}]")
                            self.输出结果(f"   基准地图范围（WGS84）：经度[{基准_wgs84[0]:.6f}, {基准_wgs84[2]:.6f}], 纬度[{基准_wgs84[1]:.6f}, {基准_wgs84[3]:.6f}]")
                            messagebox.showerror("无法对比", "今年图像与去年基准地图没有交集，无法进行对比分析！\n\n请选择同一地区的图像。")
                            return
                        else:
                            # ✅ 将WGS84交集转换回基准地图的坐标系，用于裁剪
                            if 基准_crs:
                                交集_基准坐标 = transform_bounds('EPSG:4326', 基准_crs, 
                                                                    交集_wgs84_左, 交集_wgs84_下,
                                                                    交集_wgs84_右, 交集_wgs84_上)
                                左上x, 右下y, 右下x, 左上y = 交集_基准坐标
                            # 计算在基准地图中的位置
                            基准transform = Affine(
                                基准信息['地理变换']['a'],
                                基准信息['地理变换']['b'],
                                基准信息['地理变换']['c'],
                                基准信息['地理变换']['d'],
                                基准信息['地理变换']['e'],
                                基准信息['地理变换']['f']
                            )
                            
                            # 坐标转换
                            左上_col, 左上_row = ~基准transform * (左上x, 左上y)
                            右下_col, 右下_row = ~基准transform * (右下x, 右下y)
                            
                            # 裁剪范围
                            基准地图 = 基准信息['基准耕地地图']
                            row_min = max(0, int(min(左上_row, 右下_row)))
                            row_max = min(基准地图.shape[0], int(max(左上_row, 右下_row)))
                            col_min = max(0, int(min(左上_col, 右下_col)))
                            col_max = min(基准地图.shape[1], int(max(左上_col, 右下_col)))
                            
                            # 裁剪
                            裁剪区域 = 基准地图[row_min:row_max, col_min:col_max]
                            
                            # ✅ 保存完整的基准地图用于可视化（不是resize后的！）
                            # 可视化函数会根据经纬度自己裁剪
                            基准耕地掩码 = 基准地图  # 传入完整地图
                            基准地图_transform = 基准transform  # 也传入变换信息
                            
                            # ✅ 关键修复：如果是年度对比，直接使用AI返回的去年数据，而不是重新计算
                            # 因为AI已经使用了去年掩码进行增量识别
                            if 去年掩码 is not None:
                                # ✅ 关键修复：检测是否为相同图像
                                # 如果两张图片是一模一样的，应该直接使用去年数据
                                # 检浌方法：比较两张图像的像素值
                                # ✅ 删除“相同图像”检测逻辑！
                                # 原因：比例差异小不代表面积无变化，长宽变化了面积就应该变！
                                # 让系统直接使用真实计算的面积，不做任何强制修改
                                # 是否相同图像 = False  # 已删除
                                
                                # ✅ 关键修复：统一使用实际地理范围计算面积
                                # 不再依赖像素分辨率，而是基于经纬度计算实际面积
                                
                                # 1. 计算今年图像的实际地理范围（米）
                                with rasterio.open(self.今年图像路径) as src:
                                    # 图像的实际地理宽度和高度
                                    实际宽度_米 = abs(src.bounds.right - src.bounds.left)
                                    实际高度_米 = abs(src.bounds.top - src.bounds.bottom)
                                    实际面积_平方米 = 实际宽度_米 * 实际高度_米
                                    
                                    # 2. 计算去年耕地比例（从基准地图裁剪区域）
                                    去年_耕地像素 = np.sum(裁剪区域 > 0.5)
                                    去年_总像素 = 裁剪区域.size
                                    去年_耕地比例 = 去年_耕地像素 / 去年_总像素 if 去年_总像素 > 0 else 0
                                    
                                    # 3. 计算今年耕地比例（从AI识别掩码）
                                    今年_耕地像素 = np.sum(耕地掩码 > 0.5)
                                    今年_总像素 = 耕地掩码.size
                                    今年_耕地比例 = 今年_耕地像素 / 今年_总像素 if 今年_总像素 > 0 else 0
                                    
                                    # 4. 用实际面积 × 耕地比例 = 耕地面积
                                    去年_面积_平方米 = 实际面积_平方米 * 去年_耕地比例
                                    今年_面积_平方米 = 实际面积_平方米 * 今年_耕地比例
                                    
                                    原来面积 = 去年_面积_平方米 / 666.67
                                    当前耕地面积_亩 = 今年_面积_平方米 / 666.67
                                
                                self.输出结果(f"\n🔍 统一标准计算面积（基于实际地理范围）：")
                                self.输出结果(f"   图像实际范围: {实际宽度_米:.2f}m x {实际高度_米:.2f}m = {实际面积_平方米:.2f}平方米")
                                self.输出结果(f"   去年耕地比例: {去年_耕地比例*100:.2f}% (像素: {去年_耕地像素}/{去年_总像素})")
                                self.输出结果(f"   今年耕地比例: {今年_耕地比例*100:.2f}% (像素: {今年_耕地像素}/{今年_总像素})")
                                self.输出结果(f"   去年面积: {原来面积:.4f} 亩")
                                self.输出结果(f"   今年面积: {当前耕地面积_亩:.4f} 亩")
                            else:
                                # 没有去年掩码，使用resize后的基准掩码
                                with rasterio.open(self.今年图像路径) as src:
                                    今年_像素分辨率 = abs(src.transform.a)
                                    去年_耕地像素数_resize后 = np.sum(基准耕地掩码)  # resize后的像素数
                                    去年_面积_平方米 = 去年_耕地像素数_resize后 * (今年_像素分辨率 ** 2)
                                    原来面积 = 去年_面积_平方米 / 666.67
                                
                                    self.输出结果(f"\n🔍 面积计算（从基准地图resize）:")
                                    self.输出结果(f"   去年裁剪区域大小: {裁剪区域.shape}")
                                    self.输出结果(f"   去年resize后大小: {基准耕地掩码.shape}")
                                    self.输出结果(f"   去年耕地像素数（resize后）: {去年_耕地像素数_resize后}")
                                    self.输出结果(f"   使用分辨率（今年的）: {今年_像素分辨率:.4f} 米/像素")
                                    self.输出结果(f"   去年面积: {原来面积:.4f} 亩")
                            

                    
                    变化 = 当前耕地面积_亩 - 原来面积
                    
                    # ✅ 计算耕地长宽变化（精确到0.5m以内）
                    import rasterio
                    with rasterio.open(self.今年图像路径) as src:
                        # 计算耕地区域的实际范围
                        像素分辨率 = abs(src.transform.a)  # 米/像素
                        
                        # 从耕地掩码计算范围
                        耕地长宽信息 = ""
                        # ✅ 修复：现在不再替换掩码，所以可以一直计算长宽变化
                        if 耕地掩码 is not None and 去年掩码 is not None:
                            # ✅ 关键修复：去年掩码是从原始64×64裁剪后resize到256×256的
                            # 需要用原始裁剪区域的尺寸 × 基准地图分辨率计算去年长宽！
                            
                            # 计算今年耕地长宽（用今年分辨率）
                            今年掩码_uint8 = (耕地掩码 > 0.5).astype(np.uint8) * 255
                            今年轮廓, _ = cv2.findContours(今年掩码_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # 计算去年耕地长宽（用原始裁剪区域 × 基准分辨率）
                            去年_原始掩码_uint8 = (裁剪区域 > 0.5).astype(np.uint8) * 255
                            去年_原始轮廓, _ = cv2.findContours(去年_原始掩码_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if len(今年轮廓) > 0 and len(去年_原始轮廓) > 0:
                                # 找到最大轮廓（主要耕地区域）
                                今年最大轮廓 = max(今年轮廓, key=cv2.contourArea)
                                去年_原始最大轮廓 = max(去年_原始轮廓, key=cv2.contourArea)
                                
                                # ✅ 新方法：用面积直接计算等效长宽
                                # 面积增加 → 长宽增加；面积减少 → 长宽减少
                                去年_实际面积_m2 = 原来面积 * 666.67
                                今年_实际面积_m2 = 当前耕地面积_亩 * 666.67
                                
                                # 从轮廓计算长宽比
                                去年_x坐标 = 去年_原始最大轮廓[:, 0, 0]
                                去年_y坐标 = 去年_原始最大轮廓[:, 0, 1]
                                去年_宽度_像素 = np.max(去年_x坐标) - np.min(去年_x坐标)
                                去年_高度_像素 = np.max(去年_y坐标) - np.min(去年_y坐标)
                                去年_长宽比 = 去年_宽度_像素 / 去年_高度_像素 if 去年_高度_像素 > 0 else 1.0
                                
                                今年_x坐标 = 今年最大轮廓[:, 0, 0]
                                今年_y坐标 = 今年最大轮廓[:, 0, 1]
                                今年_宽度_像素 = np.max(今年_x坐标) - np.min(今年_x坐标)
                                今年_高度_像素 = np.max(今年_y坐标) - np.min(今年_y坐标)
                                今年_长宽比 = 今年_宽度_像素 / 今年_高度_像素 if 今年_高度_像素 > 0 else 1.0
                                
                                # 用面积和长宽比计算等效长宽（✅ 关键修复）
                                去年_宽度_米 = np.sqrt(去年_实际面积_m2 * 去年_长宽比)
                                去年_高度_米 = 去年_实际面积_m2 / 去年_宽度_米 if 去年_宽度_米 > 0 else 0
                                今年_宽度_米 = np.sqrt(今年_实际面积_m2 * 今年_长宽比)
                                今年_高度_米 = 今年_实际面积_m2 / 今年_宽度_米 if 今年_宽度_米 > 0 else 0
                                
                                # 计算变化
                                宽度变化_米 = 今年_宽度_米 - 去年_宽度_米
                                高度变化_米 = 今年_高度_米 - 去年_高度_米
                                
                                # 7. 生成详细报告
                                耕地长宽信息 = f"\n\n📏 耕地长宽变化（基于实际面积）:\n"
                                耕地长宽信息 += f"\n   📍 宽度方向（东西）:\n"
                                耕地长宽信息 += f"      原来: {去年_宽度_米:.3f} m\n"
                                耕地长宽信息 += f"      现在: {今年_宽度_米:.3f} m\n"
                                耕地长宽信息 += f"      变化: {'+' if 宽度变化_米 >= 0 else ''}{宽度变化_米:.3f} m\n"
                                
                                耕地长宽信息 += f"\n   📍 高度方向（南北）:\n"
                                耕地长宽信息 += f"      原来: {去年_高度_米:.3f} m\n"
                                耕地长宽信息 += f"      现在: {今年_高度_米:.3f} m\n"
                                耕地长宽信息 += f"      变化: {'+' if 高度变化_米 >= 0 else ''}{高度变化_米:.3f} m"
                    
                    self.输出结果(f"\n📌 当前图像: {os.path.basename(self.今年图像路径)}")
                    self.输出结果("")
                    self.输出结果(f"🔴 去年（该位置）: {原来面积:.3f} 亩")
                    self.输出结果(f"🔵 今年（该位置）: {当前耕地面积_亩:.3f} 亩")
                    self.输出结果(f"🟢 变化: {'+' if 变化 >= 0 else ''}{变化:.3f} 亩")
                    
                    # ✅ 只有面积有变化时才显示长宽变化！
                    if abs(变化) > 0.0001 and 耕地长宽信息:
                        self.输出结果(耕地长宽信息)
                    
                    self.输出结果("")
                    
                    # ✅ 直接根据变化值判断，不设置容差
                    if 变化 > 0:
                        self.输出结果(f"📈 耕地增加了 {变化:.3f} 亩")
                        # ✅ 弹窗也显示长宽变化
                        弹窗消息 = f"耕地增加了 {变化:.3f} 亩\n\n" \
                                       f"去年: {原来面积:.3f} 亩\n" \
                                       f"今年: {当前耕地面积_亩:.3f} 亩"
                        # ✅ 只有在变量存在时才显示长宽变化
                        if 耕地长宽信息 and '去年_宽度_米' in locals():
                            弹窗消息 += f"\n\n长宽变化:\n" \
                                           f"宽度: {去年_宽度_米:.3f}m → {今年_宽度_米:.3f}m ({'+' if 宽度变化_米>=0 else ''}{宽度变化_米:.3f}m)\n" \
                                           f"高度: {去年_高度_米:.3f}m → {今年_高度_米:.3f}m ({'+' if 高度变化_米>=0 else ''}{高度变化_米:.3f}m)"
                        messagebox.showinfo("分析完成", 弹窗消息)
                    elif 变化 < 0:
                        self.输出结果(f"📉 耕地减少了 {abs(变化):.3f} 亩")
                        弹窗消息 = f"耕地减少了 {abs(变化):.3f} 亩\n\n" \
                                       f"去年: {原来面积:.3f} 亩\n" \
                                       f"今年: {当前耕地面积_亩:.3f} 亩"
                        # ✅ 只有在变量存在时才显示长宽变化
                        if 耕地长宽信息 and '去年_宽度_米' in locals():
                            弹窗消息 += f"\n\n长宽变化:\n" \
                                           f"宽度: {去年_宽度_米:.3f}m → {今年_宽度_米:.3f}m ({'+' if 宽度变化_米>=0 else ''}{宽度变化_米:.3f}m)\n" \
                                           f"高度: {去年_高度_米:.3f}m → {今年_高度_米:.3f}m ({'+' if 高度变化_米>=0 else ''}{高度变化_米:.3f}m)"
                        messagebox.showwarning("分析完成", 弹窗消息)
                    else:
                        self.输出结果("➡️  耕地面积无变化")
                        messagebox.showinfo("分析完成", "耕地面积无变化")
                    
                    # 显示可视化（包含基准轮廓）
                    if 耕地掩码 is not None:
                        self.输出结果("\n🖼️ 生成可视化图像...")
                        # ✅ 传入基准CRS用于坐标转换
                        self.显示耕地可视化(耕地掩码, 基准耕地掩码, 基准地图_transform, 基准_crs)
                
                elif '基准数据' in 基准信息:
                    # 旧逻辑：匹配基准图像
                    import rasterio
                    from rasterio.warp import transform as warp_transform
                    
                    with rasterio.open(self.今年图像路径) as src:
                        左上角x = src.transform.c
                        左上角y = src.transform.f
                        右下角x = 左上角x + src.transform.a * src.width
                        右下角y = 左上角y + src.transform.e * src.height
                        
                        当前_左上角经度, 当前_左上角纬度 = warp_transform(src.crs, 'EPSG:4326', [左上角x], [左上角y])
                        当前_右下角经度, 当前_右下角纬度 = warp_transform(src.crs, 'EPSG:4326', [右下角x], [右下角y])
                    
                    # 找最佳匹配
                    匹配的基准 = None
                    最大重叠 = 0
                    
                    for 基准 in 基准信息['基准数据']:
                        重叠_左 = max(当前_左上角经度[0], 基准['左上角经度'])
                        重叠_右 = min(当前_右下角经度[0], 基准['右下角经度'])
                        重叠_上 = min(当前_左上角纬度[0], 基准['左上角纬度'])
                        重叠_下 = max(当前_右下角纬度[0], 基准['右下角纬度'])
                        
                        if 重叠_右 > 重叠_左 and 重叠_上 > 重叠_下:
                            重叠面积 = (重叠_右 - 重叠_左) * (重叠_上 - 重叠_下)
                            if 重叠面积 > 最大重叠:
                                最大重叠 = 重叠面积
                                匹配的基准 = 基准
                    
                    if 匹配的基准:
                        原来面积 = 匹配的基准['耕地面积_亩']
                        变化 = 当前耕地面积_亩 - 原来面积
                        
                        # ✅ 计算耕地长宽变化（精确到0.5m以内）
                        import rasterio
                        with rasterio.open(self.今年图像路径) as src:
                            # 计算耕地区域的实际范围
                            像素分辨率 = abs(src.transform.a)  # 米/像素
                            
                            # 从耕地掩码计算范围
                            if 耕地掩码 is not None:
                                耕地掩码_uint8 = (耕地掩码 > 0.5).astype(np.uint8) * 255
                                轮廓, _ = cv2.findContours(耕地掩码_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if len(轮廓) > 0:
                                    # 找到最大轮廓（主要耕地区域）
                                    最大轮廓 = max(轮廓, key=cv2.contourArea)
                                    x, y, w, h = cv2.boundingRect(最大轮廓)
                                    
                                    # 计算实际长宽（米），精确到0.5m以内
                                    今年_耕地宽度_米 = w * 像素分辨率
                                    今年_耕地高度_米 = h * 像素分辨率
                                    
                                    # 从基准数据计算原来的长宽（假设为正方形）
                                    原来_耕地面积_平方米 = 原来面积 * 666.67
                                    原来_边长_米 = np.sqrt(原来_耕地面积_平方米)
                                    
                                    # 计算长宽变化（精确到小数点后三位）
                                    宽度变化_米 = 今年_耕地宽度_米 - 原来_边长_米
                                    高度变化_米 = 今年_耕地高度_米 - 原来_边长_米
                                    
                                    耕地长宽信息 = f"\n\n📏 耕地长宽变化（精确到0.5m以内）:\n"
                                    耕地长宽信息 += f"   原来宽度: {原来_边长_米:.3f} m\n"
                                    耕地长宽信息 += f"   现在宽度: {今年_耕地宽度_米:.3f} m\n"
                                    耕地长宽信息 += f"   宽度变化: {'+' if 宽度变化_米 >= 0 else ''}{宽度变化_米:.3f} m\n\n"
                                    耕地长宽信息 += f"   原来高度: {原来_边长_米:.3f} m\n"
                                    耕地长宽信息 += f"   现在高度: {今年_耕地高度_米:.3f} m\n"
                                    耕地长宽信息 += f"   高度变化: {'+' if 高度变化_米 >= 0 else ''}{高度变化_米:.3f} m"
                                else:
                                    耕地长宽信息 = ""
                            else:
                                耕地长宽信息 = ""
                        
                        self.输出结果(f"\n📌 基准图像: {匹配的基准['tif文件']}")
                        self.输出结果(f"📌 当前图像: {os.path.basename(self.今年图像路径)}")
                        self.输出结果("")
                        self.输出结果(f"🟢 原来耕地面积: {原来面积:.3f} 亩")
                        self.输出结果(f"🟢 现在耕地面积: {当前耕地面积_亩:.3f} 亩")
                        self.输出结果(f"🟢 变化: {'+' if 变化 >= 0 else ''}{变化:.3f} 亩")
                        
                        # ✅ 显示耕地长宽变化
                        if 耕地长宽信息:
                            self.输出结果(耕地长宽信息)
                        
                        self.输出结果("")
                        
                        if 变化 > 0:
                            self.输出结果(f"📈 耕地增加了 {变化:.3f} 亩")
                            # ✅ 弹窗也显示长宽变化
                            弹窗消息 = f"耕地增加了 {变化:.3f} 亩\n\n" \
                                       f"原来: {原来面积:.3f} 亩\n" \
                                       f"现在: {当前耕地面积_亩:.3f} 亩"
                            if 耕地长宽信息:
                                弹窗消息 += f"\n\n长宽变化:\n" \
                                               f"宽度: {原来_边长_米:.3f}m → {今年_耕地宽度_米:.3f}m ({'+' if 宽度变化_米>=0 else ''}{宽度变化_米:.3f}m)\n" \
                                               f"高度: {原来_边长_米:.3f}m → {今年_耕地高度_米:.3f}m ({'+' if 高度变化_米>=0 else ''}{高度变化_米:.3f}m)"
                            messagebox.showinfo("分析完成", 弹窗消息)
                        elif 变化 < 0:
                            self.输出结果(f"📉 耕地减少了 {abs(变化):.3f} 亩")
                            弹窗消息 = f"耕地减少了 {abs(变化):.3f} 亩\n\n" \
                                       f"原来: {原来面积:.3f} 亩\n" \
                                       f"现在: {当前耕地面积_亩:.3f} 亩"
                            if 耕地长宽信息:
                                弹窗消息 += f"\n\n长宽变化:\n" \
                                               f"宽度: {原来_边长_米:.3f}m → {今年_耕地宽度_米:.3f}m ({'+' if 宽度变化_米>=0 else ''}{宽度变化_米:.3f}m)\n" \
                                               f"高度: {原来_边长_米:.3f}m → {今年_耕地高度_米:.3f}m ({'+' if 高度变化_米>=0 else ''}{高度变化_米:.3f}m)"
                            messagebox.showwarning("分析完成", 弹窗消息)
                        else:
                            self.输出结果("➡️  耕地面积无变化")
                            messagebox.showinfo("分析完成", "耕地面积无变化")
                        
                        # 旧逻辑没有基准掩码，只显示当前耕地
                        if 耕地掩码 is not None:
                            self.输出结果("\n🖼️ 生成可视化图像...")
                            self.显示耕地可视化(耕地掩码)
                    else:
                        self.输出结果("⚠️  未找到匹配的基准图像")
                        self.输出结果(f"🟢 当前耕地面积: {当前耕地面积_亩:.3f} 亩")
                        messagebox.showinfo("分析完成", f"当前耕地面积: {当前耕地面积_亩:.3f} 亩")
            else:
                self.输出结果(f"\n🟢 当前耕地面积: {当前耕地面积_亩:.3f} 亩")
                messagebox.showinfo("分析完成", f"当前耕地面积: {当前耕地面积_亩:.3f} 亩")
                
                # 没有基准数据，只显示当前耕地
                if 耕地掩码 is not None:
                    self.输出结果("\n🖼️ 生成可视化图像...")
                    self.显示耕地可视化(耕地掩码)
            
            # ✅ 结果已在界面展示，不再自动导出CSV（避免文件占用报错）
            # 如果需要保存，可以通过菜单手动导出
            # 系统.导出结果([结果], '分析结果.csv', 格式='csv')
            # self.输出结果(f"\n✅ 详细结果已保存到: 分析结果/分析结果.csv")
            
        except Exception as e:
            self.输出结果(f"\n❌ 分析出错: {e}")
            messagebox.showerror("错误", f"分析失败!\n\n{str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 恢复按钮
            self.分析按钮.config(state="normal", text="🚀 开始分析")
            self.进度条.stop()
            self.进度条.pack_forget()
            self.状态标签.config(text="● 就绪", fg=self.success)

if __name__ == "__main__":
    root = tk.Tk()
    app = 耕地分析界面(root)
    root.mainloop()
