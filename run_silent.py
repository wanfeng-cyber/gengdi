import sys
import os
import subprocess
import ctypes
from ctypes import wintypes

# 隐藏控制台窗口
def hide_console():
    """
    隐藏控制台窗口
    """
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)  # 0 = SW_HIDE
        ctypes.windll.user32.ShowWindow(whnd, 6)  # 6 = SW_MINIMIZE

def run_silent():
    """
    静默运行耕地分析系统
    """
    # 隐藏当前控制台窗口
    hide_console()

    # 要运行的脚本
    script = "耕地分析工具_图形界面.py"

    # 创建新进程但不显示窗口
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE

    # 运行程序
    subprocess.Popen([sys.executable, script],
                    startupinfo=startupinfo)

if __name__ == "__main__":
    run_silent()