"""进度显示工具模块

该模块提供任务进度跟踪和显示功能，
包括进度条显示和耗时统计。

主要类：
ProgressTracker - 进度跟踪器，支持上下文管理

使用示例：
    with ProgressTracker("Processing") as tracker:
        for item in items:
            process(item)
            tracker.update(1)
    print(f"耗时: {tracker.elapsed:.2f}秒")
"""
from time import time
from tqdm import tqdm

class ProgressTracker:
    def __init__(self, desc: str = "Processing"):
        """初始化进度跟踪器
        
        参数:
            desc: 进度条描述文本 (类型: str)
            
        属性:
            desc: 进度描述文本
            start_time: 任务开始时间戳
            end_time: 任务结束时间戳
            pbar: tqdm进度条对象
        """
        self.desc = desc
        self.start_time = None
        self.end_time = None
        self.pbar = None
        
    def __enter__(self):
        """进入上下文时启动计时器和进度条
        
        返回:
            ProgressTracker: 自身实例
            
        注意:
            作为上下文管理器使用时自动调用
        """
        self.start_time = time()
        self.pbar = tqdm(desc=self.desc)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时停止计时器和进度条
        
        参数:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪
            
        注意:
            作为上下文管理器使用时自动调用
        """
        self.end_time = time()
        self.pbar.close()
        
    def update(self, n: int = 1):
        """更新进度条
        
        参数:
            n: 要更新的进度步数 (类型: int)
            
        示例:
            >>> tracker.update(1)  # 前进1步
            >>> tracker.update(5)  # 前进5步
        """
        self.pbar.update(n)
        
    @property
    def elapsed(self) -> float:
        """获取任务已用时间(秒)
        
        返回:
            float: 从开始到当前或结束的秒数
            
        示例:
            >>> print(f"耗时: {tracker.elapsed:.2f}秒")
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time or time()
        return end - self.start_time
