"""实时摄像头风格迁移处理器

功能：
- 从摄像头捕获实时画面
- 应用风格迁移效果
- 实时显示处理结果
- 支持性能统计显示
"""
import cv2
import time
from threading import Thread
from queue import Queue
from core.style_applier import StyleApplier
from utils.progress import ProgressTracker

class CameraProcessor:
    def __init__(self, style_path, frame_size=(640, 480)):
        """初始化摄像头处理器
        
        参数:
            style_path: 风格图片路径
            frame_size: 处理帧尺寸(宽,高)
        """
        self.style_path = style_path
        self.frame_size = frame_size
        self.running = False
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.fps = 0
        self.style_applier = StyleApplier(style_path)
        
    def start_camera(self):
        """启动摄像头捕获线程"""
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames)
        self.capture_thread.start()
        
    def stop_camera(self):
        """停止摄像头捕获"""
        self.running = False
        self.capture_thread.join()
        
    def _capture_frames(self):
        """摄像头帧捕获线程"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        
        last_time = time.time()
        frame_count = 0
        
        with ProgressTracker("摄像头捕获") as tracker:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # 计算FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    now = time.time()
                    self.fps = 10 / (now - last_time)
                    last_time = now
                
                # 放入处理队列
                if self.frame_queue.empty():
                    self.frame_queue.put(frame)
                    
                tracker.update(1)
                
        cap.release()
        
    def process_frames(self):
        """风格迁移处理线程"""
        with ProgressTracker("风格迁移") as tracker:
            while self.running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # 应用风格迁移
                    styled_frame = self.style_applier.apply_frame(frame)
                    
                    # 显示结果
                    if self.result_queue.empty():
                        self.result_queue.put(styled_frame)
                        
                    tracker.update(1)
                    
    def display_results(self):
        """结果显示线程"""
        cv2.namedWindow("Style Transfer", cv2.WINDOW_NORMAL)
        
        while self.running:
            if not self.result_queue.empty():
                frame = self.result_queue.get()
                
                # 显示FPS
                cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                cv2.imshow("Style Transfer", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True, help="风格图片路径")
    parser.add_argument("--size", type=int, default=640, 
                       help="处理帧宽度(默认640)")
    args = parser.parse_args()
    
    processor = CameraProcessor(args.style, (args.size, int(args.size*0.75)))
    processor.start_camera()
    
    # 启动处理线程
    process_thread = Thread(target=processor.process_frames)
    process_thread.start()
    
    # 启动显示线程
    display_thread = Thread(target=processor.display_results)
    display_thread.start()
    
    try:
        while processor.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        processor.running = False
        
    process_thread.join()
    display_thread.join()
    processor.stop_camera()

if __name__ == "__main__":
    main()
