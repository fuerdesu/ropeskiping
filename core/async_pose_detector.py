"""
异步姿态检测器 - 将姿态检测从主线程分离出来
"""

import numpy as np
import time
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from core.rtmpose_processor import RTMPoseProcessor


class FrameBuffer:
    """线程安全的帧缓冲池，复用内存减少GC压力"""
    
    def __init__(self, max_size=5, frame_shape=(360, 640, 3)):
        self.max_size = max_size
        self.frame_shape = frame_shape
        self.available_frames = deque()
        self.mutex = QMutex()
        
        # 预分配帧缓冲
        for _ in range(max_size):
            frame = np.zeros(frame_shape, dtype=np.uint8)
            self.available_frames.append(frame)
    
    def get_frame(self):
        """获取一个可用的帧缓冲，如果没有则创建新的"""
        self.mutex.lock()
        try:
            if self.available_frames:
                return self.available_frames.popleft()
            else:
                # 如果缓冲池空了，创建新的帧
                return np.zeros(self.frame_shape, dtype=np.uint8)
        finally:
            self.mutex.unlock()
    
    def return_frame(self, frame):
        """归还帧到缓冲池"""
        self.mutex.lock()
        try:
            if len(self.available_frames) < self.max_size:
                # 重置帧数据并归还到池中
                frame.fill(0)
                self.available_frames.append(frame)
        finally:
            self.mutex.unlock()


class AsyncPoseDetector(QThread):
    """异步姿态检测线程"""
    
    # 信号：发送检测结果 (frame_id, angle, keypoints, timestamp)
    pose_detected = pyqtSignal(int, object, object, float)
    
    def __init__(self, exercise_counter, model_mode='balanced'):
        super().__init__()
        
        # 初始化姿态检测器
        self.pose_processor = RTMPoseProcessor(exercise_counter, mode=model_mode)
        
        # 帧队列管理
        self.frame_queue = deque()
        self.max_queue_size = 3  # 限制队列大小防止内存堆积
        self.frame_buffer = FrameBuffer()
        
        # 线程控制
        self._running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = deque(maxlen=30)  # 保存最近30帧的处理时间
        
        # 当前运动类型
        self.exercise_type = "squat"
        
    def add_frame(self, frame, frame_id, exercise_type):
        """添加帧到处理队列"""
        self.mutex.lock()
        try:
            # 更新运动类型
            self.exercise_type = exercise_type
            
            # 如果队列已满，丢弃最旧的帧
            if len(self.frame_queue) >= self.max_queue_size:
                old_frame_data = self.frame_queue.popleft()
                # 归还旧帧到缓冲池
                self.frame_buffer.return_frame(old_frame_data['frame'])
            
            # 获取新的帧缓冲并复制数据
            buffer_frame = self.frame_buffer.get_frame()
            
            # 确保缓冲帧尺寸正确
            if buffer_frame.shape != frame.shape:
                buffer_frame = np.zeros(frame.shape, dtype=np.uint8)
            
            # 复制帧数据
            np.copyto(buffer_frame, frame)
            
            # 添加到队列
            frame_data = {
                'frame': buffer_frame,
                'frame_id': frame_id,
                'timestamp': time.time(),
                'exercise_type': exercise_type
            }
            self.frame_queue.append(frame_data)
            
            # 唤醒处理线程
            self.condition.wakeOne()
            
        finally:
            self.mutex.unlock()
    
    def update_model(self, model_mode):
        """更新模型模式"""
        self.mutex.lock()
        try:
            self.pose_processor.update_model(model_mode)
        finally:
            self.mutex.unlock()
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.processing_times:
            return {"avg_processing_time": 0, "fps": 0, "queue_size": 0}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_processing_time": avg_time * 1000,  # 转换为毫秒
            "fps": fps,
            "queue_size": len(self.frame_queue)
        }
    
    def run(self):
        """主处理循环"""
        print("AsyncPoseDetector started")
        
        while self._running:
            self.mutex.lock()
            
            # 等待有帧可处理
            while len(self.frame_queue) == 0 and self._running:
                self.condition.wait(self.mutex, 100)  # 100ms超时
            
            if not self._running:
                self.mutex.unlock()
                break
            
            # 获取最新的帧进行处理
            if self.frame_queue:
                frame_data = self.frame_queue.popleft()
            else:
                self.mutex.unlock()
                continue
            
            self.mutex.unlock()
            
            # 处理帧
            start_time = time.time()
            
            try:
                # 进行姿态检测
                results = self.pose_processor.process_frame(
                    frame_data['frame'], 
                    frame_data['exercise_type']
                )
                
                # 处理返回结果格式
                if results is None:
                    results = []
                elif isinstance(results, tuple) and len(results) == 2:
                    # 如果是元组格式 (all_angles, all_keypoints)，转换为列表格式
                    all_angles, all_keypoints = results
                    
                    # 调试信息：打印原始结果格式
                    print(f"RTMPose原始结果 - all_angles类型: {type(all_angles)}, 长度: {len(all_angles) if isinstance(all_angles, (list, tuple)) else 'N/A'}")
                    print(f"RTMPose原始结果 - all_keypoints类型: {type(all_keypoints)}, 长度: {len(all_keypoints) if isinstance(all_keypoints, (list, tuple)) else 'N/A'}")
                    
                    if isinstance(all_angles, (list, tuple)) and isinstance(all_keypoints, (list, tuple)):
                        # 多人检测结果：将角度和关键点配对
                        results = []
                        person_count = min(len(all_angles), len(all_keypoints))
                        print(f"检测到{person_count}个人，开始配对角度和关键点")
                        
                        for i in range(person_count):
                            results.append((all_angles[i], all_keypoints[i]))
                            print(f"第{i+1}个人配对完成 - 角度: {all_angles[i]}, 关键点数量: {len(all_keypoints[i]) if hasattr(all_keypoints[i], '__len__') else 'N/A'}")
                    elif isinstance(all_angles, (float, int)) and isinstance(all_keypoints, (list, tuple)):
                        # 单人检测结果
                        results = [(all_angles, all_keypoints)]
                        print(f"单人检测结果 - 角度: {all_angles}, 关键点数量: {len(all_keypoints)}")
                    else:
                        results = []
                        print("结果格式不匹配，设置为空列表")
                elif not isinstance(results, list):
                    results = []
                    print("结果不是列表格式，设置为空列表")
                
                print(f"最终返回结果格式: {type(results)}, 长度: {len(results)}")
                
                # 记录处理时间
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # 发送结果
                self.pose_detected.emit(
                    frame_data['frame_id'],
                    results,
                    None,
                    frame_data['timestamp']
                )
                
                self.frame_count += 1
                
            except Exception as e:
                print(f"Error in pose detection: {e}")
            
            finally:
                # 归还帧到缓冲池
                self.frame_buffer.return_frame(frame_data['frame'])
        
        print("AsyncPoseDetector stopped")
    
    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self._running = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
        # 等待线程结束
        self.wait(3000)  # 3秒超时
        
        # 清理队列
        self.mutex.lock()
        while self.frame_queue:
            frame_data = self.frame_queue.popleft()
            self.frame_buffer.return_frame(frame_data['frame'])
        self.mutex.unlock()


class PoseDetectionManager:
    """姿态检测管理器，简化接口"""
    
    def __init__(self, exercise_counter, model_mode='balanced'):
        self.detector = AsyncPoseDetector(exercise_counter, model_mode)
        self.frame_id_counter = 0
        self.latest_results = {}  # 存储最新的检测结果
        
        # 连接信号
        self.detector.pose_detected.connect(self._on_pose_detected)
        
        # 启动检测线程
        self.detector.start()
    
    def process_frame_async(self, frame, exercise_type):
        """异步处理帧"""
        self.frame_id_counter += 1
        self.detector.add_frame(frame, self.frame_id_counter, exercise_type)
        return self.frame_id_counter
    
    def get_latest_results(self):
        """获取最新的检测结果列表 - 支持多人"""
        if not self.latest_results:
            return []
        
        # 获取最新的结果ID
        latest_id = max(self.latest_results.keys())
        results = self.latest_results[latest_id]
        
        # 返回结果列表格式，支持多人扩展
        return results
    
    def _on_pose_detected(self, frame_id, results, _, timestamp):
        """处理检测结果"""
        # 保存最新结果
        self.latest_results[frame_id] = results
        
        # 只保留最近10个结果
        if len(self.latest_results) > 10:
            oldest_id = min(self.latest_results.keys())
            del self.latest_results[oldest_id]
    
    def update_model(self, model_mode):
        """更新模型"""
        self.detector.update_model(model_mode)
    
    def get_performance_stats(self):
        """获取性能统计"""
        return self.detector.get_performance_stats()
    
    def stop(self):
        """停止检测器"""
        self.detector.stop()