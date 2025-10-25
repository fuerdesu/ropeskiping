"""
主窗口类 - 负责基本的UI设置和信号连接
"""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QStatusBar, QMessageBox, QAction, QActionGroup, QMenu, QFileDialog)
from PyQt5.QtCore import Qt, QTimer

from core.video_thread import VideoThread
from core.rtmpose_processor import RTMPoseProcessor
from core.sound_manager import SoundManager
from core.workout_tracker import WorkoutTracker
from core.translations import Translations as T
from exercise_counters import ExerciseCounter
from ui.video_display import VideoDisplay
from ui.control_panel import ControlPanel
from ui.workout_stats_panel import WorkoutStatsPanel
from ui.styles import AppStyles

from .mode_manager import ModeManager
from .menu_manager import MenuManager
from .stats_manager import StatsManager
from .video_processor import VideoProcessor
from .counter_manager import CounterManager

class WorkoutTrackerApp(QMainWindow):
    """AI Fitness Assistant Main Window Class"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(T.get("app_title"))
        self.setMinimumSize(900, 900)
        
        # 初始化核心组件
        self._init_core_components()
        
        # 初始化管理器
        self._init_managers()
        
        # 创建UI
        self.setup_ui()
        
        # 初始化视频线程
        self.setup_video_thread()
        
        # 创建动画定时器
        self.setup_animation_timer()
        
        # 初始化面板
        self._init_panels()
        
        # 开始视频处理
        self.start_video()
        
        # 初始化状态变量
        self._init_state_variables()
        
        # 显示欢迎消息
        self.statusBar.showMessage(f"{T.get('welcome')} - RTMPose ({self.model_mode}) on {self.device}")
    
    def _init_core_components(self):
        """初始化核心组件"""
        # 设备设置
        self.device = 'cpu'
        self.model_mode = 'balanced'
        
        # 创建运动计数器
        self.exercise_counter = ExerciseCounter()
        
        # 初始化RTMPose姿态处理器
        print(f"Initializing RTMPose processor (mode: {self.model_mode}, device: {self.device})")
        self.pose_processor = RTMPoseProcessor(
            exercise_counter=self.exercise_counter,
            mode=self.model_mode,
            backend='onnxruntime',
            device=self.device
        )
        
        # 设置默认运动类型
        self.exercise_type = "overhead_press"
        
        # 创建声音管理器
        self.sound_manager = SoundManager()
        
        # 创建运动追踪器
        self.workout_tracker = WorkoutTracker()
        
    
    def _init_managers(self):
        """初始化管理器"""
        # 模式管理器
        self.mode_manager = ModeManager(self)
        
        # 菜单管理器
        self.menu_manager = MenuManager(self)
        
        
        # 统计管理器
        self.stats_manager = StatsManager(self)
        
        # 视频处理器
        self.video_processor = VideoProcessor(self)
        
        # 初始化异步姿态检测
        self.video_processor.init_pose_detection(self.exercise_counter, self.model_mode)
        
        # 计数器管理器
        self.counter_manager = CounterManager(self)
    
    def _init_panels(self):
        """初始化面板"""
        # 初始化运动统计面板
        self.stats_manager.init_workout_stats()
        
    
    def _init_state_variables(self):
        """初始化状态变量"""
        # 当前计数值
        self.current_count = 0
        
        # 手动计数追踪
        self.manual_count = 0
        
        # 重置操作标志
        self.is_resetting = False
        
        # 默认不显示运动统计面板
        self.stats_panel.setVisible(False)
        
        # 镜像模式相关属性
        self.mirror_mode = True
    
    def setup_ui(self):
        """设置用户界面"""
        # 应用样式
        self.setPalette(AppStyles.get_window_palette())
        self.setStyleSheet(AppStyles.get_global_stylesheet())
        
        # 创建主窗口布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧区域（视频和运动统计）
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加视频显示区域
        self.video_display = VideoDisplay()
        left_layout.addWidget(self.video_display)
        
        # 创建右侧区域（仅控制面板）
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加控制面板
        self.control_panel = ControlPanel()
        right_layout.addWidget(self.control_panel)
        
        # 添加拉伸以将控制面板推到顶部
        right_layout.addStretch()
        
        # 将左侧区域和右侧部件添加到主布局
        main_layout.addWidget(left_widget, 7)  # 为左侧区域分配70%空间
        main_layout.addWidget(right_widget, 3)  # 为右侧区域分配30%空间
        
        # 添加状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(T.get("ready"))
        
        # 设置菜单栏
        self.menu_manager.setup_menu_bar()
        
        # 连接控制面板信号
        self.connect_signals()
    
    def connect_signals(self):
        """连接信号和槽"""
        # 连接控制面板信号
        self.control_panel.exercise_changed.connect(self.change_exercise)
        self.control_panel.counter_reset.connect(self.reset_counter)
        self.control_panel.camera_changed.connect(self.change_camera)
        self.control_panel.rotation_toggled.connect(self.toggle_rotation)
        self.control_panel.skeleton_toggled.connect(self.toggle_skeleton)
        self.control_panel.model_changed.connect(self.change_model)
        self.control_panel.mirror_toggled.connect(self.toggle_mirror)
        
        # 连接新按钮信号
        self.control_panel.counter_increase.connect(self.increase_counter)
        self.control_panel.counter_decrease.connect(self.decrease_counter)
        self.control_panel.record_confirmed.connect(self.confirm_record)
        
        # 连接统计面板信号
        if hasattr(self, 'stats_panel'):
            self.stats_panel.goal_updated.connect(self.update_goal)
            self.stats_panel.weekly_goal_updated.connect(self.update_weekly_goal)
            self.stats_panel.month_changed.connect(self.load_month_stats)
    
    def setup_video_thread(self):
        """设置视频处理线程"""
        # 设置双分辨率：UI显示高分辨率，模型推理低分辨率
        self.video_thread = VideoThread(
            camera_id=0,
            rotate=True,
            display_width=1920,
            display_height=1080,
            inference_width=640,
            inference_height=360
        )
        
        # 设置主窗口引用，用于存储推理帧
        self.video_thread.main_window = self
        
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        
        # 初始化FPS值和推理帧
        self.current_fps = 0
        self.current_inference_frame = None
    
    def set_inference_frame(self, inference_frame):
        """设置推理帧并触发异步处理"""
        self.current_inference_frame = inference_frame
        # 发送到异步姿态检测器
        if hasattr(self.video_processor, 'process_inference_frame'):
            self.video_processor.process_inference_frame(inference_frame)
    
    def setup_animation_timer(self):
        """设置动画定时器"""
        self.count_animation_timer = QTimer()
        self.count_animation_timer.setSingleShot(True)
        self.count_animation_timer.timeout.connect(self.control_panel.reset_counter_style)
    
    def start_video(self):
        """开始视频处理"""
        self.video_thread.start()
    
    def update_image(self, frame, fps=0):
        """更新图像显示并处理姿态检测"""
        self.video_processor.update_image(frame, fps)
    
    def change_exercise(self, exercise_type):
        """更改运动类型"""
        self.counter_manager.change_exercise(exercise_type)
    
    def reset_counter(self):
        """重置计数器"""
        self.counter_manager.reset_counter()
    
    def reset_exercise_state(self):
        """重置运动状态"""
        self.counter_manager.reset_exercise_state()
    
    def increase_counter(self, new_count):
        """手动增加计数器值"""
        self.counter_manager.increase_counter(new_count)
    
    def decrease_counter(self, new_count):
        """手动减少计数器值"""
        self.counter_manager.decrease_counter(new_count)
    
    def confirm_record(self, exercise_type):
        """确认记录当前计数结果到历史记录"""
        self.counter_manager.confirm_record(exercise_type)
    
    def change_camera(self, index):
        """切换摄像头"""
        self.video_processor.change_camera(index)
    
    def toggle_rotation(self, rotate):
        """切换视频旋转模式"""
        self.video_processor.toggle_rotation(rotate)
    
    def toggle_skeleton(self, show):
        """切换骨架显示"""
        self.video_processor.toggle_skeleton(show)
    
    def toggle_mirror(self, mirror):
        """切换镜像模式"""
        self.video_processor.toggle_mirror(mirror)
    
    def open_video_file(self):
        """打开视频文件"""
        self.video_processor.open_video_file()
    
    def switch_to_camera_mode(self):
        """切换回摄像头模式"""
        self.video_processor.switch_to_camera_mode()
    
    def change_model(self, model_mode):
        """切换RTMPose模型模式"""
        # 更新模型模式
        self.model_mode = model_mode
        
        # 更新异步检测器的模型
        if hasattr(self.video_processor, 'update_model_mode'):
            self.video_processor.update_model_mode(model_mode)
        
        # 更新同步处理器的模型（如果仍在使用）
        self.video_processor.change_model(model_mode)
    
    def switch_to_workout_mode(self):
        """切换到运动模式"""
        self.mode_manager.switch_to_workout_mode()
    
    def switch_to_stats_mode(self):
        """切换到统计管理模式"""
        self.mode_manager.switch_to_stats_mode()
    
    def switch_to_voice_control_mode(self):
        """切换到语音控制模式"""
        self.mode_manager.switch_to_voice_control_mode()
    
    def show_about(self):
        """显示关于对话框"""
        self.menu_manager.show_about()
    
    def change_language(self, language):
        """更改界面语言"""
        self.menu_manager.change_language(language)
    
    def update_today_stats(self):
        """更新今日运动统计"""
        self.stats_manager.update_today_stats()
    
    def update_stats_overview(self):
        """更新所有统计概览"""
        self.stats_manager.update_stats_overview()
    
    def load_month_stats(self, year, month):
        """加载指定月份的统计数据"""
        self.stats_manager.load_month_stats(year, month)
    
    def update_goal(self, exercise_type, count):
        """更新运动目标"""
        self.stats_manager.update_goal(exercise_type, count)
    
    def update_weekly_goal(self, count):
        """更新周目标"""
        self.stats_manager.update_weekly_goal(count)
    
    def export_csv_results(self):
        """导出CSV结果文件"""
        self.video_processor.export_csv_results()
    
    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        # 停止视频线程
        if self.video_thread.isRunning():
            self.video_thread.stop()
        
        # 清理异步姿态检测器
        if hasattr(self.video_processor, 'cleanup'):
            self.video_processor.cleanup()
        
        event.accept()