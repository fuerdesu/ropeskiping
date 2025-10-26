"""
视频处理模块 - 负责图像更新和异步姿态检测
"""

import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer
from core.translations import Translations as T
from core.async_pose_detector import PoseDetectionManager
from app.csv_output_manager import CSVOutputManager

class VideoProcessor:
    """视频处理器类 - 集成异步姿态检测"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.pose_detection_manager = None
        # 改为列表存储多人数据
        self.latest_keypoints_list = []  # 存储多人的关键点
        self.latest_angles_list = []     # 存储多人的角度
        self.person_count = 0           # 检测到的人数

        self.previous_centers = {}   # 上一帧的中心点 {person_id: (x, y)}
        self.next_person_id = 1      # 分配新person_id用

        # 初始化异步姿态检测管理器
        self.pose_detection_manager = None
        self.latest_keypoints = None
        self.latest_angle = None
        
        # 初始化CSV输出管理器
        self.csv_manager = CSVOutputManager()
        
        # 性能监控
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._log_performance)
        self.performance_timer.start(5000)  # 每5秒记录一次性能
    
    def init_pose_detection(self, exercise_counter, model_mode='balanced'):
        """初始化异步姿态检测"""
        if self.pose_detection_manager:
            self.pose_detection_manager.stop()
        
        self.pose_detection_manager = PoseDetectionManager(exercise_counter, model_mode)
    
    def cleanup(self):
        """清理资源"""
        if self.pose_detection_manager:
            self.pose_detection_manager.stop()
        if self.performance_timer:
            self.performance_timer.stop()
    
    def update_image(self, frame, fps=0):
        """更新图像显示 - 异步姿态检测版本"""
        try:
            # 更新FPS值
            self.main_window.current_fps = fps
            
            # 准备显示帧
            display_frame = frame.copy()
            
            # 使用最新的异步检测结果
            current_angle = self.latest_angle
            keypoints = self.latest_keypoints
            
            # 如果有关键点信息且启用骨架显示，在高分辨率帧上绘制骨架
            if keypoints is not None and hasattr(self.main_window.pose_processor, 'show_skeleton') and self.main_window.pose_processor.show_skeleton:
                # 在显示帧上绘制骨架（关键点已经是显示帧坐标）
                display_frame = self.draw_skeleton_on_frame(display_frame, keypoints)
            
            # 如果启用镜像模式，应用镜像处理
            if self.main_window.mirror_mode:
                display_frame = cv2.flip(display_frame, 1)
            
            # 转换BGR到RGB（Qt需要RGB格式）
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # 更新视频显示
            self.main_window.video_display.update_image(display_frame)
            
            # 更新UI组件
            self.update_ui_components(current_angle, keypoints)
            
        except Exception as e:
            print(f"Error updating image: {e}")
    
    def process_inference_frame(self, inference_frame):
        """处理推理帧进行异步姿态检测"""
        if not self.pose_detection_manager:
            return
        
        # 异步提交帧进行处理
        self.pose_detection_manager.process_frame_async(
            inference_frame, 
            self.main_window.exercise_type
        )
        
        # 获取最新的检测结果
        results = self.pose_detection_manager.get_latest_results()
        
        # 清空旧数据
        self.latest_angles_list = []
        self.latest_keypoints_list = []
        
        if not results or len(results) == 0:
            self.person_count = 0
            return
        
        self.person_count = len(results)
        # 先提取关键点以便位置匹配
        raw_keypoints_list = [res[1] if res else None for res in results]
        # 获取图像宽度用于动态阈值计算
        frame_width = inference_frame.shape[1]
        id_map = self.match_person_ids(raw_keypoints_list, frame_width)
        # 遍历所有检测到的人
        for i, result in enumerate(results):
            if not result:
                self.latest_angles_list.append(None)
                self.latest_keypoints_list.append(None)
                continue
                
            try:
                angle, keypoints = result
                
                # 处理关键点缩放
                if keypoints is not None:
                    if hasattr(self.main_window, 'video_display') and hasattr(self.main_window.video_display, 'current_frame_size'):
                        display_size = self.main_window.video_display.current_frame_size
                        if display_size:
                            scale_x = display_size[0] / inference_frame.shape[1]
                            scale_y = display_size[1] / inference_frame.shape[0]
                            
                            scaled_keypoints = keypoints.copy()
                            scaled_keypoints[:, 0] *= scale_x
                            scaled_keypoints[:, 1] *= scale_y
                            keypoints = scaled_keypoints  # 更新关键点
                            self.latest_keypoints_list.append(scaled_keypoints)
                        else:
                            self.latest_keypoints_list.append(keypoints)
                    else:
                        self.latest_keypoints_list.append(keypoints)
                    
                    # ⬇️ 调用 CounterManager 来处理运动逻辑（支持跳绳）
                    person_id = id_map.get(i, i + 1)
                    self.main_window.counter_manager.process_exercise_frame(keypoints, person_id=person_id)

                else:
                    self.latest_keypoints_list.append(None)
                
                # 保存角度
                self.latest_angles_list.append(angle)
                
            except Exception as e:
                print(f"Error processing person {i}: {e}")
                self.latest_angles_list.append(None)
                self.latest_keypoints_list.append(None)
        
        # 为兼容旧逻辑（单人显示）
        if len(self.latest_keypoints_list) > 0:
            self.latest_keypoints = self.latest_keypoints_list[0]
            self.latest_angle = self.latest_angles_list[0]
            
        self.id_map = id_map  # 保存稳定ID映射，供UI绘制使用

    def _log_performance(self):
        """记录性能统计"""
        if self.pose_detection_manager:
            stats = self.pose_detection_manager.get_performance_stats()
            print(f"Pose Detection Performance: "
                  f"Avg Time: {stats['avg_processing_time']:.1f}ms, "
                  f"FPS: {stats['fps']:.1f}, "
                  f"Queue: {stats['queue_size']}")

    def match_person_ids(self, detected_keypoints_list, frame_width=640):
        """
        根据关键点位置匹配跨帧ID，确保person_id稳定
        
        Args:
            detected_keypoints_list: 检测到的关键点列表
            frame_width: 图像宽度，用于动态阈值计算
        """
        matched = {}  # {new_index: stable_id}
        new_centers = {}
        
        # 动态阈值设置：基于图像尺寸的比例
        distance_threshold = frame_width * 0.15  # 15%的图像宽度作为阈值
        
        # 跟踪ID的活跃状态
        active_ids = set()

        for new_i, keypoints in enumerate(detected_keypoints_list):
            if keypoints is None or len(keypoints) < 17:
                continue

            # 用双脚踝或髋关节中心估计人体位置
            try:
                left_ankle = keypoints[15][:2]
                right_ankle = keypoints[16][:2]
                center = np.mean([left_ankle, right_ankle], axis=0)
            except Exception:
                center = np.mean(keypoints[:, :2], axis=0)

            best_id, best_dist = None, float('inf')
            for pid, prev_center in self.previous_centers.items():
                dist = np.linalg.norm(center - prev_center)
                if dist < distance_threshold and dist < best_dist:
                    best_id, best_dist = pid, dist

            if best_id is not None:
                matched[new_i] = best_id
                active_ids.add(best_id)
            else:
                # 检查是否可以重用最近不活跃的ID（改进的重用逻辑）
                reused_id = None
                min_reuse_dist = float('inf')
                
                for pid in self.previous_centers.keys():
                    if pid not in active_ids:
                        # 检查距离是否在可接受范围内
                        dist = np.linalg.norm(center - self.previous_centers[pid])
                        if dist < distance_threshold * 1.5 and dist < min_reuse_dist:
                            reused_id = pid
                            min_reuse_dist = dist
                
                if reused_id is not None:
                    matched[new_i] = reused_id
                    active_ids.add(reused_id)
                else:
                    # 分配新ID，确保不重复
                    while self.next_person_id in active_ids:
                        self.next_person_id += 1
                    matched[new_i] = self.next_person_id
                    active_ids.add(self.next_person_id)
                    self.next_person_id += 1

            new_centers[matched[new_i]] = center

        # 清理长时间不活跃的ID
        max_id_age = 30  # 最大ID保留帧数
        if hasattr(self, 'id_age_counter'):
            # 更新ID年龄计数器
            for pid in list(self.id_age_counter.keys()):
                if pid in active_ids:
                    self.id_age_counter[pid] = 0  # 重置活跃ID的年龄
                else:
                    self.id_age_counter[pid] += 1
                    # 移除过老的ID
                    if self.id_age_counter[pid] > max_id_age:
                        if pid in self.previous_centers:
                            del self.previous_centers[pid]
                        del self.id_age_counter[pid]
            
            # 添加新ID到年龄计数器
            for pid in active_ids:
                if pid not in self.id_age_counter:
                    self.id_age_counter[pid] = 0
        else:
            # 初始化ID年龄计数器
            self.id_age_counter = {}
            for pid in active_ids:
                self.id_age_counter[pid] = 0

        self.previous_centers = new_centers
        return matched
    
    def update_model_mode(self, model_mode):
        """更新模型模式"""
        if self.pose_detection_manager:
            self.pose_detection_manager.update_model(model_mode)
    
    def draw_skeleton_on_frame(self, frame, keypoints):
        """在高分辨率帧上绘制骨架"""
        try:
            # 定义连接关系 (COCO 17 keypoint format)
            connections = [
                # Head and face
                [0, 1], [0, 2], [1, 3], [2, 4],  # nose-eyes-ears
                # Torso
                [5, 6],   # left_shoulder-right_shoulder
                [5, 11],  # left_shoulder-left_hip
                [6, 12],  # right_shoulder-right_hip
                [11, 12], # left_hip-right_hip
                # Arms
                [5, 7], [7, 9],    # left_shoulder-left_elbow-left_wrist
                [6, 8], [8, 10],   # right_shoulder-right_elbow-right_wrist
                # Legs
                [11, 13], [13, 15], # left_hip-left_knee-left_ankle
                [12, 14], [14, 16]  # right_hip-right_knee-right_ankle
            ]
            
            # 定义颜色 (BGR format)
            colors = {
                'head': (51, 153, 255),    # Blue
                'torso': (255, 153, 51),   # Orange
                'arms': (153, 255, 51),    # Green
                'legs': (255, 51, 153)     # Pink
            }
            
            # 绘制连接线
            for connection in connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    
                    # 跳过无效点
                    if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                        continue
                    
                    # 选择颜色
                    if pt1_idx in [0, 1, 2, 3, 4]:  # Head
                        color = colors['head']
                    elif pt1_idx in [5, 6, 11, 12]:  # Torso
                        color = colors['torso']
                    elif pt1_idx in [7, 8, 9, 10]:  # Arms
                        color = colors['arms']
                    else:  # Legs
                        color = colors['legs']
                    
                    # 绘制连接线 - 根据分辨率调整线条粗细
                    line_thickness = max(2, int(frame.shape[1] / 640 * 3))
                    cv2.line(frame, 
                            (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), 
                            color, line_thickness)
            
            # 绘制关键点
            for i, point in enumerate(keypoints):
                if point[0] == 0 and point[1] == 0:  # 跳过无效点
                    continue
                
                # 选择颜色
                if i in [0, 1, 2, 3, 4]:  # Head
                    color = colors['head']
                elif i in [5, 6, 11, 12]:  # Torso
                    color = colors['torso']
                elif i in [7, 8, 9, 10]:  # Arms
                    color = colors['arms']
                else:  # Legs
                    color = colors['legs']
                
                # 根据分辨率调整关键点大小
                point_radius = max(3, int(frame.shape[1] / 640 * 5))
                cv2.circle(frame, (int(point[0]), int(point[1])), point_radius, color, -1)
                cv2.circle(frame, (int(point[0]), int(point[1])), point_radius + 2, color, 2)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing skeleton: {e}")
            return frame
    
    def update_image(self, frame, fps=0):
        """更新图像显示 - 支持多人姿态显示"""
        try:
            self.main_window.current_fps = fps
            display_frame = frame.copy()

            # 检查多人关键点
            if len(self.latest_keypoints_list) > 0 and hasattr(self.main_window.pose_processor, 'show_skeleton') and self.main_window.pose_processor.show_skeleton:
                for i, keypoints in enumerate(self.latest_keypoints_list):
                    if keypoints is not None:
                        # 为每个人绘制骨架并添加ID标签
                        display_frame = self.draw_skeleton_on_frame(display_frame, keypoints)
                        # 在骨架上方添加人员ID
                        if len(keypoints) > 0 and keypoints[0][0] > 0 and keypoints[0][1] > 0:
                            stable_id = getattr(self, "id_map", None)
                            pid = stable_id.get(i, i + 1) if stable_id else i + 1
                            cv2.putText(display_frame, f"Person {pid}",

                                       (int(keypoints[0][0]) - 20, int(keypoints[0][1]) - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 镜像模式
            if self.main_window.mirror_mode:
                display_frame = cv2.flip(display_frame, 1)

            # 转换颜色 BGR->RGB
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.main_window.video_display.update_image(display_frame)

            # 多人UI更新：显示所有检测到的人的角度和计数信息
            if len(self.latest_angles_list) > 0:
                # 更新UI显示多人信息
                self.update_ui_components_multi_person(self.latest_angles_list, self.latest_keypoints_list)
                
                # 跳绳模式下显示每个人的计数 - 这部分逻辑已经移到update_ui_components_multi_person方法中
                # 避免重复调用update_multi_person_info方法
            else:
                # 无人检测时清空UI
                self.update_ui_components(None, None)

        except Exception as e:
            print(f"Error updating image: {e}")
            
    def update_ui_components(self, current_angle, keypoints):
        """更新UI组件显示"""
        try:
            # 更新角度显示 - 注释掉这部分代码
            # if current_angle is not None:
            #     self.main_window.control_panel.update_angle(str(int(current_angle)), self.main_window.exercise_type)
            
            # 更新阶段显示（上/下）
            if hasattr(self.main_window.exercise_counter, 'stage'):
                self.main_window.control_panel.update_phase(self.main_window.exercise_counter.stage)
            
            # 获取当前计数 - 使用总计数（所有人员的计数之和）
            current_count = self.main_window.exercise_counter.get_total_count()
            
            # 如果计数增加且不是重置操作，播放声音（但不自动记录）
            if current_count > self.main_window.current_count and not self.main_window.is_resetting:
                # 为每次计数增加播放计数声音
                self.main_window.sound_manager.play_count_sound()
                
                # 每10次计数播放成功声音
                if current_count % 10 == 0:
                    self.main_window.sound_manager.play_milestone_sound(current_count)
                    self.main_window.statusBar.showMessage(f"Congratulations on completing {current_count} {self.main_window.control_panel.exercise_display_map[self.main_window.exercise_type]}!")
                
                # 更新缓存的当前计数
                self.main_window.current_count = current_count
            
            # 更新计数器显示
            self.main_window.control_panel.update_counter(str(current_count))
        except Exception as e:
            print(f"Error updating image: {str(e)}")
    
    def update_ui_components_multi_person(self, angles_list, keypoints_list):
        """更新UI组件显示 - 多人版本"""
        try:
            # 检查角度列表是否为空或无效
            if not angles_list or all(angle is None for angle in angles_list):
                # 无人检测时清空UI
                if hasattr(self.main_window.control_panel, 'update_multi_person_info'):
                    self.main_window.control_panel.update_multi_person_info([])
                return
            
            # 更新状态栏显示检测到的人数
            person_count = len(angles_list)
            self.main_window.statusBar.showMessage(f"Detected {person_count} person(s)")
            
            # 更新阶段显示（上/下）- 使用第一个人的阶段信息
            if hasattr(self.main_window.exercise_counter, 'stage'):
                self.main_window.control_panel.update_phase(self.main_window.exercise_counter.stage)
            
            # 获取当前计数 - 使用总计数方法
            current_count = self.main_window.exercise_counter.get_total_count()
            
            # 如果计数增加且不是重置操作，播放声音（但不自动记录）
            if current_count > self.main_window.current_count and not self.main_window.is_resetting:
                # 为每次计数增加播放计数声音
                self.main_window.sound_manager.play_count_sound()
                
                # 每10次计数播放成功声音
                if current_count % 10 == 0:
                    self.main_window.sound_manager.play_milestone_sound(current_count)
                    self.main_window.statusBar.showMessage(f"Congratulations on completing {current_count} {self.main_window.control_panel.exercise_display_map[self.main_window.exercise_type]}!")
                
                # 更新缓存的当前计数
                self.main_window.current_count = current_count
            
            # 更新计数器显示
            self.main_window.control_panel.update_counter(str(current_count))
            
            # 如果有控制面板的多人显示功能，更新多人角度信息
            if hasattr(self.main_window.control_panel, 'update_multi_person_info'):
                # 准备多人角度信息和计数
                multi_person_info = []
                violations_list = []
                
                for i, angle in enumerate(angles_list):
                    if angle is not None:
                        # 获取该人员的计数
                        count = self.main_window.exercise_counter.counters.get(i + 1, 0)
                        
                        # 获取违规信息（跳绳模式专用）
                        violations = []
                        if self.main_window.exercise_type == 'jump_rope' and hasattr(self.main_window.exercise_counter, 'violations'):
                            violations = self.main_window.exercise_counter.violations.get(i + 1, [])
                        
                        # 跳绳模式特殊处理：使用高度信息代替角度
                        if self.main_window.exercise_type == 'jump_rope':
                            # 跳绳模式显示跳跃高度信息
                            if hasattr(self.main_window.exercise_counter, 'jump_rope_states'):
                                state = self.main_window.exercise_counter.jump_rope_states.get(i + 1, {})
                                jump_height = state.get('last_jump_height', 0)
                                stage = state.get('stage', 'unknown')
                                
                                # 将高度转换为可读格式
                                height_display = f"{jump_height:.3f}" if jump_height > 0 else "0.000"
                                multi_person_info.append({
                                    'person_id': i + 1,
                                    'angle': f"{stage} ({height_display})",
                                    'count': count,
                                    'exercise_type': self.main_window.exercise_type,
                                    'violations': violations
                                })
                            else:
                                multi_person_info.append({
                                    'person_id': i + 1,
                                    'angle': "检测中...",
                                    'count': count,
                                    'exercise_type': self.main_window.exercise_type,
                                    'violations': violations
                                })
                        else:
                            # 其他运动模式使用角度
                            multi_person_info.append({
                                'person_id': i + 1,
                                'angle': int(angle),
                                'count': count,
                                'exercise_type': self.main_window.exercise_type,
                                'violations': violations
                            })
                        
                        violations_list.append(violations)
                
                # 更新控制面板的多人信息显示
                self.main_window.control_panel.update_multi_person_info(multi_person_info)
                
                # 跳绳模式下，更新CSV输出
                if self.main_window.exercise_type == 'jump_rope':
                    # 准备CSV数据
                    csv_data = []
                    for i, person_info in enumerate(multi_person_info):
                        if keypoints_list[i] is not None:
                            # 计算人员点位（使用脚踝关键点）
                            left_ankle = keypoints_list[i][15] if len(keypoints_list[i]) > 15 else [0, 0]
                            right_ankle = keypoints_list[i][16] if len(keypoints_list[i]) > 16 else [0, 0]
                            
                            # 计算中心点
                            center_x = (left_ankle[0] + right_ankle[0]) / 2
                            center_y = (left_ankle[1] + right_ankle[1]) / 2
                            
                            csv_data.append({
                                'person_id': person_info['person_id'],
                                'position_x': round(center_x, 2),
                                'position_y': round(center_y, 2),
                                'jump_count': person_info['count'],
                                'violations': violations_list[i]
                            })
                    
                    # 更新CSV输出管理器
                    if csv_data:
                        self.csv_manager.update_jump_rope_data(csv_data)
                
        except Exception as e:
            print(f"Error updating multi-person UI: {str(e)}")
    
    def change_camera(self, index):
        """切换摄像头"""
        self.main_window.video_thread.set_camera(index)
        self.main_window.statusBar.showMessage(f"Switched to camera {index}")
    
    def toggle_rotation(self, rotate):
        """切换视频旋转模式"""
        # 更新视频线程旋转设置
        self.main_window.video_thread.set_rotation(rotate)
        
        # 更新视频显示方向设置
        # rotate=True表示竖屏，False表示横屏
        self.main_window.video_display.set_orientation(portrait_mode=rotate)
        
        if rotate:
            self.main_window.toggle_rotation_action.setText("Turn off rotation mode")
            self.main_window.statusBar.showMessage("Switched to portrait mode (9:16)")
        else:
            self.main_window.toggle_rotation_action.setText("Turn on rotation mode")
            self.main_window.statusBar.showMessage("Switched to landscape mode (16:9)")
    
    def toggle_skeleton(self, show):
        """切换骨架显示"""
        self.main_window.pose_processor.set_skeleton_visibility(show)
        if show:
            self.main_window.statusBar.showMessage("Show skeleton lines")
        else:
            self.main_window.statusBar.showMessage("Hide skeleton lines")
    
    def toggle_mirror(self, mirror):
        """切换镜像模式"""
        self.main_window.mirror_mode = mirror
        self.main_window.statusBar.showMessage(f"Mirror mode: {'ON' if mirror else 'OFF'}")
        
        # 更新菜单动作状态
        if hasattr(self.main_window, 'toggle_mirror_action'):
            self.main_window.toggle_mirror_action.setChecked(mirror)
    
    def open_video_file(self):
        """打开视频文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self.main_window,
            T.get("open_video"),
            "",
            T.get("video_files"),
            options=options
        )
        
        if file_name:
            try:
                # 清除当前计数状态
                self.main_window.reset_exercise_state()
                
                # 切换到运动模式（如果当前不是）
                if hasattr(self.main_window, 'stacked_layout') and hasattr(self.main_window, 'exercise_container'):
                    if not self.main_window.stacked_layout.currentWidget() == self.main_window.exercise_container:
                        self.main_window.switch_to_workout_mode()
                
                # 设置状态栏信息
                video_name = os.path.basename(file_name)
                self.main_window.statusBar.showMessage(f"Current video: {video_name}")
                
                # 将文件路径传递给视频线程，设置为非循环播放模式
                self.main_window.video_thread.set_video_file(file_name, loop=False)
            except Exception as e:
                print(f"Error opening video file: {e}")
                self.main_window.statusBar.showMessage(f"Failed to open video file: {str(e)}")
    
    def switch_to_camera_mode(self):
        """切换回摄像头模式"""
        try:
            # 清除当前计数状态
            self.main_window.reset_exercise_state()
            
            # 切换到运动模式（如果当前不是）
            if hasattr(self.main_window, 'stacked_layout') and hasattr(self.main_window, 'exercise_container'):
                if not self.main_window.stacked_layout.currentWidget() == self.main_window.exercise_container:
                    self.main_window.switch_to_workout_mode()
                
            # 设置状态栏信息
            self.main_window.statusBar.showMessage("Current mode: Camera")
            
            # 返回摄像头模式
            self.main_window.video_thread.set_camera(0)  # 使用默认摄像头
        except Exception as e:
            print(f"Error switching to camera mode: {e}")
            self.main_window.statusBar.showMessage(f"Failed to switch to camera mode: {str(e)}")
    
    def change_model(self, model_mode):
        """切换RTMPose模型模式"""
        try:
            if model_mode == self.main_window.model_mode:
                # 如果是相同模式，无需重新加载
                return
                
            # 停止视频处理
            self.main_window.video_thread.stop()
            
            # 显示状态信息
            self.main_window.statusBar.showMessage(f"Switching RTMPose mode to: {model_mode}...")
            
            # 更新模型模式
            old_model_mode = self.main_window.model_mode
            self.main_window.model_mode = model_mode
            
            print(f"Switching RTMPose mode: {old_model_mode} -> {model_mode}")
            
            # 更新RTMPose处理器模式
            self.main_window.pose_processor.update_model(model_mode)
            
            # 重新初始化视频线程
            self.main_window.setup_video_thread()
            
            # 重新启动视频处理
            QTimer.singleShot(500, self.main_window.start_video)  # 延迟500ms后开始视频
            
            # 更新状态栏
            self.main_window.statusBar.showMessage(f"Switched to RTMPose {model_mode} mode")
            
        except Exception as e:
            # 如果切换失败，显示错误消息
            error_msg = f"RTMPose mode switching failed: {str(e)}"
            self.main_window.statusBar.showMessage(error_msg)
            print(error_msg)
            
            # 尝试回滚到原始模式
            try:
                self.main_window.model_mode = old_model_mode
                self.main_window.pose_processor.update_model(old_model_mode)
                self.main_window.setup_video_thread()
                QTimer.singleShot(500, self.main_window.start_video)
                self.main_window.statusBar.showMessage(f"Rolled back to RTMPose {old_model_mode} mode")
                
            except:
                # 如果回滚也失败，显示严重错误
                self.main_window.statusBar.showMessage("Critical error in RTMPose mode switching")
    
    def export_csv_results(self):
        """导出CSV结果文件"""
        try:
            # 导出CSV文件
            success = self.csv_manager.export_csv()
            if success:
                self.main_window.statusBar.showMessage("CSV results exported successfully to /output/result.csv")
                return True
            else:
                self.main_window.statusBar.showMessage("Failed to export CSV results")
                return False
        except Exception as e:
            print(f"Error exporting CSV results: {str(e)}")
            self.main_window.statusBar.showMessage(f"Error exporting CSV: {str(e)}")
            return False