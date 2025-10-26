import os
import cv2
import sys
import numpy as np
from rtmlib import Wholebody, draw_skeleton

class RTMPoseProcessor:
    """RTMPose pose detection processor"""
    
    def __init__(self, exercise_counter, mode='balanced', backend='onnxruntime', device='cpu'):
        self.exercise_counter = exercise_counter
        self.show_skeleton = True
        self.conf_threshold = 0.5
        self.device = device
        self.backend = backend
        
        # Initialize RTMPose model
        self.init_rtmpose(mode)
        
        self.keypoint_mapping = self.get_keypoint_mapping()
    
    def get_models_dir(self):
        """Get model file directory, compatible with development and packaged environments"""
        if getattr(sys, 'frozen', False):
            # Packaged environment, model files are in temp directory
            base_path = sys._MEIPASS
            models_dir = os.path.join(base_path, 'models')
        else:
            # Development environment, model files are in project directory
            models_dir = './models'
        
        return models_dir
    
    def init_rtmpose(self, mode='balanced'):
        """Initialize RTMPose model"""
        try:
            print(f"Initializing RTMPose model (mode: {mode}, backend: {self.backend}, device: {self.device})")
            
            # Check if local model files exist
            models_dir = self.get_models_dir()
            if os.path.exists(models_dir):
                # Try to use local models
                det_model = os.path.join(models_dir, 'yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx')
                
                # Select different pose detection models based on mode
                if mode == 'lightweight':
                    pose_model = os.path.join(models_dir, 'rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx')
                    pose_input_size = (192, 256)
                elif mode == 'performance':
                    pose_model = os.path.join(models_dir, 'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx')
                    pose_input_size = (192, 256)
                else:  # balanced
                    pose_model = os.path.join(models_dir, 'rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx')
                    pose_input_size = (192, 256)
                
                if os.path.exists(det_model) and os.path.exists(pose_model):
                    print(f"Using local model files ({mode} mode)")
                    self.wholebody = Wholebody(
                        det=det_model,
                        det_input_size=(416, 416),
                        pose=pose_model,
                        pose_input_size=pose_input_size,
                        backend=self.backend,
                        device=self.device
                    )
                    print("RTMPose local model initialization successful")
                    return
                else:
                    print("Local model files incomplete, using online download")
            else:
                print("models directory doesn't exist, using online download")
                self.wholebody = Wholebody(
                    mode=mode,
                    backend=self.backend,
                    device=self.device
                )
                print("RTMPose online model initialization successful")
            
        except Exception as e:
            print(f"RTMPose initialization failed: {e}")

    def get_keypoint_mapping(self):
        """Get keypoint mapping (COCO 17 keypoint format)"""
        # RTMPose and YOLO both use COCO 17 keypoint format, same order
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        return list(range(17))  # 1:1 mapping
    
    def update_model(self, mode='balanced'):
        """Update model"""
        print(f"Updating RTMPose model to mode: {mode}")
        self.init_rtmpose(mode)
        print(f"RTMPose processor updated to mode: {mode}")
    
    def process_frame(self, frame, exercise_type):
        # 图像尺寸调整
        h, w = frame.shape[:2]
        original_size = (w, h)
        
        if w > 640 or h > 640:
            scale = min(640/w, 640/h)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            scale_factor = scale
        else:
            scale_factor = 1.0
        
        # 初始化多人结果列表
        all_angles = []
        all_keypoints = []
        all_angle_points = []
        
        try:
            # 使用RTMPose进行姿态检测
            detected_keypoints, scores = self.wholebody(frame)
            
            # 调试信息：打印检测结果格式
            print(f"RTMPose检测结果 - detected_keypoints类型: {type(detected_keypoints)}, 长度: {len(detected_keypoints) if detected_keypoints is not None else 0}")
            if detected_keypoints is not None and len(detected_keypoints) > 0:
                print(f"第一个人的关键点形状: {detected_keypoints[0].shape}")
            
            # 处理多人结果
            if detected_keypoints is not None and len(detected_keypoints) > 0:
                # 遍历所有检测到的人
                for i, person_keypoints in enumerate(detected_keypoints):
                    confidence_scores = scores[i] if scores is not None else None
                    
                    # 过滤低置信度关键点
                    if confidence_scores is not None:
                        valid_mask = confidence_scores > self.conf_threshold
                        person_keypoints[~valid_mask] = [0, 0]
                    
                    # 缩放回原始尺寸
                    if scale_factor != 1.0:
                        person_keypoints = person_keypoints / scale_factor
                    
                    # 为每个人计算角度
                    current_angle, angle_point = self.get_exercise_angle(person_keypoints, exercise_type)
                    
                    # 存储每个人的结果
                    all_keypoints.append(person_keypoints)
                    all_angles.append(current_angle)
                    all_angle_points.append(angle_point)
                    
                    print(f"检测到第{i+1}个人 - 角度: {current_angle}, 关键点数量: {len(person_keypoints)}")
                    
        except Exception as e:
            print(f"RTMPose processing failed: {e}")
        
        # 返回多人结果
        return all_angles, all_keypoints
    
    def get_exercise_angle(self, keypoints, exercise_type):
        """Get angle based on exercise type"""
        current_angle = None
        angle_point = None
        
        try:
            if exercise_type == "squat":
                current_angle = self.exercise_counter.count_squat(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[12], keypoints[14], keypoints[16]]
            elif exercise_type == "pushup":
                current_angle = self.exercise_counter.count_pushup(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[6], keypoints[8], keypoints[10]]
            elif exercise_type == "situp":
                current_angle = self.exercise_counter.count_situp(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[5], keypoints[11], keypoints[12]]
            elif exercise_type == "bicep_curl":
                current_angle = self.exercise_counter.count_bicep_curl(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[6], keypoints[8], keypoints[10]]
            elif exercise_type == "lateral_raise":
                current_angle = self.exercise_counter.count_lateral_raise(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[12], keypoints[6], keypoints[8]]
            elif exercise_type == "overhead_press":
                current_angle = self.exercise_counter.count_overhead_press(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[12], keypoints[6], keypoints[8]]
            elif exercise_type == "leg_raise":
                current_angle = self.exercise_counter.count_leg_raise(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[12], keypoints[14], keypoints[16]]
            elif exercise_type == "knee_raise":
                current_angle = self.exercise_counter.count_knee_raise(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[12], keypoints[14], keypoints[16]]
            elif exercise_type == "knee_press":
                current_angle = self.exercise_counter.count_knee_press(keypoints)
                if current_angle is not None:
                    angle_point = [keypoints[11], keypoints[13], keypoints[15]]
            elif exercise_type == "jump_rope":
                # 跳绳模式特殊处理：返回一个固定的角度值，确保角度列表不为空
                # 这样update_ui_components_multi_person方法就能正常处理跳绳模式
                current_angle = 0  # 跳绳模式不使用角度，但需要非None值来保持列表不为空
                angle_point = [keypoints[15], keypoints[16]] if len(keypoints) > 16 else None
        except Exception as e:
            print(f"Error calculating exercise angle: {e}")
            
        return current_angle, angle_point
    
    def set_skeleton_visibility(self, show):
        """Set skeleton display state"""
        self.show_skeleton = show
        print(f"RTMPose skeleton display: {'On' if show else 'Off'}")