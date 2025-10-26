import numpy as np
from collections import deque, defaultdict
import time

class ExerciseCounter:
    """Enhanced exercise counter with multi-person support"""
    
    def __init__(self, smoothing_window=5):
        # Core counting variables with multi-person support
        self.counters = defaultdict(int)  # Counter for each person ID
        self.stages = defaultdict(lambda: None)  # Stage for each person
        
        # Leg exercise tracking with multi-person support
        self.leg_stages = defaultdict(lambda: {'left': None, 'right': None})
        
        # Basic features
        self.smoothing_window = smoothing_window
        self.angle_histories = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.last_count_times = defaultdict(float)  # Last count time for each person
        self.last_hand_time = defaultdict(float)  # Last hand movement time for each person
        self.min_rep_time = 0.3  # 减小最小计数时间间隔，允许更快的连续跳跃
        
        # Exercise configurations
        self.exercise_configs = self.get_exercise_configs()
        
        # Independent counting for leg exercises
        self.leg_exercises = ['leg_raise', 'knee_raise', 'knee_press']

        self.jump_stages = defaultdict(lambda: None)
        self.last_jump_times = defaultdict(float)
        
        # 跳绳违规检测相关属性
        self.violations = defaultdict(list)  # 存储每个人的违规信息

    def get_exercise_configs(self):
        """Exercise-specific angle thresholds"""
        return {
            'squat': {
                'down_angle': 110,
                'up_angle': 160,
                'keypoints': {
                    'left': [11, 13, 15],  # hip, knee, ankle
                    'right': [12, 14, 16]  # hip, knee, ankle
                }
            },
            'pushup': {
                'down_angle': 110,
                'up_angle': 160,
                'keypoints': {
                    'left': [5, 7, 9],    # shoulder, elbow, wrist
                    'right': [6, 8, 10]   # shoulder, elbow, wrist
                }
            },
            'situp': {
                'down_angle': 145,
                'up_angle': 170,
                'keypoints': {
                    'left': [5, 11, 15],  # shoulder, hip, ankle
                    'right': [6, 12, 16]  # shoulder, hip, ankle
                }
            },
            'bicep_curl': {
                'down_angle': 160,
                'up_angle': 60,
                'keypoints': {
                    'left': [5, 7, 9],    # shoulder, elbow, wrist
                    'right': [6, 8, 10]   # shoulder, elbow, wrist
                }
            },
            'lateral_raise': {
                'down_angle': 30,
                'up_angle': 80,
                'keypoints': {
                    'left': [11, 5, 7],    # hip, shoulder, elbow
                    'right': [12, 6, 8]   # hip, shoulder, elbow
                }
            },
            'overhead_press': {
                'down_angle': 30,
                'up_angle': 150,
                'keypoints': {
                    'left': [11, 5, 7],    # hip, shoulder, elbow
                    'right': [12, 6, 8]   # hip, shoulder, elbow
                }
            },
            'leg_raise': {
                'down_angle': 130,
                'up_angle': 160,
                'keypoints': {
                    'left': [5, 11, 13],  # shoulder, hip, knee
                    'right': [6, 12, 14]  # shoulder, hip, knee
                }
            },
            'knee_raise': {
                'down_angle': 110,
                'up_angle': 160,
                'keypoints': {
                    'left': [11, 13, 15],  # hip, knee, ankle
                    'right': [12, 14, 16]  # hip, knee, ankle
                }
            },
            'knee_press': {
                'down_angle': 110,
                'up_angle': 160,
                'keypoints': {
                    'left': [11, 13, 15],  # hip, knee, ankle
                    'right': [12, 14, 16]  # hip, knee, ankle
                }
            },
            'jump_rope': {
                'type': 'height_based',
                'jump_threshold': 0.025,    # 更严格的跳跃检测阈值
                'landing_threshold': 0.015,  # 更严格的落地判定阈值
                'min_jump_height': 0.035,   # 更严格的最小跳跃高度
                'stability_frames': 3,       # 要求连续稳定帧数
                'max_jump_duration': 0.8,    # 最大跳跃持续时间
                'min_jump_interval': 0.25,   # 最小跳跃间隔
                'keypoints': {
                    'left_foot': 15,
                    'right_foot': 16,
                    'hip_center': 11,      # 使用左右髋中点
                    'left_hip': 11,        # 新增：用于稳定性检查
                    'right_hip': 12,       # 新增：用于稳定性检查
                    'left_knee': 13,       # 新增：用于姿态检查
                    'right_knee': 14       # 新增：用于姿态检查
                }
            }
        }
    
    def reset_counter(self, person_id=None):
        """Reset counter for a specific person or all persons"""
        if person_id is None:
            # Reset all counters
            self.counters.clear()
            self.stages.clear()
            self.angle_histories.clear()
            self.last_count_times.clear()
            self.leg_stages.clear()
        else:
            # Reset specific person
            if person_id in self.counters:
                del self.counters[person_id]
            if person_id in self.stages:
                del self.stages[person_id]
            if person_id in self.angle_histories:
                del self.angle_histories[person_id]
            if person_id in self.last_count_times:
                del self.last_count_times[person_id]
            if person_id in self.leg_stages:
                del self.leg_stages[person_id]
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points with enhanced validation"""
        try:
            a = np.array(a, dtype=np.float64)
            b = np.array(b, dtype=np.float64)
            c = np.array(c, dtype=np.float64)
            
            # Check for invalid points (all zeros or NaN)
            if (np.all(a == 0) or np.all(b == 0) or np.all(c == 0) or
                np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c))):
                return None
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Check for zero vectors
            ba_norm = np.linalg.norm(ba)
            bc_norm = np.linalg.norm(bc)
            
            if ba_norm < 1e-6 or bc_norm < 1e-6:
                return None
            
            # Calculate angle using dot product
            cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent numerical errors
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
            
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return None
    
    def smooth_angle(self, angle, person_id):
        """Apply smoothing to reduce noise for a specific person"""
        if angle is None:
            return None
            
        self.angle_histories[person_id].append(angle)
        
        if len(self.angle_histories[person_id]) < 3:
            return angle
            
        # Use median filter to remove outliers, then average
        angles_array = np.array(list(self.angle_histories[person_id]))
        median_angle = np.median(angles_array)
        
        # Remove outliers (angles > 2 std devs from median)
        std_dev = np.std(angles_array)
        if std_dev < 1e-6:  # Prevent division by zero
            return median_angle
            
        filtered_angles = angles_array[np.abs(angles_array - median_angle) <= 2 * std_dev]
        
        return np.mean(filtered_angles) if len(filtered_angles) > 0 else angle
    
    def check_rep_timing(self, person_id):
        """Prevent counting reps too quickly for a specific person"""
        current_time = time.time()
        if current_time - self.last_count_times[person_id] < self.min_rep_time:
            return False
        return True
    
    def count_exercise(self, keypoints, exercise_type, person_id):
        """Generic exercise counting function with person ID"""
        try:
            if exercise_type not in self.exercise_configs:
                print(f"Unknown exercise type: {exercise_type}")
                return None

            config = self.exercise_configs[exercise_type]
            kp = config['keypoints']

            # 🧩 如果是跳绳，用高度检测逻辑
            if exercise_type == 'jump_rope':
                # 直接调用count_jump_rope方法，避免重复计数逻辑
                delta_y = self.count_jump_rope(keypoints, person_id)
                violations = self.check_jump_rope_violations(keypoints, person_id)

                # 输出违规信息
                if violations:
                    print(f"Person {person_id} violations: {violations}")

                return delta_y

            # 🧩 其他常规运动（原有逻辑）
            left_angle = self.calculate_angle(
                keypoints[kp['left'][0]],
                keypoints[kp['left'][1]],
                keypoints[kp['left'][2]]
            )

            right_angle = self.calculate_angle(
                keypoints[kp['right'][0]],
                keypoints[kp['right'][1]],
                keypoints[kp['right'][2]]
            )

            # Handle cases where one side is missing
            if left_angle is None and right_angle is None:
                return None
            elif left_angle is None:
                left_angle = right_angle
            elif right_angle is None:
                right_angle = left_angle

            # Handle leg exercises differently
            if exercise_type in self.leg_exercises:
                return self.count_leg_exercise(left_angle, right_angle, config, person_id)

            # For other exercises, use average angle
            avg_angle = (left_angle + right_angle) / 2
            smoothed_angle = self.smooth_angle(avg_angle, person_id)

            if smoothed_angle is None:
                return None

            # Get thresholds
            up_threshold = config['up_angle']
            down_threshold = config['down_angle']

            # Initialize stage if not set
            if self.stages[person_id] is None:
                self.stages[person_id] = "up" if smoothed_angle > up_threshold else "down"

            # Counting logic with timing check
            if smoothed_angle > up_threshold:
                self.stages[person_id] = "up"
            elif (smoothed_angle < down_threshold and
                  self.stages[person_id] == "up" and
                  self.check_rep_timing(person_id)):

                self.stages[person_id] = "down"
                self.counters[person_id] += 1
                self.last_count_times[person_id] = time.time()

            return smoothed_angle

        except Exception as e:
            print(f"Exercise counting error for person {person_id}: {e}")
            return None

    
    def count_leg_exercise(self, left_angle, right_angle, config, person_id):
        """Count leg exercises with complete up-down cycles for a specific person"""
        up_threshold = config['up_angle']
        down_threshold = config['down_angle']
        
        # Initialize leg stages if not set
        if person_id not in self.leg_stages:
            self.leg_stages[person_id] = {'left': None, 'right': None}
        
        leg_stages = self.leg_stages[person_id]
        
        # Check timing first
        can_count = self.check_rep_timing(person_id)
        
        # Left leg
        if left_angle is not None:
            if left_angle > up_threshold:
                leg_stages['left'] = "up"
            elif (left_angle < down_threshold and 
                  leg_stages['left'] == "up" and
                  can_count):
                self.counters[person_id] += 1
                self.last_count_times[person_id] = time.time()
                leg_stages['left'] = "down"
        
        # Right leg
        if right_angle is not None:
            if right_angle > up_threshold:
                leg_stages['right'] = "up"
            elif (right_angle < down_threshold and 
                  leg_stages['right'] == "up" and
                  can_count):
                self.counters[person_id] += 1
                self.last_count_times[person_id] = time.time()
                leg_stages['right'] = "down"
        
        # Return average angle for display purposes
        if left_angle is not None and right_angle is not None:
            return (left_angle + right_angle) / 2
        elif left_angle is not None:
            return left_angle
        elif right_angle is not None:
            return right_angle
        return None

    def count_jump_rope(self, keypoints, person_id=0):
        """Count jump rope using vertical movement with enhanced stability checks"""
        config = self.exercise_configs['jump_rope']
        jump_threshold = config['jump_threshold']
        landing_threshold = config['landing_threshold']
            
        left_foot = keypoints[config['keypoints']['left_foot']]
        right_foot = keypoints[config['keypoints']['right_foot']]
        hip_center = keypoints[config['keypoints']['hip_center']]
    
        # 取平均脚底高度
        avg_foot_y = (left_foot[1] + right_foot[1]) / 2.0
        hip_y = hip_center[1]
    
        # 扩展状态缓存以包含更多稳定性检查
        if not hasattr(self, 'jump_rope_states'):
            self.jump_rope_states = defaultdict(lambda: {
                'prev_foot_y': avg_foot_y,
                'stage': 'landed',
                'last_jump_time': 0.0,
                'last_jump_height': 0.0,
                'stable_frames': 0,
                'last_stable_y': avg_foot_y,
                'jump_start_y': avg_foot_y,
                'jump_peak_y': avg_foot_y,
                'min_jump_height': config.get('min_jump_height', 0.05),
                'movement_history': deque(maxlen=5),  # 新增：移动历史
                'height_history': deque(maxlen=10),   # 新增：高度历史
                'false_start_count': 0,               # 新增：误判计数器
                'consecutive_jumps': 0                # 新增：连续跳跃计数
            })

        state = self.jump_rope_states[person_id]
        prev_y = state['prev_foot_y']
        delta_y = avg_foot_y - prev_y

        # 记录移动历史
        state['movement_history'].append(delta_y)
        state['height_history'].append(avg_foot_y)
        
        # 计算髋部移动和稳定性
        if not hasattr(state, 'prev_hip_y'):
            state['prev_hip_y'] = hip_y
        hip_delta_y = hip_y - state['prev_hip_y']
        
        # 增强的稳定性检测
        stability_threshold = 0.003  # 降低阈值，提高稳定性要求
        movement_variance = np.var(list(state['movement_history'])) if len(state['movement_history']) > 3 else float('inf')
        is_movement_stable = movement_variance < 0.0001  # 新增：检查移动的方差
        
        # 优化的landed状态稳定性检测
        if state['stage'] == 'landed':
            if abs(avg_foot_y - state['last_stable_y']) < stability_threshold and is_movement_stable:
                state['stable_frames'] += 1
                if state['stable_frames'] >= 3:  # 需要连续3帧稳定
                    state['last_stable_y'] = avg_foot_y
                    state['false_start_count'] = max(0, state['false_start_count'] - 1)  # 稳定时减少误判计数
            else:
                state['stable_frames'] = 0
        
        # 增强的起跳检测逻辑
        if state['stage'] == 'landed':
            # 要求更严格的起跳条件
            consistent_upward = all(dy < -0.01 for dy in list(state['movement_history'])[-3:]) if len(state['movement_history']) >= 3 else False
            significant_movement = abs(delta_y) > jump_threshold
            stable_before_jump = state['stable_frames'] >= 2
            hip_stable = abs(hip_delta_y) < 0.02  # 髋部相对稳定
            
            if consistent_upward and significant_movement and stable_before_jump and hip_stable:
                state['stage'] = 'airborne'
                state['last_jump_time'] = time.time()
                state['jump_start_y'] = prev_y
                state['jump_peak_y'] = avg_foot_y
                state['consecutive_jumps'] += 1
            else:
                state['false_start_count'] += 1
                if state['false_start_count'] >= 5:  # 连续5次误判，提高阈值
                    jump_threshold *= 1.1
                    state['false_start_count'] = 0

        # 优化的空中阶段和落地检测
        if state['stage'] == 'airborne':
            if avg_foot_y < state['jump_peak_y']:
                state['jump_peak_y'] = avg_foot_y
            
            # 更严格的落地检测
            is_descending = delta_y > 0
            near_start_position = abs(avg_foot_y - state['jump_start_y']) < landing_threshold
            stable_landing = movement_variance < 0.0002  # 要求落地时movement较稳定
            
            if (is_descending and near_start_position and stable_landing) or \
               (time.time() - state['last_jump_time'] > 0.8):  # 防止空中停留时间过长
                
                actual_jump_height = state['jump_start_y'] - state['jump_peak_y']
                min_required_height = state['min_jump_height'] * (1 + 0.1 * state['consecutive_jumps'])  # 连续跳跃要求逐渐提高
                
                if actual_jump_height >= min_required_height:
                    current_time = time.time()
                    if current_time - self.last_count_times.get(person_id, 0) > 0.25:  # 增加最小时间间隔
                        self.counters[person_id] += 1
                        self.last_count_times[person_id] = current_time
                else:
                    state['consecutive_jumps'] = 0  # 重置连续跳跃计数
                
                state['stage'] = 'landed'
                state['last_jump_height'] = actual_jump_height

        # 更新缓存
        state['prev_foot_y'] = avg_foot_y
        state['prev_hip_y'] = hip_y
        return delta_y

    def check_jump_rope_violations(self, keypoints, person_id=0):
        """检测跳绳违规情况，包括：单脚跳和跳跃姿态异常"""
        violations = []
        
        # 初始化违规状态追踪
        if not hasattr(self, 'violation_states'):
            self.violation_states = defaultdict(lambda: {
                'consecutive_single_leg': 0,  # 连续单脚跳次数
                'last_violation_time': 0,     # 上次违规时间
                'violation_history': deque(maxlen=5)  # 最近5次违规记录
            })
        
        state = self.violation_states[person_id]
        current_time = time.time()
        
        # 只在新的检测周期开始时重置计数（避免频繁重置）
        if current_time - state['last_violation_time'] > 2.0:  # 2秒无违规则重置
            state['consecutive_single_leg'] = 0
            state['violation_history'].clear()
        
        # 获取关键点
        left_foot_y = keypoints[15][1]   # 左脚踝
        right_foot_y = keypoints[16][1]  # 右脚踝
        left_knee_y = keypoints[13][1]   # 左膝
        right_knee_y = keypoints[14][1]  # 右膝
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2  # 髋部中心
        
        # 1️⃣ 单脚跳检测（更严格的判定）
        foot_height_diff = abs(left_foot_y - right_foot_y)
        knee_height_diff = abs(left_knee_y - right_knee_y)
        
        # 同时考虑脚踝和膝盖的高度差，并检查连续性
        if foot_height_diff > 0.05 and knee_height_diff > 0.04:  # 提高阈值
            state['consecutive_single_leg'] += 1
            if state['consecutive_single_leg'] >= 3:  # 需要连续3次检测到才报警
                violations.append("单脚跳")
        else:
            state['consecutive_single_leg'] = max(0, state['consecutive_single_leg'] - 1)
        
        # 2️⃣ 跳跃姿态检测
        if hasattr(self, 'jump_rope_states'):
            jump_state = self.jump_rope_states.get(person_id, {})
            
            # 获取跳跃相关数据
            jump_height = jump_state.get('last_jump_height', 0)
            movement_history = list(jump_state.get('movement_history', []))
            
            # 只在空中阶段检测姿态
            if jump_state.get('stage') == 'airborne':
                # 计算身体姿态的稳定性
                hip_to_foot_ratio = (hip_y - left_foot_y) / (hip_y - right_foot_y)
                
                # 检测极端情况
                if jump_height > 0.2:  # 跳跃过高（可能是跨步或跑动）
                    violations.append("动作过大")
                elif (hip_to_foot_ratio < 0.7 or hip_to_foot_ratio > 1.3) and \
                     state['consecutive_single_leg'] >= 2:  # 身体倾斜过度
                    violations.append("姿态不稳")
        
        # 更新违规历史
        if violations:
            state['violation_history'].append((current_time, violations))
            state['last_violation_time'] = current_time
            
            # 只保留最近的违规记录
            self.violations[person_id] = violations
        
        return violations

    # Wrapper functions for different exercises with person ID
    def count_squat(self, keypoints, person_id=0):
        """Count squat repetitions for a specific person"""
        return self.count_exercise(keypoints, 'squat', person_id)
    
    def count_pushup(self, keypoints, person_id=0):
        """Count pushup repetitions for a specific person"""
        return self.count_exercise(keypoints, 'pushup', person_id)
    
    def count_situp(self, keypoints, person_id=0):
        """Count situp repetitions for a specific person"""
        return self.count_exercise(keypoints, 'situp', person_id)
    
    def count_bicep_curl(self, keypoints, person_id=0):
        """Count bicep curl repetitions for a specific person"""
        return self.count_exercise(keypoints, 'bicep_curl', person_id)
    
    def count_lateral_raise(self, keypoints, person_id=0):
        """Count lateral raise repetitions for a specific person"""
        return self.count_exercise(keypoints, 'lateral_raise', person_id)
    
    def count_overhead_press(self, keypoints, person_id=0):
        """Count overhead press repetitions for a specific person"""
        return self.count_exercise(keypoints, 'overhead_press', person_id)
    
    def count_leg_raise(self, keypoints, person_id=0):
        """Count leg raise repetitions for a specific person"""
        return self.count_exercise(keypoints, 'leg_raise', person_id)
    
    def count_knee_raise(self, keypoints, person_id=0):
        """Count knee raise repetitions for a specific person"""
        return self.count_exercise(keypoints, 'knee_raise', person_id)
    
    def count_knee_press(self, keypoints, person_id=0):
        """Count knee press repetitions for a specific person"""
        return self.count_exercise(keypoints, 'knee_press', person_id)
    
    def get_counter(self, person_id=0):
        """Get counter value for a specific person"""
        return self.counters.get(person_id, 0)
    
    def get_stage(self, person_id=0):
        """Get stage for a specific person"""
        return self.stages.get(person_id, None)
    
    def get_total_count(self):
        """Get total count across all persons"""
        return sum(self.counters.values())