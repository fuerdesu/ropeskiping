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
        self.min_rep_time = 0.5  # Minimum time between reps (seconds)
        
        # Exercise configurations
        self.exercise_configs = self.get_exercise_configs()
        
        # Independent counting for leg exercises
        self.leg_exercises = ['leg_raise', 'knee_raise', 'knee_press']

        self.jump_stages = defaultdict(lambda: None)
        self.last_jump_times = defaultdict(float)

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
                'jump_threshold': 0.05,  # 跳跃检测阈值（根据归一化坐标或摄像头距离调整）
                'landing_threshold': 0.02,  # 落地判定阈值
                'keypoints': {
                    'left_foot': 15,
                    'right_foot': 16,
                    'hip_center': 11  # 可取左右髋中点，或直接使用一个髋点
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
                delta_y = self.count_jump_rope(keypoints, person_id)
                violations = self.check_jump_rope_violations(keypoints, person_id)

                # 输出违规信息
                if violations:
                    print(f"Person {person_id} violations: {violations}")

                # 检测一次完整跳跃（起跳 -> 落地）
                if self.jump_stages.get(person_id) is None:
                    self.jump_stages[person_id] = "landed"

                if delta_y > config['jump_threshold'] and self.jump_stages[person_id] == "landed":
                    self.jump_stages[person_id] = "jumping"
                    # 记录起跳时间
                    self.last_jump_times[person_id] = time.time()

                elif delta_y < config['landing_threshold'] and self.jump_stages[person_id] == "jumping":
                    self.jump_stages[person_id] = "landed"

                    # 防抖与最小间隔判断（防止抖动误计数）
                    if self.check_rep_timing(person_id):
                        self.counters[person_id] += 1
                        self.last_count_times[person_id] = time.time()
                        print(f"Person {person_id} completed jump #{self.counters[person_id]}")

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
        """Count jump rope using vertical movement instead of angles"""
        config = self.exercise_configs['jump_rope']
        jump_threshold = config['jump_threshold']
        landing_threshold = config['landing_threshold']
            
        left_foot = keypoints[config['keypoints']['left_foot']]
        right_foot = keypoints[config['keypoints']['right_foot']]
        hip_center = keypoints[config['keypoints']['hip_center']]
    
        # 取平均脚底高度
        avg_foot_y = (left_foot[1] + right_foot[1]) / 2.0
        hip_y = hip_center[1]
    
        # 初始化状态缓存
        if not hasattr(self, 'jump_rope_states'):
            self.jump_rope_states = defaultdict(lambda: {
                'prev_foot_y': avg_foot_y,
                'stage': 'landed',
                'last_jump_time': 0.0,
                'last_jump_height': 0.0
            })
    
        state = self.jump_rope_states[person_id]
        prev_y = state['prev_foot_y']
        delta_y = prev_y - avg_foot_y  # 正数表示上升
    
        # 检测“起跳”阶段
        if state['stage'] == 'landed' and delta_y > jump_threshold:
            state['stage'] = 'airborne'
            state['last_jump_time'] = time.time()
            state['last_jump_height'] = delta_y
    
        # 检测"落地"阶段
        elif state['stage'] == 'airborne' and delta_y < landing_threshold:
            state['stage'] = 'landed'
            
            # 在落地时检测违规行为
            violations = self.check_jump_rope_violations(keypoints, person_id)
            if violations:
                print(f"跳绳违规检测 (人员{person_id}): {violations}")
            
            if self.check_rep_timing(person_id):
                self.counters[person_id] += 1
                self.last_count_times[person_id] = time.time()
    
        # 更新缓存
        state['prev_foot_y'] = avg_foot_y
        return delta_y

    def check_jump_rope_violations(self, keypoints, person_id=0):
        """Detect jump rope violations: single-leg, multiple rope rotations, out-of-bounds"""
        violations = []
        config = self.exercise_configs['jump_rope']
        
        # 使用现有的关键点配置
        left_foot_y = keypoints[15][1]  # 左脚踝
        right_foot_y = keypoints[16][1]  # 右脚踝
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2  # 髋部中心
        
        # 1️⃣ 单脚跳检测
        if abs(left_foot_y - right_foot_y) > 0.03:  # 具体阈值可调
            violations.append("Single-leg jump")
        
        # 2️⃣ 一跳多摇检测（需要配合绳索检测或手部角速度）
        # 这里只放占位逻辑
        if hasattr(self, 'hand_speeds') and self.hand_speeds[person_id] > 2.5:
            violations.append("Multiple rope rotations in one jump")
        
        # 3️⃣ 出界检测（x 坐标超出区域）
        left_x = keypoints[15][0]  # 左脚踝x坐标
        right_x = keypoints[16][0]  # 右脚踝x坐标
        if left_x < 0.05 or right_x > 0.95:
            violations.append("Out of bounds")
        
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