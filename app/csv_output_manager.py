import csv
import os
import time
from datetime import datetime
from collections import defaultdict

class CSVOutputManager:
    """CSV输出管理器，负责从/input读取数据，输出result.csv到/output"""
    
    def __init__(self, input_dir="input", output_dir="output"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.csv_file = os.path.join(output_dir, "result.csv")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化CSV文件
        self._initialize_csv()
        
        # 存储每个人的数据
        self.person_data = defaultdict(lambda: {
            'position': None,
            'jump_count': 0,
            'violations': [],
            'last_update': None
        })
    
    def _initialize_csv(self):
        """初始化CSV文件，写入表头"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['点位', '跳绳数量', '违规类型', '检测时间'])
    
    def read_input_data(self):
        """从input目录读取数据文件"""
        input_files = []
        
        if os.path.exists(self.input_dir):
            for filename in os.listdir(self.input_dir):
                if filename.endswith(('.txt', '.csv', '.json')):
                    input_files.append(os.path.join(self.input_dir, filename))
        
        return input_files
    
    def calculate_position(self, keypoints, person_id):
        """根据关键点计算人员点位"""
        try:
            # 使用髋部中心作为参考点
            left_hip = keypoints[11]  # 左髋
            right_hip = keypoints[12]  # 右髋
            
            # 计算髋部中心点
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # 将坐标转换为点位编号（示例：将画面分为9个区域）
            # 可以根据实际需求调整分区逻辑
            if hip_center_x < 0.33:
                x_zone = "左"
            elif hip_center_x < 0.66:
                x_zone = "中"
            else:
                x_zone = "右"
            
            if hip_center_y < 0.33:
                y_zone = "上"
            elif hip_center_y < 0.66:
                y_zone = "中"
            else:
                y_zone = "下"
            
            position = f"{x_zone}{y_zone}区"
            return position
            
        except Exception as e:
            print(f"计算点位错误 (人员{person_id}): {e}")
            return f"未知点位{person_id}"
    
    def update_jump_rope_data(self, csv_data):
        """更新跳绳数据（从video_processor.py调用）"""
        try:
            for person_data in csv_data:
                person_id = person_data.get('person_id', 0)
                position_x = person_data.get('position_x', 0)
                position_y = person_data.get('position_y', 0)
                jump_count = person_data.get('jump_count', 0)
                violations = person_data.get('violations', [])
                
                # 将坐标转换为点位描述
                position = self._convert_coordinates_to_position(position_x, position_y)
                
                self.person_data[person_id] = {
                    'position': position,
                    'position_x': position_x,
                    'position_y': position_y,
                    'jump_count': jump_count,
                    'violations': violations,
                    'last_update': datetime.now()
                }
            
            return True
            
        except Exception as e:
            print(f"更新跳绳数据错误: {e}")
            return False
    
    def _convert_coordinates_to_position(self, x, y):
        """将坐标转换为点位描述"""
        # 根据坐标值转换为区域描述
        if x < 0.33:
            x_zone = "左"
        elif x < 0.66:
            x_zone = "中"
        else:
            x_zone = "右"
        
        if y < 0.33:
            y_zone = "上"
        elif y < 0.66:
            y_zone = "中"
        else:
            y_zone = "下"
        
        return f"{x_zone}{y_zone}区 (x:{x:.2f}, y:{y:.2f})"

    def export_csv(self):
        """导出CSV文件"""
        try:
            # 准备数据
            data_to_write = []
            for person_id, data in self.person_data.items():
                if data['jump_count'] > 0:
                    # 转换违规类型
                    violation_text = ", ".join(data['violations']) if data['violations'] else "无违规"
                    timestamp = data['last_update'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    data_to_write.append([
                        data['position'],
                        data['jump_count'],
                        violation_text,
                        timestamp
                    ])
            
            # 写入CSV文件
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                writer.writerow(['点位', '跳绳数量', '违规类型', '检测时间'])
                
                # 写入数据
                writer.writerows(data_to_write)
            
            print(f"CSV文件已导出: {self.csv_file}")
            return True
            
        except Exception as e:
            print(f"导出CSV文件错误: {e}")
            return False
    
    def get_csv_file_path(self):
        """获取CSV文件路径"""
        return self.csv_file
    
    def clear_data(self):
        """清空数据"""
        self.person_data.clear()
        
        # 重新初始化CSV文件
        self._initialize_csv()
        
        print("数据已清空")

# 单例模式
_csv_output_manager = None

def get_csv_output_manager():
    """获取CSV输出管理器实例"""
    global _csv_output_manager
    if _csv_output_manager is None:
        _csv_output_manager = CSVOutputManager()
    return _csv_output_manager