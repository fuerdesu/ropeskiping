# Rope Skipping Counter

基于 PyQt5 和 RTMPose 开发的跳绳计数应用，支持实时姿态识别和自动计数。

## 功能特色

- 实时跳绳计数和姿态分析
- 多人同时跳绳检测
- 违规动作识别（单脚跳、出界等）
- 运动数据统计和历史记录
- 支持视频文件回放分析

## 环境要求

- Python 3.9
- CUDA (可选，用于GPU加速)
- 摄像头或视频文件

## 依赖安装

```bash
# 克隆仓库
git clone https://github.com/fuerdesu/ropeskiping.git
cd ropeskiping

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 运行主程序
python run.py
```

## 核心功能说明

### 跳绳计数

应用使用先进的姿态检测算法实现跳绳动作的精确识别和计数：

1. 姿态检测：
   - 使用 RTMPose 实时检测人体关键点
   - 支持多人同时检测
   - 自动适应不同光线和背景条件

2. 跳绳动作识别：
   - 智能判断起跳和落地
   - 防误判机制，准确区分跳绳和其他动作
   - 自适应阈值调整，提高计数准确性

3. 违规检测：
   - 单脚跳检测
   - 出界检测
   - 跳跃高度异常检测
   - 一跳多摇检测

4. 数据统计：
   - 实时计数显示
   - 运动时长统计
   - 历史记录查看
   - 运动目标设置

## 项目结构

```
ropeskiping/
├── app/                # 应用核心代码
│   ├── video_processor.py    # 视频处理
│   ├── counter_manager.py    # 计数管理
│   └── main_window.py        # 主窗口
├── core/               # 核心功能实现
│   ├── rtmpose_processor.py  # RTMPose处理
│   └── async_pose_detector.py# 异步姿态检测
├── models/            # 模型文件
├── data/             # 数据存储
└── run.py            # 启动文件
```

## 模型说明

项目使用了以下预训练模型：

- YOLOX (人体检测)：`yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx`
- RTMPose (姿态估计)：根据性能需求可选择：
  - 轻量级：`rtmpose-t_simcc-body7_pt-body7_420e-256x192.onnx`
  - 平衡型：`rtmpose-s_simcc-body7_pt-body7_420e-256x192.onnx`
  - 高性能：`rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx`

## 使用建议

1. 摄像头放置：
   - 建议距离：2-3米
   - 高度：与跳绳者腰部平齐
   - 角度：正面或侧面45度

2. 光线条件：
   - 避免强逆光
   - 保持光线充足均匀
   - 避免剧烈光线变化

3. 跳绳动作要领：
   - 保持身体稳定，避免大幅晃动
   - 跳跃高度适中，不要过高或过低
   - 尽量保持在摄像头视野范围内

## 常见问题

1. 计数不准确：
   - 检查光线是否充足
   - 确保完整的身体在画面中
   - 调整与摄像头的距离

2. 无法检测：
   - 确认摄像头正常工作
   - 检查是否有遮挡
   - 验证环境依赖是否正确安装