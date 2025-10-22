import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    try:
        from PyQt5.QtWidgets import QApplication
        from app.main_window import WorkoutTrackerApp
        
        # 创建 QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("AI Fitness Assistant")
        app.setApplicationVersion("1.0.0")
        
        # 创建并显示主窗口
        window = WorkoutTrackerApp()
        window.show()
        
        # 运行应用程序
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装")
        print("运行: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 