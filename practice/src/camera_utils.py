# camera_utils.py
import cv2
import paddle
import numpy as np
from PIL import Image
import time
import warnings

warnings.filterwarnings("ignore")


class CameraRecognizer:
    """摄像头识别器"""

    def __init__(self, model, classes, input_size=(224, 224)):
        self.model = model
        self.classes = classes
        self.input_size = input_size
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.current_prediction = None
        self.current_confidence = 0.0
        self.probabilities = None
        self.processing = False

    def start_camera(self, camera_index=0):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                # 尝试其他摄像头索引
                for i in range(1, 3):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                else:
                    raise Exception("无法打开摄像头")

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.is_running = True
            return True

        except Exception as e:
            print(f"摄像头启动失败: {str(e)}")
            return False

    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_frame(self, frame):
        """处理单帧图像并进行识别"""
        if self.processing or self.model is None:
            return None, 0.0, None

        self.processing = True
        try:
            # 转换BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 预处理图像
            processed_image = self.preprocess_image(pil_image)

            # 预测
            with paddle.no_grad():
                output = self.model(processed_image)
                probabilities = paddle.nn.functional.softmax(output[0]).numpy()
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class] * 100

            self.processing = False
            return predicted_class, confidence, probabilities

        except Exception as e:
            print(f"帧处理失败: {str(e)}")
            self.processing = False
            return None, 0.0, None

    def preprocess_image(self, image):
        """预处理图像"""
        import paddle.vision.transforms as T

        transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)
        return image_tensor

    def get_frame(self):
        """获取摄像头帧"""
        if self.cap is None or not self.is_running:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def draw_prediction_on_frame(self, frame, prediction, confidence, probabilities=None):
        """在帧上绘制预测结果"""
        if frame is None:
            return frame

        # 复制帧以避免修改原始数据
        display_frame = frame.copy()

        # 添加边框
        cv2.rectangle(display_frame, (10, 10), (630, 470), (0, 255, 0), 2)

        if prediction is not None and confidence > 0:
            class_name = self.classes[prediction]

            # 确定颜色（基于置信度）
            if confidence > 80:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif confidence > 60:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 0, 255)  # 红色 - 低置信度

            # 显示主要预测结果
            text = f"{class_name}: {confidence:.1f}%"
            cv2.putText(display_frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 显示前3个预测结果
            if probabilities is not None:
                top_indices = np.argsort(probabilities)[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    prob = probabilities[idx] * 100
                    if prob > 1:  # 只显示概率大于1%的类别
                        class_text = f"{self.classes[idx]}: {prob:.1f}%"
                        cv2.putText(display_frame, class_text, (20, 80 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加操作提示
        cv2.putText(display_frame, "Press 'Q' to quit", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'C' to capture", (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display_frame

    def capture_frame(self, frame, filename=None):
        """捕获当前帧并保存"""
        if frame is None:
            return False

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"

        try:
            cv2.imwrite(filename, frame)
            print(f"图像已保存: {filename}")
            return True
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return False


def open_camera_window(model, classes, camera_index=0):
    """打开独立的摄像头窗口"""
    recognizer = CameraRecognizer(model, classes)

    if not recognizer.start_camera(camera_index):
        print("无法启动摄像头")
        return False

    print("摄像头已启动，按 'Q' 退出，按 'C' 捕获图像")

    try:
        while recognizer.is_running:
            frame = recognizer.get_frame()
            if frame is None:
                break

            # 每隔一定帧数进行一次识别（避免过于频繁）
            if int(time.time() * 10) % 5 == 0:  # 大约每秒识别2次
                prediction, confidence, probabilities = recognizer.process_frame(frame)
                recognizer.current_prediction = prediction
                recognizer.current_confidence = confidence
                recognizer.probabilities = probabilities

            # 绘制预测结果
            display_frame = recognizer.draw_prediction_on_frame(
                frame,
                recognizer.current_prediction,
                recognizer.current_confidence,
                recognizer.probabilities
            )

            # 显示帧
            cv2.imshow('智能垃圾识别 - 摄像头', display_frame)

            # 键盘输入处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                recognizer.capture_frame(frame)

    except Exception as e:
        print(f"摄像头窗口错误: {str(e)}")
    finally:
        recognizer.stop_camera()
        cv2.destroyAllWindows()

    return True