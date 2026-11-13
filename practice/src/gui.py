# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import paddle
import os
import numpy as np
import threading
import time
import cv2

# 导入所有可用的模型
try:
    from models.balanced_cnn import BalancedCNN
    from models.efficient_cnn import EfficientCNN
    from models.custom_cnn1_fixed import CustomCNN1Fixed
except ImportError as e:
    print(f"部分模型导入失败: {e}")

# 导入摄像头工具
try:
    from camera_utils import CameraRecognizer, open_camera_window

    CAMERA_AVAILABLE = True
except ImportError:
    print("摄像头模块导入失败，摄像头功能不可用")
    CAMERA_AVAILABLE = False

import paddle.vision.transforms as T


class TrashClassificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能垃圾识别分类系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # 设置样式
        self.setup_styles()

        self.model = None
        self.current_model_name = None
        self.classes = ['Cardboard', 'Food Organics', 'Glass', 'Metal',
                        'Miscellaneous Trash', 'Paper', 'Plastic', 'Vegetation']

        # 摄像头相关属性
        self.camera_recognizer = None
        self.is_camera_running = False
        self.camera_thread = None

        # 自动搜索模型目录
        self.model_dir = "./data"
        self.available_models = self.scan_available_models()

        self.setup_ui()

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'))
        style.configure('Section.TLabelframe', font=('Arial', 10, 'bold'))
        style.configure('Accent.TButton', font=('Arial', 10))

    def scan_available_models(self):
        """扫描可用的模型文件"""
        available = {}
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if file.endswith('_best.pdparams'):
                    model_name = file.replace('_best.pdparams', '')
                    available[model_name] = os.path.join(self.model_dir, file)
        return available

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="🧠 智能垃圾识别分类系统",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 模型选择区域
        model_frame = ttk.LabelFrame(main_frame, text="模型选择",
                                     style='Section.TLabelframe', padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(1, weight=1)

        # 模型选择下拉菜单
        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                        state="readonly", width=20)

        # 填充可用的模型
        model_options = []
        for model_name in self.available_models.keys():
            model_options.append(model_name)

        # 如果没有扫描到模型，添加默认选项
        if not model_options:
            model_options = ['BalancedCNN', 'EfficientCNN', 'CustomCNN1Fixed']

        self.model_combo['values'] = model_options
        if model_options:
            self.model_combo.set(model_options[0])
        self.model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        # 模型加载按钮
        self.load_btn = ttk.Button(model_frame, text="加载模型",
                                   command=self.load_model, style='Accent.TButton')
        self.load_btn.grid(row=0, column=2, padx=5)

        # 模型信息显示
        self.model_info = ttk.Label(model_frame, text="未加载模型", foreground="gray")
        self.model_info.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        # 图像选择区域
        image_frame = ttk.LabelFrame(main_frame, text="图像处理",
                                     style='Section.TLabelframe', padding="10")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.select_btn = ttk.Button(image_frame, text="选择图像",
                                     command=self.load_image, style='Accent.TButton')
        self.select_btn.grid(row=0, column=0, padx=5)

        # 摄像头按钮 - 根据可用性设置状态
        self.camera_btn = ttk.Button(image_frame, text="打开摄像头",
                                     command=self.toggle_camera, style='Accent.TButton')
        self.camera_btn.grid(row=0, column=1, padx=5)

        if not CAMERA_AVAILABLE:
            self.camera_btn.configure(state='disabled', text="摄像头不可用")

        self.capture_btn = ttk.Button(image_frame, text="捕获图像",
                                      command=self.capture_image, style='Accent.TButton')
        self.capture_btn.grid(row=0, column=2, padx=5)
        self.capture_btn.configure(state='disabled')

        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="识别结果",
                                      style='Section.TLabelframe', padding="10")
        result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=2)
        result_frame.columnconfigure(1, weight=1)
        result_frame.rowconfigure(0, weight=1)

        # 图像显示区域
        image_display_frame = ttk.Frame(result_frame)
        image_display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        image_display_frame.columnconfigure(0, weight=1)
        image_display_frame.rowconfigure(0, weight=1)

        # 创建图像显示的Canvas，支持滚动条
        self.canvas_frame = ttk.Frame(image_display_frame)
        self.canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)

        # 创建Canvas和滚动条
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=600, height=500)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 垂直滚动条
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        # 水平滚动条
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)

        # 在Canvas上创建图像容器
        self.image_container = self.canvas.create_rectangle(2, 2, 598, 498, outline="lightgray", width=1)

        # 初始提示文本
        self.canvas.create_text(300, 250, text="暂无图像\n\n请选择图像文件或打开摄像头",
                                font=('Arial', 12), justify=tk.CENTER, tags="prompt")

        # 结果显示区域
        result_display_frame = ttk.Frame(result_frame)
        result_display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_display_frame.columnconfigure(0, weight=1)
        result_display_frame.rowconfigure(0, weight=1)

        # 结果文本框
        self.result_text = tk.Text(result_display_frame, height=25, width=45,
                                   font=('Consolas', 10), wrap=tk.WORD)
        result_scrollbar = ttk.Scrollbar(result_display_frame, orient="vertical",
                                         command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)

        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请先加载模型")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, padding=(5, 2))
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_model(self):
        """加载模型"""
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("警告", "请先选择模型类型")
            return

        # 查找模型文件路径
        model_path = None
        if model_name in self.available_models:
            model_path = self.available_models[model_name]
        else:
            # 如果自动扫描没有找到，让用户手动选择
            model_path = filedialog.askopenfilename(
                title=f"选择{model_name}模型文件",
                filetypes=[("PaddlePaddle模型", "*.pdparams"), ("所有文件", "*.*")],
                initialdir=self.model_dir
            )

        if not model_path:
            return

        def load_model_thread():
            try:
                self.progress.start()
                self.status_var.set(f"正在加载模型: {model_name}...")
                self.load_btn.configure(state='disabled')

                # 根据模型名称创建对应的模型实例
                if model_name == 'BalancedCNN':
                    model = BalancedCNN(num_classes=8)
                elif model_name == 'EfficientCNN':
                    model = EfficientCNN(num_classes=8)
                elif model_name == 'CustomCNN1Fixed':
                    model = CustomCNN1Fixed(num_classes=8)
                else:
                    raise ValueError(f"不支持的模型类型: {model_name}")

                # 加载模型权重
                state_dict = paddle.load(model_path)
                model.set_state_dict(state_dict)
                model.eval()

                # 更新模型
                self.model = model
                self.current_model_name = model_name

                # 初始化摄像头识别器
                if CAMERA_AVAILABLE:
                    self.camera_recognizer = CameraRecognizer(model, self.classes)

                # 计算参数量
                total_params = sum(p.numel() for p in model.parameters())

                # 在UI线程中更新状态
                self.root.after(0, lambda: self.on_model_loaded(model_name, total_params))

            except Exception as e:
                self.root.after(0, lambda: self.on_model_load_error(str(e)))
            finally:
                self.progress.stop()

        # 在新线程中加载模型
        threading.Thread(target=load_model_thread, daemon=True).start()

    def on_model_loaded(self, model_name, total_params):
        """模型加载完成回调"""
        self.status_var.set(f"模型加载成功: {model_name}")
        self.load_btn.configure(state='normal')

        # 更新模型信息
        info_text = f"已加载: {model_name} | 参数量: {total_params:,} | 类别数: 8"
        self.model_info.configure(text=info_text, foreground="green")

        # 启用摄像头按钮（如果可用）
        if CAMERA_AVAILABLE:
            self.camera_btn.configure(state='normal')

        messagebox.showinfo("成功", f"{model_name} 模型加载成功！\n参数量: {total_params:,}")

    def on_model_load_error(self, error_msg):
        """模型加载错误回调"""
        self.status_var.set("模型加载失败")
        self.load_btn.configure(state='normal')
        self.model_info.configure(text="模型加载失败", foreground="red")
        messagebox.showerror("错误", f"模型加载失败:\n{error_msg}")

    def load_image(self):
        """加载并识别图像"""
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型！")
            return

        file_path = filedialog.askopenfilename(
            title="选择垃圾图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.process_image(file_path)

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型！")
            return

        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """启动摄像头"""
        if not CAMERA_AVAILABLE:
            messagebox.showwarning("警告", "摄像头功能不可用，请检查OpenCV安装")
            return

        def camera_thread():
            try:
                self.status_var.set("正在启动摄像头...")

                if self.camera_recognizer is None:
                    self.camera_recognizer = CameraRecognizer(self.model, self.classes)

                if self.camera_recognizer.start_camera():
                    self.is_camera_running = True
                    self.root.after(0, self.on_camera_started)

                    # 摄像头循环
                    while self.is_camera_running and self.camera_recognizer.is_running:
                        frame = self.camera_recognizer.get_frame()
                        if frame is not None:
                            # 处理帧并识别
                            prediction, confidence, probabilities = self.camera_recognizer.process_frame(frame)

                            # 更新显示
                            self.root.after(0, lambda: self.update_camera_display(
                                frame, prediction, confidence, probabilities))

                        time.sleep(0.03)  # 控制帧率

                    self.root.after(0, self.stop_camera)
                else:
                    self.root.after(0, lambda: self.on_camera_error("无法启动摄像头"))

            except Exception as e:
                self.root.after(0, lambda: self.on_camera_error(str(e)))

        self.camera_thread = threading.Thread(target=camera_thread, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        """停止摄像头"""
        self.is_camera_running = False
        if self.camera_recognizer:
            self.camera_recognizer.stop_camera()

        self.camera_btn.configure(text="打开摄像头")
        self.capture_btn.configure(state='disabled')
        self.status_var.set("摄像头已停止")

        # 清除显示
        self.canvas.delete("all")
        self.canvas.create_rectangle(2, 2, 598, 498, outline="lightgray", width=1)
        self.canvas.create_text(300, 250, text="摄像头已关闭\n\n请选择图像文件或打开摄像头",
                                font=('Arial', 12), justify=tk.CENTER, tags="prompt")
        self.result_text.delete(1.0, tk.END)

    def on_camera_started(self):
        """摄像头启动成功回调"""
        self.camera_btn.configure(text="关闭摄像头")
        self.capture_btn.configure(state='normal')
        self.status_var.set("摄像头运行中 - 实时识别中...")

    def on_camera_error(self, error_msg):
        """摄像头错误回调"""
        self.is_camera_running = False
        self.camera_btn.configure(text="打开摄像头")
        self.capture_btn.configure(state='disabled')
        self.status_var.set("摄像头错误")
        messagebox.showerror("摄像头错误", f"摄像头操作失败:\n{error_msg}")

    def update_camera_display(self, frame, prediction, confidence, probabilities):
        """更新摄像头显示"""
        if frame is None:
            return

        # 转换OpenCV帧为PIL图像
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # 调整大小以适应显示区域
        display_width = 580
        display_height = 480

        img_ratio = pil_image.width / pil_image.height
        canvas_ratio = display_width / display_height

        if img_ratio > canvas_ratio:
            # 图像更宽
            display_width = display_width
            display_height = int(display_width / img_ratio)
        else:
            # 图像更高
            display_height = display_height
            display_width = int(display_height * img_ratio)

        display_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)

        # 清除Canvas并显示新图像
        self.canvas.delete("all")
        self.canvas.create_rectangle(2, 2, 598, 498, outline="lightgray", width=1)

        # 计算居中位置
        x = (600 - display_width) // 2
        y = (500 - display_height) // 2

        self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.canvas.photo = photo  # 保持引用

        # 更新识别结果
        if prediction is not None and confidence > 0:
            self.update_result_text(prediction, confidence, probabilities, "摄像头实时识别")

    def update_result_text(self, prediction, confidence, probabilities, source="摄像头"):
        """更新结果文本框"""
        predicted_class_name = self.classes[prediction]

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"📹 识别来源: {source}\n")
        self.result_text.insert(tk.END, f"🤖 使用模型: {self.current_model_name}\n")
        self.result_text.insert(tk.END, "─" * 50 + "\n\n")
        self.result_text.insert(tk.END, f"🎯 识别结果: {predicted_class_name}\n")
        self.result_text.insert(tk.END, f"📊 置信度: {confidence:.2f}%\n\n")

        # 显示所有类别概率（按置信度排序）
        if probabilities is not None:
            self.result_text.insert(tk.END, "📈 各类别概率:\n")
            self.result_text.insert(tk.END, "─" * 30 + "\n")

            # 创建概率列表并排序
            prob_list = [(cls, prob * 100) for cls, prob in zip(self.classes, probabilities)]
            prob_list.sort(key=lambda x: x[1], reverse=True)

            for i, (cls_name, prob) in enumerate(prob_list):
                if i == 0:
                    prefix = "🥇 "
                elif i == 1:
                    prefix = "🥈 "
                elif i == 2:
                    prefix = "🥉 "
                else:
                    prefix = "   "

                self.result_text.insert(tk.END, f"{prefix}{cls_name:<20}: {prob:6.2f}%\n")

                # 添加进度条效果
                bar_length = int(prob / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                self.result_text.insert(tk.END, f"     [{bar}]\n\n")

    def capture_image(self):
        """捕获当前摄像头图像"""
        if not self.is_camera_running or self.camera_recognizer is None:
            return

        # 获取当前帧
        frame = self.camera_recognizer.get_frame()
        if frame is not None:
            # 保存图像
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_capture_{timestamp}.jpg"
            if self.camera_recognizer.capture_frame(frame, filename):
                self.status_var.set(f"图像已保存: {filename}")
                messagebox.showinfo("成功", f"图像已保存为: {filename}")
            else:
                messagebox.showerror("错误", "保存图像失败")

    def process_image(self, image_path):
        """处理图像并显示结果"""

        def process_thread():
            try:
                self.progress.start()
                self.status_var.set("正在处理图像...")
                self.select_btn.configure(state='disabled')

                # 显示图像
                image = Image.open(image_path)

                # 调整图像大小用于显示
                canvas_width = 580
                canvas_height = 480

                # 计算适合Canvas的尺寸，保持宽高比
                img_ratio = image.width / image.height
                canvas_ratio = canvas_width / canvas_height

                if img_ratio > canvas_ratio:
                    # 图像更宽
                    display_width = canvas_width
                    display_height = int(canvas_width / img_ratio)
                else:
                    # 图像更高
                    display_height = canvas_height
                    display_width = int(canvas_height * img_ratio)

                # 确保尺寸不会太小
                display_width = max(display_width, 100)
                display_height = max(display_height, 100)

                display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)

                # 进行分类预测
                prediction, probabilities = self.predict_image(image_path)

                # 在UI线程中更新结果
                self.root.after(0, lambda: self.display_results(
                    prediction, probabilities, photo, image_path, display_width, display_height
                ))

            except Exception as e:
                self.root.after(0, lambda: self.on_process_error(str(e)))
            finally:
                self.progress.stop()

        threading.Thread(target=process_thread, daemon=True).start()

    def predict_image(self, image_path):
        """预测图像类别"""
        # 图像预处理
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # 预测
        with paddle.no_grad():
            output = self.model(image_tensor)
            probabilities = paddle.nn.functional.softmax(output[0]).numpy()
            predicted_class = np.argmax(probabilities)

        return predicted_class, probabilities

    def display_results(self, prediction, probabilities, photo, image_path, img_width, img_height):
        """显示识别结果"""
        # 清除Canvas上的旧内容
        self.canvas.delete("all")

        # 重新绘制边框
        self.canvas.create_rectangle(2, 2, 598, 498, outline="lightgray", width=1)

        # 计算图像居中位置
        x = (600 - img_width) // 2
        y = (500 - img_height) // 2

        # 在Canvas上显示图像
        self.canvas.create_image(x, y, anchor=tk.NW, image=photo)

        # 保存图像引用防止被垃圾回收
        self.canvas.photo = photo

        # 设置Canvas的滚动区域
        self.canvas.configure(scrollregion=(0, 0, max(600, img_width + 10), max(500, img_height + 10)))

        # 更新按钮状态
        self.select_btn.configure(state='normal')
        self.status_var.set("分类完成")

        # 显示结果
        predicted_class_name = self.classes[prediction]
        confidence = probabilities[prediction] * 100

        # 获取文件名
        filename = os.path.basename(image_path)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"📁 图像文件: {filename}\n")
        self.result_text.insert(tk.END, f"🤖 使用模型: {self.current_model_name}\n")
        self.result_text.insert(tk.END, "─" * 50 + "\n\n")
        self.result_text.insert(tk.END, f"🎯 识别结果: {predicted_class_name}\n")
        self.result_text.insert(tk.END, f"📊 置信度: {confidence:.2f}%\n\n")

        # 显示所有类别概率（按置信度排序）
        self.result_text.insert(tk.END, "📈 各类别概率:\n")
        self.result_text.insert(tk.END, "─" * 30 + "\n")

        # 创建概率列表并排序
        prob_list = [(cls, prob * 100) for cls, prob in zip(self.classes, probabilities)]
        prob_list.sort(key=lambda x: x[1], reverse=True)

        for i, (cls_name, prob) in enumerate(prob_list):
            if i == 0:
                prefix = "🥇 "
            elif i == 1:
                prefix = "🥈 "
            elif i == 2:
                prefix = "🥉 "
            else:
                prefix = "   "

            self.result_text.insert(tk.END, f"{prefix}{cls_name:<20}: {prob:6.2f}%\n")

            # 添加进度条效果
            bar_length = int(prob / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            self.result_text.insert(tk.END, f"     [{bar}]\n\n")

    def on_process_error(self, error_msg):
        """处理错误"""
        self.select_btn.configure(state='normal')
        self.status_var.set("处理失败")
        messagebox.showerror("错误", f"图像处理失败:\n{error_msg}")

    def on_closing(self):
        """窗口关闭时的清理工作"""
        if self.is_camera_running:
            self.stop_camera()
        self.root.destroy()


def main():
    # 设置高DPI显示（Windows）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()

    # 设置窗口图标（如果有的话）
    try:
        root.iconbitmap("icon.ico")
    except:
        pass

    app = TrashClassificationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()