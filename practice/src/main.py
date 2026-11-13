# main.py
import os
import time
from pathlib import Path


class TrashClassificationSystem:
    def __init__(self):
        # 使用 Path 对象处理路径，更安全
        base_dir = Path("./data")
        self.original_data_dir = base_dir / "data/archive"
        self.data_dir = base_dir / "data/dataset"
        self.save_dir = base_dir / "saved_models"

        # 创建必要的目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 状态标志
        self.is_training = False
        self.stop_training = False

    def setup_environment(self):
        """快速环境检查"""
        print("=== 环境检查 ===")
        try:
            import paddle
            print(f"✓ PaddlePaddle 版本: {paddle.__version__}")

            # 检查GPU
            if paddle.device.is_compiled_with_cuda():
                gpu_count = paddle.device.cuda.device_count()
                print(f"✓ GPU 支持: 可用 ({gpu_count}个GPU)")
                # 设置GPU设备
                paddle.device.set_device('gpu:0')
            else:
                print("⚠ GPU 支持: 不可用，使用CPU")
                paddle.device.set_device('cpu')

            return True
        except ImportError as e:
            print(f"✗ PaddlePaddle 未安装: {e}")
            return False

    def show_menu(self):
        """显示优化后的主菜单"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "=" * 50)
        print("       智能垃圾识别分类系统")
        print("=" * 50)
        print("1. 📁 划分数据集")
        print("2. 🏋️ 训练所有模型")
        print("3. 🔄 继续训练模型")  # 新增选项
        print("4. 📊 评估所有模型")
        print("5. 📈 模型对比")
        print("6. 🖼️  启动GUI界面")
        print("7. 🔄 完整流程（1-4）")
        print("8. ❌ 退出")
        print("=" * 50)

        # 显示当前状态
        if self.is_training:
            print("⚠  当前正在训练模型，请勿重复操作")

    def get_user_choice(self):
        """快速获取用户选择"""
        try:
            choice = input("\n请选择操作 (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return choice
            else:
                print("❌ 无效选择，请输入1-8之间的数字")
                return None
        except (KeyboardInterrupt, EOFError):
            return '8'

    def check_dataset_exists(self):
        """检查数据集是否存在"""
        if not self.original_data_dir.exists():
            print(f"❌ 原始数据集目录不存在: {self.original_data_dir}")
            return False
        return True

    def run_data_split(self):
        """运行数据集划分"""
        print("\n=== 数据集划分 ===")

        if not self.check_dataset_exists():
            return False

        try:
            # 延迟导入，减少启动时间
            from split_dataset import split_dataset, check_dataset

            print("正在划分数据集...")
            split_dataset(str(self.original_data_dir), str(self.data_dir))
            check_dataset(str(self.data_dir))
            print("✅ 数据集划分完成")
            return True
        except Exception as e:
            print(f"❌ 数据集划分失败: {str(e)}")
            return False

    def get_training_params(self):
        params = {
            'epochs': 30,  # 训练轮次
            'batch_size': 64,  # 批次大小
            'learning_rate': 0.001  # 学习率
        }

        try:
            epochs = input(f"训练轮数 (默认{params['epochs']}): ").strip()
            if epochs:
                params['epochs'] = int(epochs)

            batch_size = input(f"批次大小 (默认{params['batch_size']}): ").strip()
            if batch_size:
                params['batch_size'] = int(batch_size)

            lr = input(f"学习率 (默认{params['learning_rate']}): ").strip()
            if lr:
                params['learning_rate'] = float(lr)

        except ValueError:
            print("⚠  输入参数无效，使用默认值")

        return params

    def get_continued_training_params(self):
        """获取继续训练参数"""
        params = {
            'epochs': 15,  # 较少的轮次
            'batch_size': 32,
            'learning_rate': 0.0001  # 更小的学习率
        }

        try:
            print("\n📝 继续训练参数设置 (使用更小的学习率进行微调)")

            epochs = input(f"训练轮数 (默认{params['epochs']}): ").strip()
            if epochs:
                params['epochs'] = int(epochs)

            batch_size = input(f"批次大小 (默认{params['batch_size']}): ").strip()
            if batch_size:
                params['batch_size'] = int(batch_size)

            lr = input(f"学习率 (默认{params['learning_rate']}): ").strip()
            if lr:
                params['learning_rate'] = float(lr)

        except ValueError:
            print("⚠️  输入参数无效，使用默认值")

        return params

    def run_model_training(self):
        """运行模型训练"""
        if self.is_training:
            print("❌ 已有训练任务在进行中，请等待完成")
            return False

        print("\n=== 训练所有模型 ===")

        # 检查数据集是否存在
        if not (self.data_dir / 'train').exists():
            print("❌ 训练数据集不存在，请先划分数据集")
            return False

        # 获取参数
        params = self.get_training_params()

        try:
            # 设置训练状态
            self.is_training = True

            # 延迟导入训练模块
            try:
                from train_fast import train_all_models_fast as train_all_models
            except ImportError:
                from train import train_all_models  # 回退到原版本

            # 执行训练
            print("开始训练模型...")
            results = train_all_models(
                data_dir=str(self.data_dir),
                save_dir=str(self.save_dir),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate']
            )

            # 检查训练结果
            success_count = sum(1 for model in ['BalancedCNN', 'EfficientCNN', 'CustomCNN1Fixed']
                                if model in results)

            if success_count > 0:
                print(f"✅ {success_count}个模型训练完成")
                return True
            else:
                print("❌ 所有模型训练失败")
                return False

        except Exception as e:
            print(f"❌ 模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.is_training = False

    def run_continued_training(self):
        """运行继续训练"""
        if self.is_training:
            print("❌ 已有训练任务在进行中，请等待完成")
            return False

        print("\n=== 继续训练模型 ===")

        # 检查数据集和模型是否存在
        if not (self.data_dir / 'train').exists():
            print("❌ 训练数据集不存在，请先划分数据集")
            return False

        # 检查是否有训练好的模型
        model_files = list(self.save_dir.glob('*_best.pdparams'))
        if not model_files:
            print("❌ 未找到训练好的模型文件，请先训练模型")
            return False

        print("📁 找到以下模型文件:")
        for model_file in model_files:
            print(f"  - {model_file.name}")

        # 获取参数
        params = self.get_continued_training_params()

        try:
            # 设置训练状态
            self.is_training = True

            # 导入继续训练模块
            try:
                from train_continued import continue_training_all_models
            except ImportError:
                print("❌ 继续训练模块未找到，请确保 train_continued.py 存在")
                return False

            # 执行继续训练
            print("开始在已训练模型基础上继续训练...")
            results = continue_training_all_models(
                data_dir=str(self.data_dir),
                save_dir=str(self.save_dir),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate']
            )

            # 检查训练结果
            success_count = len(results) if results else 0

            if success_count > 0:
                print(f"✅ {success_count}个模型继续训练完成")
                return True
            else:
                print("❌ 所有模型继续训练失败")
                return False

        except Exception as e:
            print(f"❌ 继续训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.is_training = False

    def run_model_evaluation(self):
        """运行模型评估"""
        print("\n=== 评估所有模型 ===")

        # 检查模型文件是否存在
        model_files = list(self.save_dir.glob('*_best.pdparams'))
        if not model_files:
            print("❌ 未找到训练好的模型文件，请先训练模型")
            return None

        try:
            from evaluate import evaluate_all_models
            print("正在评估模型...")
            results = evaluate_all_models(str(self.data_dir), str(self.save_dir))

            if results:
                print("\n📊 评估结果汇总:")
                for model_name, metrics in results.items():
                    print(f"   {model_name}: 准确率 = {metrics['accuracy']:.4f}")
                return results
            else:
                print("❌ 未获得有效的评估结果")
                return None

        except Exception as e:
            print(f"❌ 模型评估失败: {str(e)}")
            return None

    def run_model_comparison(self, results):
        """运行模型对比"""
        print("\n=== 模型对比 ===")
        try:
            from evaluate import compare_models
            compare_models(results)
            return True
        except Exception as e:
            print(f"❌ 模型对比失败: {str(e)}")
            return False

    def run_gui(self):
        """运行GUI界面"""
        print("\n=== 启动GUI界面 ===")
        try:
            from gui import main as gui_main
            print("正在启动GUI界面...")
            gui_main()
            return True
        except Exception as e:
            print(f"❌ GUI启动失败: {str(e)}")
            return False

    def run_complete_workflow(self):
        """运行完整流程"""
        print("\n=== 开始完整流程 ===")

        # 1. 划分数据集
        print("\n▶ 正在执行: 划分数据集")
        if not self.run_data_split():
            return False

        # 2. 训练所有模型
        print("\n▶ 正在执行: 训练所有模型")
        if not self.run_model_training():
            return False

        # 3. 评估所有模型
        print("\n▶ 正在执行: 评估所有模型")
        results = self.run_model_evaluation()
        if not results:
            return False

        # 4. 模型对比
        print("\n▶ 正在执行: 模型对比")
        self.run_model_comparison(results)

        print("\n✅ 完整流程执行完毕")
        return True

    def main_loop(self):
        """主循环"""
        # 欢迎信息
        print("欢迎使用智能垃圾识别分类系统!")
        print("正在初始化环境...")

        # 环境检查
        if not self.setup_environment():
            print("❌ 环境检查失败，请安装必要的依赖包")
            input("按Enter键退出...")
            return

        while True:
            try:
                self.show_menu()
                choice = self.get_user_choice()

                if not choice:
                    time.sleep(1)  # 短暂延迟后重新显示
                    continue

                start_time = time.time()

                # 执行选择的操作
                if choice == '1':
                    self.run_data_split()
                elif choice == '2':
                    self.run_model_training()
                elif choice == '3':  # 新增继续训练选项
                    self.run_continued_training()
                elif choice == '4':
                    self.run_model_evaluation()
                elif choice == '5':
                    results = self.run_model_evaluation()
                    if results:
                        self.run_model_comparison(results)
                elif choice == '6':
                    self.run_gui()
                elif choice == '7':
                    self.run_complete_workflow()
                elif choice == '8':
                    print("\n感谢使用智能垃圾识别分类系统，再见！👋")
                    break

                # 显示执行时间（GUI模式除外）
                if choice != '6':
                    execution_time = time.time() - start_time
                    print(f"\n⏱️  操作执行时间: {execution_time:.2f}秒")
                    input("\n按Enter键继续...")

            except KeyboardInterrupt:
                print("\n\n检测到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生未预期错误: {str(e)}")
                input("按Enter键继续...")


def main():
    """主函数"""
    try:
        system = TrashClassificationSystem()
        system.main_loop()
    except Exception as e:
        print(f"系统启动失败: {str(e)}")
        input("按Enter键退出...")


if __name__ == "__main__":
    main()