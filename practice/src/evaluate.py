# evaluate.py
import os
import paddle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import warnings


# 设置中文字体，避免警告
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
    except:
        print("⚠️  中文字体设置失败，使用默认字体")


# 在文件开头调用
setup_chinese_font()

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)


def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))

    # 使用热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 10})  # 设置注解字体大小

    # 使用英文标题避免字体问题
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # 旋转标签以避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 调整布局
    plt.tight_layout()

    if save_path:
        # 保存时使用更高DPI
        plt.savefig(save_path, bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')
    plt.show()


def evaluate_model(model, data_loader, device):
    """评估单个模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with paddle.no_grad():
        for data, labels in data_loader:
            outputs = model(data)
            probabilities = paddle.nn.functional.softmax(outputs, axis=1)
            predictions = paddle.argmax(outputs, axis=1)

            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def evaluate_single_model(model_name, data_dir, save_dir):
    """评估单个模型性能"""
    try:
        from utils import TrashDataset, get_transforms

        _, val_transform = get_transforms()
        val_dataset = TrashDataset(os.path.join(data_dir, 'val'), transform=val_transform)
        val_loader = paddle.io.DataLoader(val_dataset, batch_size=32, shuffle=False)

        class_names = val_dataset.get_class_names()

        # 动态导入模型
        if model_name == 'EfficientCNN':
            from models.efficient_cnn import EfficientCNN
            model_class = EfficientCNN
        elif model_name == 'BalancedCNN':
            from models.balanced_cnn import BalancedCNN
            model_class = BalancedCNN
        elif model_name == 'CustomCNN1Fixed':
            from models.custom_cnn1_fixed import CustomCNN1Fixed
            model_class = CustomCNN1Fixed
        else:
            print(f"❌ 不支持的模型: {model_name}")
            return None

        model_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None

        model = model_class(num_classes=len(class_names))
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
        model.eval()

        # 简单计算准确率
        correct = 0
        total = 0
        with paddle.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                predicted = paddle.argmax(outputs, axis=1)
                total += target.shape[0]
                correct += (predicted == target).astype('float32').sum().item()

        accuracy = correct / total
        return {'accuracy': accuracy}

    except Exception as e:
        print(f"❌ 评估模型 {model_name} 时出错: {str(e)}")
        return None

def calculate_metrics(predictions, labels, class_names):
    """计算评估指标"""
    accuracy = np.mean(predictions == labels)

    # 计算精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # 计算宏平均和加权平均
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # 生成分类报告
    report = classification_report(labels, predictions,
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)

    # 生成混淆矩阵
    cm = confusion_matrix(labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm
    }

    return metrics


def plot_metrics_comparison(metrics_dict, save_path=None):
    """绘制模型指标对比图 - 修复字体版本"""
    models = list(metrics_dict.keys())
    accuracy = [metrics_dict[model]['accuracy'] for model in models]
    f1_macro = [metrics_dict[model]['f1_macro'] for model in models]
    f1_weighted = [metrics_dict[model]['f1_weighted'] for model in models]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, accuracy, width, label='Accuracy', alpha=0.8, color='skyblue')
    plt.bar(x, f1_macro, width, label='F1 Macro', alpha=0.8, color='lightgreen')
    plt.bar(x + width, f1_weighted, width, label='F1 Weighted', alpha=0.8, color='salmon')

    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 在柱状图上显示数值
    for i, (acc, f1_m, f1_w) in enumerate(zip(accuracy, f1_macro, f1_weighted)):
        plt.text(i - width, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, f1_m + 0.01, f'{f1_m:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width, f1_w + 0.01, f'{f1_w:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')
    plt.show()


def evaluate_all_models(data_dir, save_dir):
    """评估所有模型"""
    # 加载数据
    from utils import TrashDataset, get_transforms
    _, val_transform = get_transforms()
    val_dataset = TrashDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_names = val_dataset.get_class_names()

    models_info = {
        'EfficientCNN': None,
    }

    # 动态导入模型
    try:
        from models.efficient_cnn import EfficientCNN
        models_info['EfficientCNN'] = EfficientCNN
    except ImportError:
        print("⚠️  无法导入EfficientCNN")

    # 可选：尝试导入其他模型
    try:
        from models.balanced_cnn import BalancedCNN
        models_info['BalancedCNN'] = BalancedCNN
    except ImportError:
        pass

    try:
        from models.custom_cnn1_fixed import CustomCNN1Fixed
        models_info['CustomCNN1Fixed'] = CustomCNN1Fixed
    except ImportError:
        pass

    results = {}

    for model_name, model_class in models_info.items():
        if model_class is None:
            continue

        model_path = os.path.join(save_dir, f'{model_name}_best.pdparams')

        if not os.path.exists(model_path):
            print(f"⚠️  模型 {model_name} 未找到，跳过评估")
            continue

        print(f"\n评估模型: {model_name}")

        try:
            # 加载模型
            model = model_class(num_classes=len(class_names))
            state_dict = paddle.load(model_path)
            model.set_state_dict(state_dict)

            # 评估模型
            predictions, labels, probabilities = evaluate_model(model, val_loader, paddle.device.get_device())

            # 计算指标
            metrics = calculate_metrics(predictions, labels, class_names)
            results[model_name] = metrics

            # 打印结果
            print(f"准确率: {metrics['accuracy']:.4f}")
            print(f"F1宏平均: {metrics['f1_macro']:.4f}")
            print(f"F1加权平均: {metrics['f1_weighted']:.4f}")

            # 绘制混淆矩阵
            cm_save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
            plot_confusion_matrix(metrics['confusion_matrix'], class_names, model_name, cm_save_path)

        except Exception as e:
            print(f"评估模型 {model_name} 时出错: {str(e)}")
            continue

    return results


def compare_models(results):
    """对比所有模型性能"""
    if not results:
        print("没有可用的评估结果进行对比")
        return

    print("\n" + "=" * 60)
    print("                   Model Performance Comparison")
    print("=" * 60)

    # 打印对比表格
    print(
        f"{'Model':<15} {'Accuracy':<10} {'Precision(M)':<12} {'Recall(M)':<12} {'F1(Macro)':<12} {'F1(Weighted)':<12}")
    print("-" * 80)

    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision_macro']:<12.4f} "
              f"{metrics['recall_macro']:<12.4f} {metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f}")

    print("-" * 80)

    # 绘制对比图
    if len(results) > 1:
        comparison_save_path = os.path.join(os.path.dirname(list(results.keys())[0]), 'model_comparison.png')
        plot_metrics_comparison(results, comparison_save_path)

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['f1_weighted'])
    print(f"\n🏆 Best Model: {best_model[0]} (Weighted F1: {best_model[1]['f1_weighted']:.4f})")


def generate_detailed_report(results, save_dir):
    """生成详细评估报告"""
    report_path = os.path.join(save_dir, 'model_evaluation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Trash Classification System - Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")

        for model_name, metrics in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"Macro F1 Score: {metrics['f1_macro']:.4f}\n")
            f.write(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}\n\n")

    print(f"详细评估报告已保存至: {report_path}")