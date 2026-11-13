# train.py
import paddle
import os
import time
import warnings

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)


def setup_training_environment():
    """设置训练环境"""
    os.environ['GLOG_minloglevel'] = '2'


def train_model_stable(model_name, data_dir, save_dir, epochs=20, batch_size=32, learning_rate=0.001):
    """稳定训练版本 - 支持所有模型"""
    print(f"🚀 训练模型: {model_name}")

    # 设置环境
    setup_training_environment()

    # 获取设备
    device = paddle.device.get_device()
    print(f"使用设备: {device}")

    try:
        # 动态导入模型
        if model_name == 'EfficientCNN':
            from models.efficient_cnn import EfficientCNN
            model = EfficientCNN(num_classes=8)
        elif model_name == 'BalancedCNN':
            from models.balanced_cnn import BalancedCNN
            model = BalancedCNN(num_classes=8)
        elif model_name == 'CustomCNN1Fixed':
            from models.custom_cnn1_fixed import CustomCNN1Fixed
            model = CustomCNN1Fixed(num_classes=8)
        else:
            print(f"❌ 不支持的模型: {model_name}")
            return None

        # 将模型移动到设备
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")

        # 导入数据工具
        from utils import TrashDataset, get_transforms, calculate_accuracy

        # 数据加载
        train_transform, val_transform = get_transforms()

        train_path = os.path.join(data_dir, 'train')
        val_path = os.path.join(data_dir, 'val')

        if not os.path.exists(train_path):
            print(f"❌ 训练集路径不存在: {train_path}")
            return None

        print("正在加载数据集...")
        train_dataset = TrashDataset(train_path, transform=train_transform)
        val_dataset = TrashDataset(val_path, transform=val_transform)

        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

        # 数据加载器
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # 根据模型调整学习率
        adjusted_lr = learning_rate
        if model_name == 'BalancedCNN':
            adjusted_lr = learning_rate * 0.5  # 大模型使用更小学习率
            print(f"📉 调整学习率为: {adjusted_lr}")
        elif model_name == 'CustomCNN1Fixed':
            adjusted_lr = learning_rate * 0.8  # 中等学习率
            print(f"📉 调整学习率为: {adjusted_lr}")

        # 优化器
        optimizer = paddle.optimizer.Adam(
            learning_rate=adjusted_lr,
            parameters=model.parameters(),
            weight_decay=1e-4
        )

        criterion = paddle.nn.CrossEntropyLoss()

        # 训练状态
        best_acc = 0.0
        train_losses = []
        val_accuracies = []
        no_improvement_count = 0
        patience = 10  # 早停耐心值

        start_time = time.time()

        print(f"\n开始训练，共 {epochs} 个轮次")
        print(f"批次大小: {batch_size}, 学习率: {adjusted_lr}")
        print("-" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练阶段
            model.train()
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            batch_count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                # 前向传播
                output = model(data)
                loss = criterion(output, target)

                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                # 计算指标
                try:
                    current_loss = float(loss.numpy())
                except:
                    current_loss = 0.0

                current_acc = calculate_accuracy(output, target)

                epoch_train_loss += current_loss
                epoch_train_acc += current_acc
                batch_count += 1

                # 进度显示
                if batch_idx % 50 == 0:
                    progress = 100. * batch_idx / len(train_loader)
                    print(f'E{epoch + 1} [{batch_idx:3d}/{len(train_loader)} ({progress:2.0f}%)] '
                          f'Loss: {current_loss:.4f} | Acc: {current_acc:.4f}')

            # 验证阶段
            model.eval()
            val_acc = 0.0
            val_batches = 0

            with paddle.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)

                    output = model(data)
                    val_acc += calculate_accuracy(output, target)
                    val_batches += 1

            avg_val_acc = val_acc / val_batches if val_batches > 0 else 0

            # 计算训练指标
            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
            avg_train_acc = epoch_train_acc / batch_count if batch_count > 0 else 0

            train_losses.append(avg_train_loss)
            val_accuracies.append(avg_val_acc)

            epoch_time = time.time() - epoch_start

            print(f'Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.1f}s | '
                  f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | '
                  f'Val Acc: {avg_val_acc:.4f}')

            # 保存最佳模型
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                model_save_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
                paddle.save(model.state_dict(), model_save_path)
                no_improvement_count = 0
                print(f'  ✅ 保存最佳模型: {best_acc:.4f}')
            else:
                no_improvement_count += 1
                print(f'  ⏳ 未提升，连续 {no_improvement_count} 轮')

            # 早停检查
            if no_improvement_count >= patience and epoch >= 15:
                print(f"🛑 早停触发: 连续 {patience} 轮验证准确率未提升")
                break

        # 训练完成
        training_time = time.time() - start_time
        minutes, seconds = divmod(training_time, 60)

        print(f'\n🎉 训练完成!')
        print(f'⏱️  总时间: {int(minutes)}分{int(seconds)}秒')
        print(f'🏆 最佳验证准确率: {best_acc:.4f}')
        print(f'📈 最终验证准确率: {val_accuracies[-1]:.4f}')

        # 保存最终模型
        final_model_path = os.path.join(save_dir, f'{model_name}_final.pdparams')
        paddle.save(model.state_dict(), final_model_path)
        print(f'💾 模型已保存至: {save_dir}')

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': best_acc,
            'final_accuracy': val_accuracies[-1],
            'training_time': training_time,
            'final_epoch': epoch + 1
        }

    except Exception as e:
        print(f"❌ {model_name} 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_all_models_stable(data_dir, save_dir, epochs=20, batch_size=32, learning_rate=0.001):
    """稳定训练所有模型"""
    # 按训练难度排序：先训练简单的，再训练复杂的
    models = ['EfficientCNN', 'CustomCNN1Fixed', 'BalancedCNN']
    results = {}

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 开始稳定训练所有模型")
    print(f"📁 数据目录: {data_dir}")
    print(f"💾 保存目录: {save_dir}")
    print(f"⚙️  超参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print("=" * 80)

    successful_models = 0

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"🧠 训练模型: {model_name}")
        print(f"{'=' * 60}")

        try:
            # 检查是否已有训练好的模型
            model_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
            if os.path.exists(model_path):
                print(f"⚠️  发现已训练的模型，跳过 {model_name}")
                continue

            # 根据模型调整训练轮次
            model_epochs = epochs
            if model_name == 'BalancedCNN':
                model_epochs = min(epochs + 10, 40)  # 复杂模型训练更久
                print(f"📈 调整 {model_name} 训练轮次为: {model_epochs}")

            history = train_model_stable(
                model_name=model_name,
                data_dir=data_dir,
                save_dir=save_dir,
                epochs=model_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            if history is not None:
                results[model_name] = history
                successful_models += 1
                print(f"✅ {model_name} 训练完成 - 最佳准确率: {history['best_accuracy']:.4f}")
            else:
                print(f"❌ {model_name} 训练失败")

        except Exception as e:
            print(f"❌ {model_name} 训练异常: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果
    print(f"\n📊 训练汇总:")
    print("-" * 50)
    if results:
        for model_name, history in results.items():
            print(f"{model_name:<15} | 最佳准确率: {history['best_accuracy']:.4f} | "
                  f"训练轮次: {history['final_epoch']} | 时间: {history['training_time']:.1f}s")

        # 找出最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['best_accuracy'])
        print(f"\n🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]['best_accuracy']:.4f})")
    else:
        print("❌ 所有模型训练失败")

    print(f"\n✅ 成功训练: {successful_models}/{len(models)} 个模型")

    return results


# 兼容性函数
def train_all_models(data_dir, save_dir, epochs=20, batch_size=32, learning_rate=0.001):
    """兼容性函数"""
    return train_all_models_stable(data_dir, save_dir, epochs, batch_size, learning_rate)


if __name__ == "__main__":
    # 测试训练
    data_dir = "./data"
    save_dir = "./data"

    results = train_all_models_stable(
        data_dir,
        save_dir,
        epochs=25,
        batch_size=32,
        learning_rate=0.001
    )