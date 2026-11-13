# train_continued.py
import paddle
import os
import time
import warnings

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)


def setup_training_environment():
    """设置训练环境"""
    os.environ['GLOG_minloglevel'] = '2'


def evaluate_single_model_simple(model_name, data_dir, save_dir):
    """简化版的单个模型评估（避免循环导入）"""
    try:
        from utils import TrashDataset, get_transforms

        _, val_transform = get_transforms()
        val_dataset = TrashDataset(os.path.join(data_dir, 'val'), transform=val_transform)
        val_loader = paddle.io.DataLoader(val_dataset, batch_size=32, shuffle=False)

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
            return None

        model_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
        if not os.path.exists(model_path):
            return None

        model = model_class(num_classes=8)
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
        print(f"评估模型 {model_name} 时出错: {str(e)}")
        return None


def continue_training_model(model_name, data_dir, save_dir,
                            epochs=10, batch_size=32, learning_rate=0.0001):
    """在已训练模型基础上继续训练"""
    print(f"🔄 继续训练模型: {model_name}")

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

        # 加载已训练的模型权重
        model_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
        if not os.path.exists(model_path):
            print(f"❌ 找不到已训练的模型: {model_path}")
            return None

        print(f"📥 加载预训练权重: {model_path}")
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
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

        # 微调策略：使用更小的学习率
        fine_tune_lr = learning_rate
        print(f"🎯 微调学习率: {fine_tune_lr}")

        # 优化器 - 使用更小的学习率进行微调
        optimizer = paddle.optimizer.Adam(
            learning_rate=fine_tune_lr,
            parameters=model.parameters(),
            weight_decay=1e-5  # 更小的权重衰减
        )

        criterion = paddle.nn.CrossEntropyLoss()

        # 训练状态
        best_acc = 0.0
        train_losses = []
        val_accuracies = []
        no_improvement_count = 0
        patience = 8  # 更严格的早停

        start_time = time.time()

        print(f"\n开始继续训练，共 {epochs} 个轮次")
        print(f"批次大小: {batch_size}, 学习率: {fine_tune_lr}")
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

            # 保存最佳模型 - 只有当准确率提升时才保存
            if avg_val_acc > best_acc:
                improvement = avg_val_acc - best_acc
                best_acc = avg_val_acc
                model_save_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
                paddle.save(model.state_dict(), model_save_path)
                no_improvement_count = 0
                print(f'  ✅ 保存最佳模型: {best_acc:.4f} (+{improvement:.4f})')
            else:
                no_improvement_count += 1
                print(f'  ⏳ 未提升，连续 {no_improvement_count} 轮')

            # 早停检查 - 微调时更敏感
            if no_improvement_count >= patience:
                print(f"🛑 早停触发: 连续 {patience} 轮验证准确率未提升")
                break

        # 训练完成
        training_time = time.time() - start_time
        minutes, seconds = divmod(training_time, 60)

        print(f'\n🎉 继续训练完成!')
        print(f'⏱️  总时间: {int(minutes)}分{int(seconds)}秒')
        print(f'🏆 最佳验证准确率: {best_acc:.4f}')
        print(f'📈 最终验证准确率: {val_accuracies[-1]:.4f}')

        # 保存微调后的模型
        fine_tuned_path = os.path.join(save_dir, f'{model_name}_fine_tuned.pdparams')
        paddle.save(model.state_dict(), fine_tuned_path)
        print(f'💾 微调模型已保存至: {fine_tuned_path}')

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': best_acc,
            'final_accuracy': val_accuracies[-1],
            'training_time': training_time,
            'final_epoch': epoch + 1,
            'improvement': 0  # 简化处理，不计算提升
        }

    except Exception as e:
        print(f"❌ {model_name} 继续训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def continue_training_all_models(data_dir, save_dir, epochs=10, batch_size=32, learning_rate=0.0001):
    """继续训练所有已存在的模型"""
    models = ['EfficientCNN', 'CustomCNN1Fixed', 'BalancedCNN']
    results = {}

    print("🔄 开始在已训练模型基础上继续训练")
    print(f"📁 数据目录: {data_dir}")
    print(f"💾 保存目录: {save_dir}")
    print(f"⚙️  超参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print("=" * 80)

    successful_models = 0

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"🧠 继续训练模型: {model_name}")
        print(f"{'=' * 60}")

        # 检查模型是否存在
        model_path = os.path.join(save_dir, f'{model_name}_best.pdparams')
        if not os.path.exists(model_path):
            print(f"⚠️  未找到 {model_name} 的预训练模型，跳过")
            continue

        try:
            # 简化的评估（避免复杂的导入问题）
            initial_acc = 0.0
            print(f"📊 开始继续训练 {model_name}...")

            history = continue_training_model(
                model_name=model_name,
                data_dir=data_dir,
                save_dir=save_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            if history is not None:
                results[model_name] = history
                successful_models += 1
                print(f"✅ {model_name} 继续训练完成")
                print(f"📈 最终验证准确率: {history['final_accuracy']:.4f}")
            else:
                print(f"❌ {model_name} 继续训练失败")

        except Exception as e:
            print(f"❌ {model_name} 继续训练异常: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果
    print(f"\n📊 继续训练汇总:")
    print("-" * 50)
    if results:
        for model_name, history in results.items():
            print(f"{model_name:<15} | 最终准确率: {history['final_accuracy']:.4f} | "
                  f"训练轮次: {history['final_epoch']} | 时间: {history['training_time']:.1f}s")

        # 找出最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['final_accuracy'])
        print(f"\n🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]['final_accuracy']:.4f})")
    else:
        print("❌ 所有模型继续训练失败")

    print(f"\n✅ 成功继续训练: {successful_models} 个模型")

    return results


if __name__ == "__main__":
    # 测试继续训练
    data_dir = "./data"
    save_dir = "./data"

    results = continue_training_all_models(
        data_dir,
        save_dir,
        epochs=15,  # 较少的轮次进行微调
        batch_size=32,
        learning_rate=0.0001  # 更小的学习率
    )