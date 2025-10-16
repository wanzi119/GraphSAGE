import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import warnings
import time

warnings.filterwarnings('ignore')


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层
        self.convs.append(SAGEConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class LandslideDataProcessor:
    def __init__(self, factor_files, label_file):
        """
        初始化数据处理器

        Args:
            factor_files: 18个因子文件的路径列表
            label_file: 标签文件路径
        """
        self.factor_files = factor_files
        self.label_file = label_file
        self.scaler = StandardScaler()

    def load_raster_data(self):
        """加载栅格数据"""
        print("正在加载栅格数据...")

        # 加载标签数据
        with rasterio.open(self.label_file) as src:
            labels = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.height, self.width = labels.shape

        # 加载因子数据
        factors = []
        for i, factor_file in enumerate(self.factor_files):
            print(f"加载因子 {i + 1}/{len(self.factor_files)}: {factor_file}")
            with rasterio.open(factor_file) as src:
                factor_data = src.read(1)
                factors.append(factor_data)

        factors = np.stack(factors, axis=0)  # Shape: (18, height, width)

        return factors, labels

    def create_node_features(self, factors, labels):
        """创建节点特征"""
        print("创建节点特征...")

        # 将栅格数据转换为节点特征
        # Shape: (height*width, 18)
        node_features = factors.reshape(len(self.factor_files), -1).T

        # 创建有效节点掩码（排除NoData值）
        valid_mask = ~np.isnan(node_features).any(axis=1)

        # 标准化特征
        node_features_scaled = self.scaler.fit_transform(node_features)

        # 创建标签并处理异常值
        node_labels = labels.flatten()

        # 检查标签数据并进行清理
        print(f"标签数据统计:")
        print(f"最小值: {np.min(node_labels)}")
        print(f"最大值: {np.max(node_labels)}")
        print(f"唯一值: {np.unique(node_labels)}")

        # 将标签转换为0-1范围
        # 假设1表示滑坡，0表示非滑坡，其他值设为0
        node_labels_binary = np.where(node_labels == 1, 1, 0)

        # 更新有效掩码，排除NoData和异常标签
        label_mask = ~np.isnan(node_labels) & np.isin(node_labels, [0, 1])
        valid_mask = valid_mask & label_mask

        print(f"有效节点数: {np.sum(valid_mask)}")
        print(f"正样本数: {np.sum(node_labels_binary[valid_mask])}")
        print(f"负样本数: {np.sum(node_labels_binary[valid_mask] == 0)}")

        return node_features_scaled, node_labels_binary, valid_mask

    def create_spatial_graph(self, k=8):
        """创建空间图结构"""
        print("创建空间图结构...")

        # 创建栅格坐标
        rows, cols = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        coords = np.stack([rows.flatten(), cols.flatten()], axis=1)

        # 使用k近邻创建图结构
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # 创建边索引
        edge_list = []
        for i in range(len(coords)):
            for j in range(1, k + 1):  # 跳过自己（索引0）
                edge_list.append([i, indices[i, j]])

        edge_index = np.array(edge_list).T

        return edge_index

    def create_graph_data(self, node_features, node_labels, edge_index, valid_mask):
        """创建图数据对象"""
        print("创建图数据对象...")

        # 只使用有效节点
        valid_indices = np.where(valid_mask)[0]

        # 重新映射边索引
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

        # 过滤边
        valid_edges = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            if src in index_mapping and dst in index_mapping:
                valid_edges.append([index_mapping[src], index_mapping[dst]])

        if len(valid_edges) == 0:
            print("警告: 没有有效边，使用简单的网格连接")
            # 创建简单的网格连接
            valid_edges = []
            for i in range(len(valid_indices) - 1):
                valid_edges.append([i, i + 1])

        edge_index_filtered = np.array(valid_edges).T

        # 创建图数据
        x = torch.FloatTensor(node_features[valid_mask])
        y = torch.LongTensor(node_labels[valid_mask])
        edge_index_tensor = torch.LongTensor(edge_index_filtered)

        data = Data(x=x, edge_index=edge_index_tensor, y=y)

        return data, valid_indices


def train_model_without_sampling(data, num_epochs=200, lr=0.01, device='cuda'):
    """完全不采样的训练函数"""
    print("=" * 50)
    print("开始全数据训练（无采样）...")
    print("=" * 50)

    # 将数据移到设备上
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU训练")

    data = data.to(device)

    num_nodes = data.x.size(0)
    print(f"总节点数: {num_nodes:,}")
    print(f"总边数: {data.edge_index.size(1):,}")
    print(f"特征维度: {data.x.size(1)}")

    # 创建训练集和测试集掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 80/20 分割
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    # 移到设备上
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"训练节点数: {train_mask.sum().item():,}")
    print(f"测试节点数: {test_mask.sum().item():,}")

    # 检查类别平衡
    train_labels = data.y[train_mask]
    test_labels = data.y[test_mask]
    pos_ratio_train = train_labels.sum().item() / len(train_labels)
    pos_ratio_test = test_labels.sum().item() / len(test_labels)

    print(f"训练集正样本比例: {pos_ratio_train:.3f}")
    print(f"测试集正样本比例: {pos_ratio_test:.3f}")

    # 初始化模型
    model = GraphSAGE(
        input_dim=data.x.size(1),
        hidden_dim=128,  # 增大隐藏层
        output_dim=1,
        num_layers=3,
        dropout=0.5
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # 学习率调度器 - 移除verbose参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20
    )

    # 损失函数 - 处理类别不平衡
    if pos_ratio_train < 0.1 or pos_ratio_train > 0.9:
        pos_weight = torch.tensor([1.0 / pos_ratio_train]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"使用加权损失函数，正样本权重: {pos_weight.item():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 训练历史记录
    train_losses = []
    test_accuracies = []
    test_f1_scores = []
    test_auc_scores = []

    # 最佳模型
    best_test_f1 = 0
    best_epoch = 0

    print("\n开始训练...")
    print("-" * 80)

    start_time = time.time()

    # 由于数据量大，使用梯度累积
    accumulation_steps = 4

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()

        # 梯度累积
        total_loss = 0
        optimizer.zero_grad()

        # 如果数据太大，可以考虑分批训练
        if num_nodes > 1000000:  # 如果节点数超过100万
            # 随机选择一部分节点进行训练
            batch_size = min(100000, num_nodes // 10)
            batch_indices = torch.randperm(num_nodes)[:batch_size].to(device)
            batch_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
            batch_mask[batch_indices] = True
            current_train_mask = train_mask & batch_mask
        else:
            current_train_mask = train_mask

        # 前向传播
        out = model(data.x, data.edge_index)
        loss = criterion(out[current_train_mask].squeeze(), data.y[current_train_mask].float())

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_losses.append(loss.item())

        # 每20个epoch评估一次
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                # 在测试集上评估
                out = model(data.x, data.edge_index)

                # 计算概率
                probs = torch.sigmoid(out[test_mask]).squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels = data.y[test_mask].cpu().numpy()

                # 计算指标
                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, zero_division=0)
                recall = recall_score(labels, preds, zero_division=0)
                f1 = f1_score(labels, preds, zero_division=0)

                # AUC
                if len(np.unique(labels)) > 1:
                    auc = roc_auc_score(labels, probs)
                else:
                    auc = 0.0

                test_accuracies.append(accuracy)
                test_f1_scores.append(f1)
                test_auc_scores.append(auc)

                # 更新学习率
                scheduler.step(f1)
                current_lr = optimizer.param_groups[0]['lr']

                # 保存最佳模型
                if f1 > best_test_f1:
                    best_test_f1 = f1
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), 'best_model_no_sampling.pth')

                # 打印结果
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch + 1:4d}/{num_epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Test Acc: {accuracy:.4f} | "
                      f"Prec: {precision:.4f} | "
                      f"Rec: {recall:.4f} | "
                      f"F1: {f1:.4f} | "
                      f"AUC: {auc:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {elapsed_time:.1f}s")

    print("-" * 80)
    print(f"训练完成！最佳F1分数: {best_test_f1:.4f} (Epoch {best_epoch})")

    # 加载最佳模型
    if best_epoch > 0:
        model.load_state_dict(torch.load('best_model_no_sampling.pth'))

    # 返回结果
    return model, {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_f1_scores': test_f1_scores,
        'test_auc_scores': test_auc_scores,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'best_f1': best_test_f1,
        'best_epoch': best_epoch
    }


def evaluate_model_detailed(model, data, test_mask, device='cuda'):
    """详细评估模型性能"""
    print("\n" + "=" * 50)
    print("最终模型评估")
    print("=" * 50)

    model.eval()
    data = data.to(device)
    test_mask = test_mask.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out[test_mask]).squeeze().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        labels = data.y[test_mask].cpu().numpy()

        # 详细指标
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.0

        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)

        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\n混淆矩阵:")
        print(f"TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}")
        print(f"FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'probabilities': probs,
            'predictions': preds,
            'labels': labels
        }


def visualize_training_progress(history):
    """可视化训练过程"""
    plt.figure(figsize=(15, 10))

    # 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 测试准确率
    plt.subplot(2, 2, 2)
    epochs = [20 * (i + 1) for i in range(len(history['test_accuracies']))]
    plt.plot(epochs, history['test_accuracies'], 'b-o')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # F1分数
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['test_f1_scores'], 'g-o')
    if history['best_f1'] > 0:
        plt.axhline(y=history['best_f1'], color='r', linestyle='--',
                    label=f'Best F1: {history["best_f1"]:.4f}')
    plt.title('Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # AUC分数
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['test_auc_scores'], 'r-o')
    plt.title('Test AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress_no_sampling.png', dpi=300, bbox_inches='tight')
    plt.show()


def predict_full_raster_optimized(model, processor, factors, device='cuda'):
    """优化的全栅格预测"""
    print("\n对整个栅格进行预测...")

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # 准备所有数据
        all_features = factors.reshape(len(processor.factor_files), -1).T
        all_features_scaled = processor.scaler.transform(all_features)

        # 创建有效节点掩码
        valid_mask = ~np.isnan(all_features).any(axis=1)
        valid_indices = np.where(valid_mask)[0]

        print(f"有效像素数: {len(valid_indices):,}")

        # 分批处理以节省内存
        batch_size = 50000
        all_probs = np.full(len(all_features), np.nan)

        for i in range(0, len(valid_indices), batch_size):
            batch_end = min(i + batch_size, len(valid_indices))
            batch_indices = valid_indices[i:batch_end]

            print(f"处理批次 {i // batch_size + 1}/{(len(valid_indices) + batch_size - 1) // batch_size}")

            # 创建批次数据
            batch_features = all_features_scaled[batch_indices]
            x = torch.FloatTensor(batch_features).to(device)

            # 简单的边连接（相邻节点）
            edges = []
            for j in range(len(batch_indices) - 1):
                edges.append([j, j + 1])
                edges.append([j + 1, j])  # 双向边

            if edges:
                edge_index = torch.LongTensor(edges).t().to(device)
            else:
                edge_index = torch.LongTensor([[0], [0]]).to(device)

            # 预测
            out = model(x, edge_index)
            probs = torch.sigmoid(out).squeeze().cpu().numpy()

            # 处理单个值的情况
            if isinstance(probs, np.ndarray):
                if probs.ndim == 0:
                    probs = np.array([probs])
            else:
                probs = np.array([probs])

            # 存储结果
            all_probs[batch_indices] = probs

        # 重塑为原始形状
        prob_raster = all_probs.reshape(processor.height, processor.width)

        return prob_raster


def save_probability_raster(prob_raster, output_path, processor):
    """保存概率栅格"""
    print(f"保存概率栅格到: {output_path}")

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=processor.height,
            width=processor.width,
            count=1,
            dtype=rasterio.float32,
            crs=processor.crs,
            transform=processor.transform,
            nodata=np.nan
    ) as dst:
        dst.write(prob_raster.astype(rasterio.float32), 1)


def main():
    """主函数"""
    print("滑坡易发性分析 - 无采样版本")
    print("=" * 50)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("GPU不可用，使用CPU")
        device = 'cpu'

    # 数据文件路径
    factor_files = [
        r'D:\LL\data\clip factor\clip_aspect.tif',
        r'D:\LL\data\clip factor\clip_catchslo.tif',
        r'D:\LL\data\clip factor\clip_conindex.tif',
        r'D:\LL\data\clip factor\clip_convex.tif',
        r'D:\LL\data\clip factor\clip_flowlen.tif',
        r'D:\LL\data\clip factor\clip_JYL.tif',
        r'D:\LL\data\clip factor\clip_ndvi.tif',
        r'D:\LL\data\clip factor\clip_slope.tif',
        r'D:\LL\data\clip factor\clip_tdly.tif',
        r'D:\LL\data\clip factor\clip_texure.tif',
        r'D:\LL\data\clip factor\clip_TPI.tif',
        r'D:\LL\data\clip factor\clip_TRI.tif',
        r'D:\LL\data\clip factor\clip_TWI.tif',
        r'D:\LL\data\clip factor\clip_valdep.tif',
        r'D:\LL\data\clip factor\clip_距断.tif',
        r'D:\LL\data\clip factor\clip_距河.tif',
        r'D:\LL\data\clip factor\clip_距路.tif',
        r'D:\LL\data\clip factor\clip_yanzu.tif'
    ]

    label_file = r'D:\LL\data\灾害面GCN.tif'
    output_file = r'D:\LL\data\landslide_probability_no_sampling.tif'

    # 初始化数据处理器
    processor = LandslideDataProcessor(factor_files, label_file)

    # 加载数据
    factors, labels = processor.load_raster_data()

    # 创建节点特征
    node_features, node_labels, valid_mask = processor.create_node_features(factors, labels)

    # 创建图结构
    edge_index = processor.create_spatial_graph(k=8)

    # 创建图数据
    data, valid_indices = processor.create_graph_data(
        node_features, node_labels, edge_index, valid_mask
    )

    print(f"\n图数据统计:")
    print(f"节点数: {data.x.size(0):,}")
    print(f"边数: {data.edge_index.size(1):,}")
    print(f"特征维度: {data.x.size(1)}")
    print(f"正样本比例: {data.y.sum().item() / len(data.y):.3f}")

    # 训练模型（无采样）
    model, history = train_model_without_sampling(
        data,
        num_epochs=200,
        lr=0.01,
        device=device
    )

    # 详细评估
    eval_results = evaluate_model_detailed(
        model,
        data,
        history['test_mask'],
        device=device
    )

    # 可视化训练过程
    visualize_training_progress(history)

    # 对整个栅格进行预测
    prob_raster = predict_full_raster_optimized(
        model,
        processor,
        factors,
        device=device
    )

    # 保存结果
    save_probability_raster(prob_raster, output_file, processor)

    # 可视化最终结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(labels, cmap='RdYlBu_r')
    plt.title('Original Labels')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(prob_raster, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.title('Predicted Probability (No Sampling)')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    # 绘制概率分布直方图
    valid_probs = prob_raster[~np.isnan(prob_raster)]
    plt.hist(valid_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.legend()

    plt.tight_layout()
    plt.savefig('landslide_analysis_results_no_sampling.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n分析完成！")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    main()