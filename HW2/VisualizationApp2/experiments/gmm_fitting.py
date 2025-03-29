import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import sys
import json
import os
import shutil
from typing import Any, Optional

# 添加pykan到Python路径
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root / 'pykan'))

from kan import *
# 针对gmm_dataset的导入，尝试不同的导入路径
try:
    from .gmm_dataset import GeneralizedGaussianMixture
except ImportError:
    from gmm_dataset import GeneralizedGaussianMixture

def train_and_evaluate(dataset: GeneralizedGaussianMixture, 
                      save_dir: Path,
                      kan_config: Optional[dict[str, Any]] = None,
                      random_state: int = 42) -> dict[str, Any]:
    """训练和评估不同模型"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成训练和测试数据
    X_train, y_train = dataset.generate_samples(N=1000)
    X_test, y_test = dataset.generate_samples(N=200)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)  # 设置为双精度
    
    # 转换数据为PyTorch格式
    train_data = {
        'train_input': torch.FloatTensor(X_train).to(device),
        'train_label': torch.FloatTensor(y_train).reshape(-1, 1).to(device),
        'test_input': torch.FloatTensor(X_test).to(device),
        'test_label': torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    }
    
    # 保存训练数据
    np.savez(save_dir / f'data_{random_state}.npz', 
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    
    # 训练KAN
    if kan_config is None:
        kan_config = {
            'width': [dataset.D, 5, 1],
            'grid': 5,
            'k': 3
        }
    
    # 确保device参数是字符串
    kan_model = KAN(**kan_config, seed=random_state, device=str(device))
    kan_model = kan_model.to(device)  # 确保模型在正确的设备上
    results = kan_model.fit(train_data, opt="LBFGS", steps=50, lamb=0.001)
    
    # 训练MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 5),
        max_iter=1000,
        random_state=random_state
    )
    mlp.fit(X_train, y_train)
    
    # 计算和保存预测结果
    grid_x = np.linspace(X_train.min(), X_train.max(), 100)
    grid_y = np.linspace(X_train.min(), X_train.max(), 100)
    XX, YY = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack((XX.ravel(), YY.ravel()))
    
    with torch.no_grad():
        kan_pred = kan_model(torch.FloatTensor(grid_points).to(device)).cpu().numpy()
        mlp_pred = mlp.predict(grid_points)
        true_density = dataset.pdf(grid_points)
    
        # 计算测试集RMSE
        kan_test_rmse = np.sqrt(np.mean((kan_model(train_data['test_input']).cpu().numpy() - y_test.reshape(-1, 1))**2))
        mlp_test_rmse = np.sqrt(np.mean((mlp.predict(X_test).reshape(-1, 1) - y_test.reshape(-1, 1))**2))
    
    evaluation = {
        'random_state': random_state,
        'kan_test_rmse': float(kan_test_rmse),
        'mlp_test_rmse': float(mlp_test_rmse),
        'training_history': results
    }
    
    # 保存预测结果
    np.savez(save_dir / f'predictions_{random_state}.npz',
             grid_points=grid_points,
             kan_pred=kan_pred,
             mlp_pred=mlp_pred,
             true_density=true_density)
    
    # 保存评估结果
    with open(save_dir / f'evaluation_{random_state}.json', 'w') as f:
        json.dump(evaluation, f)
    
    return evaluation

def run_experiments(save_dir: Path, n_experiments: int = 5) -> dict[str, float]:
    """进行多次随机实验"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    base_seed = 42
    
    for i in range(n_experiments):
        print(f"Running experiment {i+1}/{n_experiments}")
        random_state = base_seed + i
        
        # 创建数据集
        dataset = GeneralizedGaussianMixture(
            D=2,
            K=3,
            p=2.0,
            centers=np.array([[-2, -2], [0, 0], [2, 2]]),
            scales=np.array([[0.3, 0.3], [0.2, 0.2], [0.4, 0.4]]),
            weights=np.array([0.3, 0.4, 0.3]),
            seed=random_state
        )
        
        # 训练和评估
        result = train_and_evaluate(dataset, save_dir / str(random_state), random_state=random_state)
        all_results.append(result)
    
    # 保存所有结果
    with open(save_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f)
    
    # 计算统计量
    kan_rmses = [r['kan_test_rmse'] for r in all_results]
    mlp_rmses = [r['mlp_test_rmse'] for r in all_results]
    
    statistics = {
        'kan_mean_rmse': float(np.mean(kan_rmses)),
        'kan_std_rmse': float(np.std(kan_rmses)),
        'mlp_mean_rmse': float(np.mean(mlp_rmses)),
        'mlp_std_rmse': float(np.std(mlp_rmses)),
    }
    
    with open(save_dir / 'statistics.json', 'w') as f:
        json.dump(statistics, f)
    
    return statistics

if __name__ == '__main__':
    # 使用相对路径，保存在experiments/results目录下
    results_dir = Path(__file__).parent / 'results'
    stats = run_experiments(results_dir)
    print("\nExperiment Statistics:")
    print(f"KAN Test RMSE: {stats['kan_mean_rmse']:.4f} ± {stats['kan_std_rmse']:.4f}")
    print(f"MLP Test RMSE: {stats['mlp_mean_rmse']:.4f} ± {stats['mlp_std_rmse']:.4f}")