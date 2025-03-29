import numpy as np
from pathlib import Path
from scipy.special import gamma
from typing import Optional, Tuple, Dict, List, Union
import torch
import os

class GeneralizedGaussianMixture:
    r"""广义高斯混合分布数据集生成器
    P_{\theta_k}(x_i) = \eta_k \exp(-s_k d_k(x_i)) = \frac{p}{2\alpha_k \Gamma(1/p)}\exp(-|\frac{x_i-c_k}{\alpha_k}|^p)
    """
    
    def __init__(self, 
                 D: int = 2,           # 维度
                 K: int = 3,           # 聚类数量
                 p: float = 2.0,       # 幂次，p=2为标准高斯分布
                 centers: Optional[np.ndarray] = None,  # 聚类中心
                 scales: Optional[np.ndarray] = None,   # 尺度参数
                 weights: Optional[np.ndarray] = None,  # 混合权重
                 seed: int = 42):      # 随机种子
        """初始化GMM数据集生成器
        Args:
            D: 数据维度
            K: 聚类数量
            p: 幂次参数，控制分布的形状
            centers: 聚类中心，形状为(K, D)
            scales: 尺度参数，形状为(K, D)
            weights: 混合权重，形状为(K,)
            seed: 随机种子
        """
        self.D = D
        self.K = K
        self.p = p
        self.seed = seed
        np.random.seed(seed)
        
        # 初始化分布参数
        if centers is None:
            self.centers = np.random.randn(K, D) * 2
        else:
            self.centers = centers
            
        if scales is None:
            self.scales = np.random.uniform(0.1, 0.5, size=(K, D))
        else:
            self.scales = scales
            
        if weights is None:
            self.weights = np.random.dirichlet(np.ones(K))
        else:
            self.weights = weights / weights.sum()  # 确保权重和为1
        
    def component_pdf(self, x: np.ndarray, k: int) -> np.ndarray:
        """计算第k个分量的概率密度
        Args:
            x: 输入数据点，形状为(N, D)
            k: 分量索引
        Returns:
            概率密度值，形状为(N,)
        """
        # 计算归一化常数
        norm_const = self.p / (2 * self.scales[k] * gamma(1/self.p))
        
        # 计算|x_i - c_k|^p / α_k^p
        z = np.abs(x - self.centers[k]) / self.scales[k]
        exp_term = np.exp(-np.sum(z**self.p, axis=1))
        
        return np.prod(norm_const) * exp_term
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """计算混合分布的概率密度
        Args:
            x: 输入数据点，形状为(N, D)
        Returns:
            概率密度值，形状为(N,)
        """
        density = np.zeros(len(x))
        for k in range(self.K):
            density += self.weights[k] * self.component_pdf(x, k)
        return density
    
    def generate_component_samples(self, n: int, k: int) -> np.ndarray:
        """从第k个分量生成样本
        Args:
            n: 样本数量
            k: 分量索引
        Returns:
            样本点，形状为(n, D)
        """
        # 使用幂指数分布的反变换采样
        u = np.random.uniform(-1, 1, size=(n, self.D))
        r = np.abs(u) ** (1/self.p)
        samples = self.centers[k] + self.scales[k] * np.sign(u) * r
        return samples
    
    def generate_samples(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成混合分布的样本
        Args:
            N: 总样本数量
        Returns:
            X: 生成的数据点，形状为(N, D)
            y: 对应的概率密度值，形状为(N,)
        """
        # 根据混合权重确定每个分量的样本数量
        n_samples = np.random.multinomial(N, self.weights)
        
        # 从每个分量生成样本
        samples = []
        for k in range(self.K):
            x = self.generate_component_samples(n_samples[k], k)
            samples.append(x)
        
        # 合并并打乱样本
        X = np.vstack(samples)
        idx = np.random.permutation(N)
        X = X[idx]
        
        # 计算概率密度
        y = self.pdf(X)
        
        return X, y
    
    def save_dataset(self, save_dir: Union[str, Path], name: str = 'gmm_dataset') -> None:
        """保存数据集到文件
        Args:
            save_dir: 保存目录
            name: 数据集名称
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 生成并保存数据
        X, y = self.generate_samples(N=1000)
        np.savez(str(save_path / f'{name}.npz'),
                 X=X, y=y,
                 centers=self.centers,
                 scales=self.scales,
                 weights=self.weights,
                 D=self.D,
                 K=self.K,
                 p=self.p)
    
    @classmethod
    def load_dataset(cls, file_path: Union[str, Path]) -> "GeneralizedGaussianMixture":
        """从文件加载数据集
        Args:
            file_path: 数据文件路径
        Returns:
            加载的GMM对象
        """
        data = np.load(str(file_path))
        return cls(
            D=int(data['D']),
            K=int(data['K']),
            p=float(data['p']),
            centers=data['centers'],
            scales=data['scales'],
            weights=data['weights']
        )

def test_gmm_dataset():
    """测试GMM数据集生成器"""
    # 创建2D的GMM数据集
    gmm = GeneralizedGaussianMixture(
        D=2, 
        K=3, 
        p=2.0,
        centers=np.array([[-2, -2], [0, 0], [2, 2]]),
        scales=np.array([[0.3, 0.3], [0.2, 0.2], [0.4, 0.4]]),
        weights=np.array([0.3, 0.4, 0.3])
    )
    
    # 生成样本
    X, y = gmm.generate_samples(1000)
    
    # 保存数据集
    gmm.save_dataset('test_data')
    
    # 加载数据集
    loaded_gmm = GeneralizedGaussianMixture.load_dataset('test_data/gmm_dataset.npz')
    
    # 验证保存和加载的参数是否一致
    assert np.allclose(gmm.centers, loaded_gmm.centers)
    assert np.allclose(gmm.scales, loaded_gmm.scales)
    assert np.allclose(gmm.weights, loaded_gmm.weights)
    
    print("GMM数据集测试通过！")

if __name__ == '__main__':
    test_gmm_dataset()