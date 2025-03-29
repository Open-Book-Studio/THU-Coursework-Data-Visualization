# Visualizing Kolmogorov-Arnold Networks——Data Visualization Assignment 2

叶璨铭, ycm24@mails.tsinghua.edu.cn

## 题目背景

### Kolmogorov-Arnold表示定理

在数学中，Kolmogorov-Arnold表示定理（也称为Kolmogorov's superposition theorem）是一个重要的定理，它表明任何连续的多元函数都可以被表示为一系列单变量连续函数的有限组合。具体来说，对于定义在有界域上的任意连续函数 $f : [0,1]^n \to \mathbb{R}$，存在如下表示：

$$f(x) = f(x_1,...,x_n)=\sum_{q=1}^{2n+1}\Phi_q(\sum_{p=1}^n \phi_{q,p}(x_p))$$

其中 $\phi_{q,p}:[0,1]\to\mathbb{R}$ 和 $\Phi_q:\mathbb{R}\to\mathbb{R}$ 都是连续的单变量函数。这个定理的深刻含义在于，它表明加法是唯一真正的多元运算，因为任何其他多元函数都可以通过单变量函数和加法来表示。

### Kolmogorov-Arnold Networks (KAN)

基于Kolmogorov-Arnold表示定理，我们可以将其表示写成矩阵形式：

$$f(x)={\bf \Phi}_{\rm out}\circ{\bf \Phi}_{\rm in}\circ {\bf x}$$

其中：

$${\bf \Phi}_{\rm in}= \begin{pmatrix} \phi_{1,1}(\cdot) & \cdots & \phi_{1,n}(\cdot) \\ \vdots & & \vdots \\ \phi_{2n+1,1}(\cdot) & \cdots & \phi_{2n+1,n}(\cdot) \end{pmatrix},\quad {\bf \Phi}_{\rm out}=\begin{pmatrix} \Phi_1(\cdot) & \cdots & \Phi_{2n+1}(\cdot)\end{pmatrix}$$

Kolmogorov-Arnold Networks (KAN) 是一种基于这个定理的神经网络架构。它将这种表示形式推广到任意深度和宽度，每一层都是一个Kolmogorov-Arnold层，形式为：

$${\bf \Phi}= \begin{pmatrix} \phi_{1,1}(\cdot) & \cdots & \phi_{1,n_{\rm in}}(\cdot) \\ \vdots & & \vdots \\ \phi_{n_{\rm out},1}(\cdot) & \cdots & \phi_{n_{\rm out},n_{\rm in}}(\cdot) \end{pmatrix}$$

整个网络可以表示为：

$${\rm KAN}({\bf x})={\bf \Phi}_{L-1}\circ\cdots \circ{\bf \Phi}_1\circ{\bf \Phi}_0\circ {\bf x}$$

### 广义高斯混合分布（Generalized Gaussian Mixture Model）

为了验证KAN的表达能力和拟合性能，我们选择了广义高斯混合分布(GGMM)作为测试对象。这是一个具有挑战性的密度估计任务，原因如下：

1. **复杂的分布形态**：广义高斯分布通过引入幂次参数p，可以表达多种分布形态：
   - p = 1 时退化为拉普拉斯分布
   - p = 2 时退化为标准高斯分布
   - p → ∞ 时趋近于均匀分布
   - p < 1 时具有更重的尾部

2. **数学表达**：对于第k个分量，其概率密度函数为：

   $$P_{\theta_k}(x_i) = \frac{p}{2\alpha_k \Gamma(1/p)}\exp(-\frac{|x_i-c_k|^p}{\alpha_k^p})$$

   其中：
   - $c_k$ 是中心位置参数
   - $\alpha_k$ 是尺度参数
   - p 是形状参数（幂次）
   - $\Gamma(·)$ 是伽马函数

   由于广义高斯分布是一种“指数家族分布”，可以写成这样的形式
   $$
   P_{\theta_k}(x_i) = \eta_k \exp(-s_k d_k(x_i))
   $$
   其中：

   - 

3. **混合模型的挑战**：通过混合多个广义高斯分布，可以得到更复杂的分布形态：

   $$P(x) = \sum_{k=1}^K \pi_k P_{\theta_k}(x)$$

   其中 $\pi_k$ 是混合权重，满足 $\sum_{k=1}^K \pi_k = 1$。

4. **验证KAN的优势**：
   - 多峰性：混合模型形成的多峰分布对网络的局部适应能力提出了挑战
   - 非线性：广义高斯分布中的幂次项引入了强非线性，测试网络的非线性表达能力
   - 平滑性：分布函数具有良好的平滑性，适合评估网络的泛化能力

## 可视化任务

在本次作业中，我们将设计并实现一系列可视化来展示KAN在拟合复杂分布上的性能。具体来说：

1. 生成具有不同参数设置的广义高斯混合分布数据集
2. 使用不同的方法来拟合这个分布：
   - Kolmogorov-Arnold Networks (KAN)
   - 多层感知机 (MLP)
   - 期望最大化算法 (EM Algorithm)
3. 通过可视化来比较这些方法的性能，并深入分析KAN的网络结构和训练动态

[接下来是具体的数据生成和实验部分...]
