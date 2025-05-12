## Algorithm: GradientDescentRegressor  
  
**Input:**  
- Training data $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$  
- Learning rate $\alpha$  
- Momentum coefficient $\beta$  
- L2 regularization parameter $\lambda_2$  
- L1 regularization parameter $\lambda_1$  
- Number of iterations $T$  
- Batch size $b$ (if $b=n$, use batch GD; if $b<n$, use mini-batch SGD)  
- Boolean flag `use_rmsprop` 
- Boolean flag `use_newton_step`
  
**Output:**  
- Model parameters $\theta \in \mathbb{R}^{d+1}$  
- Prediction result $y_{pred} \in \mathbb{R}^n$  
  
**Algorithm:**  
1. Initialize $\theta \leftarrow $ QR Decomposition (Coefficients)
2. Initialize $\beta_0 \leftarrow$ Mode of $y$ (Intercept)
2. Initialize $v \leftarrow 0$ (velocity for momentum)  
3. Initialize $s \leftarrow 0$ (squared gradient average for RMSProp)  
4. **For** $t = 1$ to $T$ **do**:  
   - **If** $b < n$ **then**:  
     - Randomly sample $b$ indices $\{i_1, i_2, \ldots, i_b\}$ from $\{1, 2, \ldots, n\}$  
     - $X_{batch} \leftarrow X[i_1, i_2, \ldots, i_b]$  
     - $y_{batch} \leftarrow y[i_1, i_2, \ldots, i_b]$  
     - $g \leftarrow \frac{2}{b} \cdot X_{batch}^T \cdot (X_{batch} \cdot \theta - y_{batch}) + \lambda_2 \cdot \theta + \lambda_1 \cdot sign(\theta)$  
   - **Else**:  
     - $g \leftarrow \frac{2}{n} \cdot X^T \cdot (X \cdot \theta - y) + \lambda_2 \cdot \theta + \lambda_1 \cdot sign(\theta)$  
   - **If** `use_rmsprop` **then**:  
     - $s \leftarrow \beta \cdot s + (1 - \beta) \cdot g^2$  
     - $g_{adj} \leftarrow g / (\sqrt{s} + \varepsilon)$ (where $\varepsilon$ is a small constant for numerical stability)  
     - $v \leftarrow \beta \cdot v + (1 - \beta) \cdot g_{adj}$  
   - **Else**:  
     - $v \leftarrow \beta \cdot v + \alpha \cdot g$  
   - $\theta \leftarrow \theta - v$  
   - $L \leftarrow \frac{1}{n} \cdot ||X \cdot \theta - y||^2 + \frac{\lambda_2}{2} \cdot ||\theta||^2 + \lambda_1 \cdot ||\theta||_1$  
5. **Return** $\theta$
