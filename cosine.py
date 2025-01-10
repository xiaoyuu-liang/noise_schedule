import numpy as np
import matplotlib.pyplot as plt

# 设置参数
steps = 500
s = 0.008  # 你给出的 s 参数

# 生成时间步序列
x = np.linspace(0, steps, steps)

# 计算 alphas_cumprod
alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化

# 计算 alphas 和 betas
alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
betas = 1 - alphas

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(betas, label="betas (Noise schedule)", color='blue')
plt.xlabel("Timesteps")
plt.ylabel("Beta Value")
plt.title(f"Noise Schedule with s = {s}")
plt.legend()
plt.savefig("standard-cosine.png")
plt.show()