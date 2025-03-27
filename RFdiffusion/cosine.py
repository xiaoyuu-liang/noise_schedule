import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit 

timesteps = 100
t = np.linspace(0, 1, timesteps)  # 归一化时间步
k = 5  # 控制衰减速度

# 单调递减曲线
beta_power = 2 * (1 - t**2)
beta_exp = 2 * (1 - np.exp(-k * t))
beta_tanh = (1 + np.tanh(-k * t + 2))
beta_simple_cos = (1 + np.cos(np.pi * t))
beta_simple_cos_2 = (2 + 2*np.cos(np.pi * t))
beta_laplace = (np.exp(-np.abs(np.linspace(0, 1, timesteps)) / 1) / np.exp(-np.abs(-1) / 1))
beta_std_cos = 2 * np.clip(1 - (np.cos(0.5 * np.pi * ((np.arange(1, 100+1)/(100))+0.008)/(1+0.008)) ** 2) / (np.cos(0.5 * np.pi * ((np.arange(0, 100)/(100))+0.008)/(1+0.008)) ** 2), 0, 0.999)

# 绘制图形
plt.plot(beta_power, label="Power")
# plt.plot(beta_exp, label="Exp")
plt.plot(beta_tanh, label="Tanh")
# plt.plot(beta_laplace, label="Laplace")
plt.plot(beta_simple_cos, label="Simple Cosine")
plt.plot(beta_simple_cos_2, label="Simple Cosine 2")
plt.plot(beta_std_cos, label="Standard Cosine")

plt.title("Custom Beta Functions")
plt.xlabel("Timestep")
plt.ylabel("Beta Value")
plt.title(f"Noise Schedule")
plt.legend()
plt.savefig("noise schedules.png")
plt.show()