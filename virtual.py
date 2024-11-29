import numpy as np

def calculate_new_csi(csi, delta):
    # 计算幅度变化比例
    scale = np.sqrt(10**(delta / 10))

    # 计算 H'：只对 H 的幅值进行缩放
    magnitude = np.abs(csi)            # 提取幅值
    phase = np.angle(csi)              # 提取相位
    adjusted_csi = magnitude * scale * np.exp(1j * phase)  # 幅值缩放并保持相位不变
    
    return adjusted_csi

def calculate_power(csi):
    csi_pwr = np.sum(np.abs(csi)**2)

    return 10 * np.log10(csi_pwr)

# 示例
csi = np.array([1+1j, 2+2j, 3+3j])  # 原始CSI
delta_p = 6  # 以dB为单位的delta
print(csi, calculate_power(csi))
new_csi = calculate_new_csi(csi, delta_p)
print(new_csi, calculate_power(new_csi))