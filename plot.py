import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat_file_path = 'dataset/widar/output/0-0.mat'
name = mat_file_path.split('/')[-1]
data = scipy.io.loadmat(mat_file_path)
print(data)

# Assuming the data you want to plot is in a variable named 'data_variable'
data_variable = data['pred']
real_part = data_variable.real.reshape(64, 90)
imag_part = data_variable.imag.reshape(64, 90)
print(real_part.shape, imag_part.shape)

# real_part = (real_part - real_part.mean()) / real_part.std()
# imag_part = (imag_part - imag_part.mean()) / imag_part.std()
# real_part = (real_part - real_part.min()) / (real_part.max() - real_part.min())
# imag_part = (imag_part - imag_part.min()) / (imag_part.max() - imag_part.min())

# Plot the data distribution
# real part
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(real_part, bins=50, edgecolor='k')
plt.title('Real Part')

# imag part
plt.subplot(1, 2, 2)
plt.hist(imag_part, bins=50, edgecolor='k')
plt.title('Imaginary Part')

plt.savefig(f'{name}.png')
plt.show()