import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat_file_path = 'dataset/widar/20181128/user6/user6-1-1-1-1-r1.mat'
name = mat_file_path.split('/')[-1]
data = scipy.io.loadmat(mat_file_path)

# Assuming the data you want to plot is in a variable named 'data_variable'
data_variable = data['feature']
real_part = data_variable.real
imag_part = data_variable.imag

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