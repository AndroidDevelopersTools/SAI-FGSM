import numpy as np
import matplotlib.pyplot as plt

a1 = [56, 53, 50, 54, 50, 49, 48, 50, 47]
a2 = [55, 52, 51, 50, 52, 54, 46, 49, 45]
a3 = [73, 73, 73, 73, 73, 73, 73, 73, 73]
a4 = [50, 51, 46, 48, 41, 43, 40, 42, 35]
b = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


a1_ = [100 * (i / 140) for i in a1]
a2_ = [100 * (i / 140) for i in a2]
a3_ = [100 * (i / 73) for i in a3]
a4_ = [100 * (i / 73) for i in a4]

plt.figure()
plt.plot(b, a1_, 'bo-', label='MobileNet')
plt.plot(b, a2_, 'yo-', label='NASNetMobile')
plt.plot(b, a3_, 'go-', label='DenseNet121')
plt.plot(b, a4_, 'ro-', label='DenseNet201')
plt.legend(["MobileNet", "NASNetMobile", "DenseNet121", "DenseNet201"])


# my_x_ticks = np.arange(-5, 5, 0.5)
my_y_ticks = np.arange(0, 100, 10)
# plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.xlabel('Lowest Temprature Tmin')
plt.ylabel('Success Rate(%)')

plt.show()