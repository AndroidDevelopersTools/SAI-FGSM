
import numpy as np
import matplotlib.pyplot as plt


# a = np.random.randint(0, 100, [4, 4])
#
# print(type(a))
# print(a)
# print(a.max())
# pos = np.unravel_index(np.argmax(a), a.shape)
# print(pos)
# print(pos[0])
# print(pos_y)
#
# a[pos[0]][pos[1]] = 0
#
# print(a)

# b = a.tolist()
# print(b)
# print(max(b))


# m = np.zeros(a.shape)
# print(m)
#
# m[0][0] = 1
# print(m)

# T = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# epsilons = [0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.3, 0.5, 0.9]
#
#
# plt.figure()
# plt.plot(T, epsilons, 'bo-', label='SAI value')
# plt.show()

a1 = [32, 54, 55, 57, 56, 54, 56, 56, 56]
a2 = [25, 54, 54, 54, 55, 56, 55, 53, 55]
a3 = [73, 73, 73, 73, 73, 73, 73, 73, 73]
a4 = [22, 51, 49, 49, 50, 50, 50, 49, 50]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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

plt.xlabel('Temprature T')
plt.ylabel('Success Rate(%)')

plt.show()