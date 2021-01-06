import numpy as np
import matplotlib.pyplot as plt

a1 = [21, 24, 18, 17, 17, 15, 14, 13]
a2 = [21, 26, 23, 23, 25, 26, 25, 25]
a3 = [46, 52, 50, 49, 51, 50, 52, 53]
a4 = [72, 73, 73, 73, 73, 73, 73, 73]
a5 = [72, 73, 73, 73, 73, 73, 73, 73]
a6 = [73, 73, 73, 73, 73, 73, 73, 73]
a7 = [30, 25, 22, 20, 21, 21, 17, 17]
a8 = [30, 29, 26, 27, 32, 33, 30, 30]
a9 = [55, 48, 55, 56, 60, 56, 57, 56]
a10 = [35, 32, 23, 21, 19, 19, 15, 17]
a11 = [35, 33, 34, 34, 33, 34, 35, 32]
a12 = [51, 50, 55, 57, 57, 59, 61, 62]
b = [1, 2, 3, 4, 5, 6, 7, 8]


a1_ = [100 * (i / 73) for i in a1]
a2_ = [100 * (i / 73) for i in a2]
a3_ = [100 * (i / 73) for i in a3]
a4_ = [100 * (i / 73) for i in a4]
a5_ = [100 * (i / 73) for i in a5]
a6_ = [100 * (i / 73) for i in a6]
a7_ = [100 * (i / 140) for i in a7]
a8_ = [100 * (i / 140) for i in a8]
a9_ = [100 * (i / 140) for i in a9]
a10_ = [100 * (i / 140) for i in a10]
a11_ = [100 * (i / 140) for i in a11]
a12_ = [100 * (i / 140) for i in a12]


fig = plt.figure()
plt.plot(b, a1_, 'bo:', label='DenseNet201 vs. I-FGSM')
plt.plot(b, a2_, 'bo--', label='DenseNet201 vs. MI-FGSM')
plt.plot(b, a3_, 'bo-', label='DenseNet201 vs. SAI-FGSM')
plt.plot(b, a4_, 'yo:', label='DenseNet121 vs. I-FGSM')
plt.plot(b, a5_, 'yo--', label='DenseNet121 vs. MI-FGSM')
plt.plot(b, a6_, 'yo-', label='DenseNet121 vs. SAI-FGSM')
plt.plot(b, a7_, 'go:', label='NASNetMobile vs. I-FGSM')
plt.plot(b, a8_, 'go--', label='NASNetMobile vs. MI-FGSM')
plt.plot(b, a9_, 'go-', label='NASNetMobile vs. SAI-FGSM')
plt.plot(b, a10_, 'ro:', label='MobileNet vs. I-FGSM')
plt.plot(b, a11_, 'ro--', label='MobileNet vs. MI-FGSM')
plt.plot(b, a12_, 'ro-', label='MobileNet vs. SAI-FGSM')
plt.legend(["DenseNet201 vs. I-FGSM", "DenseNet201 vs. MI-FGSM", "DenseNet201 vs. SAI-FGSM",
            "DenseNet121 vs. I-FGSM","DenseNet121 vs. MI-FGSM","DenseNet121 vs. SAI-FGSM",
            "NASNetMobile vs. I-FGSM","NASNetMobile vs. MI-FGSM","NASNetMobile vs. SAI-FGSM",
            "MobileNet vs. I-FGSM","MobileNet vs. MI-FGSM","MobileNet vs. SAI-FGSM"], bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)


# my_x_ticks = np.arange(-5, 5, 0.5)
my_y_ticks = np.arange(0, 100, 10)
plt.xticks(b)
plt.yticks(my_y_ticks)
fig.set_figheight(5)
fig.set_figwidth(10)
plt.xlabel('Number of Iterations')
plt.ylabel('Success Rate(%)')

plt.show()