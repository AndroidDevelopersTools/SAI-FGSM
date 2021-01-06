import numpy as np
import matplotlib.pyplot as plt

a1 = [13, 18, 29, 41, 52, 58, 62]
a2 = [13, 23, 40, 55, 61, 64, 65]
a3 = [17, 51, 61, 69, 71, 70, 71]
a4 = [70, 73, 73, 73, 73, 73, 73]
a5 = [71, 73, 73, 73, 73, 73, 73]
a6 = [73, 73, 73, 73, 73, 73, 73]
a7 = [14, 22, 29, 39, 55, 61, 71]
a8 = [13, 26, 48, 63, 70, 80, 88]
a9 = [22, 54, 77, 86, 104, 116, 121]
a10 = [14, 23, 34, 49, 54, 66, 72]
a11 = [14, 34, 48, 69, 73, 85, 92]
a12 = [26, 54, 75, 93, 101, 124, 129]
b = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]


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


plt.figure()
plt.subplot(2,2,1)
plt.plot(b, a1_, 'bo:', label='DenseNet201 vs. I-FGSM')
plt.plot(b, a2_, 'bo--', label='DenseNet201 vs. MI-FGSM')
plt.plot(b, a3_, 'bo-', label='DenseNet201 vs. SAI-FGSM')
plt.legend(["DenseNet201 vs. I-FGSM", "DenseNet201 vs. MI-FGSM", "DenseNet201 vs. SAI-FGSM"])

plt.xlabel('The size of perturbation')
plt.ylabel('Success Rate(%)')
plt.subplot(2,2,2)
plt.plot(b, a4_, 'yo:', label='DenseNet121 vs. I-FGSM')
plt.plot(b, a5_, 'yo--', label='DenseNet121 vs. MI-FGSM')
plt.plot(b, a6_, 'yo-', label='DenseNet121 vs. SAI-FGSM')
plt.legend(["DenseNet121 vs. I-FGSM","DenseNet121 vs. MI-FGSM","DenseNet121 vs. SAI-FGSM"])

plt.xlabel('The size of perturbation')
plt.ylabel('Success Rate(%)')
plt.subplot(2,2,3)
plt.plot(b, a7_, 'go:', label='NASNetMobile vs. I-FGSM')
plt.plot(b, a8_, 'go--', label='NASNetMobile vs. MI-FGSM')
plt.plot(b, a9_, 'go-', label='NASNetMobile vs. SAI-FGSM')
plt.legend(["NASNetMobile vs. I-FGSM","NASNetMobile vs. MI-FGSM","NASNetMobile vs. SAI-FGSM"])

plt.xlabel('The size of perturbation')
plt.ylabel('Success Rate(%)')
plt.subplot(2,2,4)
plt.plot(b, a10_, 'ro:', label='MobileNet vs. I-FGSM')
plt.plot(b, a11_, 'ro--', label='MobileNet vs. MI-FGSM')
plt.plot(b, a12_, 'ro-', label='MobileNet vs. SAI-FGSM')
plt.legend(["MobileNet vs. I-FGSM","MobileNet vs. MI-FGSM","MobileNet vs. SAI-FGSM"])



# my_x_ticks = np.arange(-5, 5, 0.5)

# plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, 100, 20)
# plt.yticks(my_y_ticks)
plt.xlabel('The size of perturbation')
plt.ylabel('Success Rate(%)')


plt.show()
