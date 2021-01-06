import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gc
import multiprocessing

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
tf.config.experimental.set_memory_growth(gpus[0], enable= True)

def load_label(url):
    find_label = {}
    file = open(url, 'r', encoding='UTF-8')
    for line in file:
        key, value = line.split(' ')
        find_label[key] = value.replace('\n', '')
    file.close()
    return find_label


def load_ng_data(url, link):
    X_train = []
    y_train = []
    find_label= load_label(link)
    imgs_dirs = url
    for imgs_dir in imgs_dirs:
        imagePaths = list(os.listdir(imgs_dir))
        for imagePath in imagePaths:
            x = img_to_array(load_img(os.path.join(imgs_dir, imagePath), target_size=(224, 224)))
            X_train.append(x / 255.0)
            y_train.append(find_label[imagePath])

    return tf.convert_to_tensor(np.array(X_train)), y_train



def from_probability_get_label(model, image):
    image_probs = model.predict(image)
    # print('--------------')
    # print(np.argmax(image_probs))
    return tf.keras.applications.densenet.decode_predictions(image_probs, top=1)[0][0], np.argmax(image_probs)


loss_temp = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_sample_iterative(model, input_image, input_label, eplison, times):
    if eplison != 0:
        alpha = eplison / times
        for i in range(times):
            with tf.GradientTape() as tape:
                tape.watch(input_image)
                prediction = model(input_image)
                loss = loss_temp(input_label, prediction)

            gradient = tape.gradient(loss, input_image)
            input_image = alpha * tf.sign(gradient) + input_image
        return input_image
    else:
        return input_image


def create_adversarial_sample_MI(model, input_image, input_label, eplison, times):
    if eplison != 0:
        alpha = eplison / times
        g = 0
        for i in range(times):
            with tf.GradientTape() as tape:
                tape.watch(input_image)
                prediction = model(input_image)
                loss = loss_temp(input_label, prediction)

            gradient = tape.gradient(loss, input_image)
            g = g + gradient / tf.norm(gradient, ord=1)

            input_image = alpha * tf.sign(g) + input_image

        return input_image
    else:
        return input_image


def create_adversarial_sample(model, input_image, input_label, epsilon, times):
    if epsilon != 0:

        T = 100
        Tmin = 0.5
        t = 0
        # alpha = epsilon / ((times * T) / (2 * Tmin))
        alpha = epsilon / times

        below = input_image - epsilon
        above = input_image + epsilon

        while T >= Tmin:
            for i in range(times):
                with tf.GradientTape() as tape:
                    tape.watch(input_image)
                    prediction = model(input_image)
                    loss = loss_temp(input_label, prediction)

                gradient = tape.gradient(loss, input_image)

                if tf.norm(gradient, ord=2).numpy() != 0:
                    input_image = alpha * tf.sign(gradient) + input_image

                else:
                    p = math.exp(-(1 / T))
                    r = np.random.uniform(low=0, high=1)
                    if r < p:
                        # 接受梯度上升，随机选择一个方向
                        # temp = tf.random.uniform([], minval=0, maxval=epsilon)
                        input_image = alpha * tf.sign(tf.random.normal(gradient.shape)) + input_image

                input_image = tf.clip_by_value(tf.clip_by_value(input_image, below, above), 0, 1)

            t = t + 2
            T = 100 / (1 + t)

        return input_image
    else:
        return input_image


def pre_process_label(index, label_shape):
    return tf.reshape(tf.one_hot(index, label_shape), (1, label_shape))


def display_images(image, description):
    # _, label, confidence = from_probability_get_label(tf.keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet'), image)
    (class_name, label, confidence), class_number = from_probability_get_label(tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet'), image)
    # plt.figure()
    # plt.imshow(image[0])
    # plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence * 100))
    # plt.show()
    return class_number


image, label = load_ng_data([r'C:\Users\Administrator\Desktop\dataset\images'], r'C:\Users\Administrator\Desktop\dataset\labels')

pretrained_model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet')
pretrained_model.trainable = False

epsilons = [0.01]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

I_times_total = []
MI_times_total = []
SAI_times_total = []
for iterative_times in range(9, 10):

    I_times = 0
    MI_times = 0
    SAI_times = 0

    for j in range(1000):
        current_img = image[j]
        current_label = pre_process_label(int(label[j]), 1000)
        current_img = current_img[None, ...]

        for i, eps in enumerate(epsilons):
            # adv_x = image + eps*perturbations
            adv_x_iterative = create_adversarial_sample_iterative(pretrained_model, current_img, current_label, eps, iterative_times + 1)
            adv_x_iterative = tf.clip_by_value(adv_x_iterative, 0, 1)
            if int(display_images(adv_x_iterative, descriptions[i])) != int(label[j]):
                I_times = I_times + 1
            del adv_x_iterative
            gc.collect()


            adv_x_MI = create_adversarial_sample_MI(pretrained_model, current_img, current_label, eps, iterative_times + 1)
            adv_x_MI = tf.clip_by_value(adv_x_MI, 0, 1)
            if int(display_images(adv_x_MI, descriptions[i])) != int(label[j]):
                MI_times = MI_times + 1
            # temp = adv_x - image
            # display_images(temp*100, 'preturbation')

            del adv_x_MI
            gc.collect()

            adv_x = create_adversarial_sample(pretrained_model, current_img, current_label, eps, iterative_times + 1)
            adv_x = tf.clip_by_value(adv_x, 0, 1)
            if int(display_images(adv_x, descriptions[i])) != int(label[j]):
                SAI_times = SAI_times + 1

            del adv_x
            gc.collect()


    I_times_total.append(I_times)
    MI_times_total.append(MI_times)
    SAI_times_total.append(SAI_times)
    print('===========')
    print(SAI_times)
    for x in locals().keys():
        del locals()[x]
    gc.collect()

plt.figure()
plt.plot(range(1, 11), SAI_times_total, 'bo-', label='SAI value')
plt.plot(range(1, 11), I_times_total, 'yo-', label='Iterative value')
plt.plot(range(1, 11), MI_times_total, 'go-', label='MI value')

plt.show()