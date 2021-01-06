import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gc
import threading

# mpl.rcParams['figure.figsize'] = (8, 8)
# mpl.rcParams['axes.grid'] = False
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], enable=True)


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
    find_label = load_label(link)
    imgs_dirs = url
    for imgs_dir in imgs_dirs:
        imagePaths = list(os.listdir(imgs_dir))
        for imagePath in imagePaths:
            x = img_to_array(load_img(os.path.join(imgs_dir, imagePath), target_size=(224, 224)))
            X_train.append(x / 255.0)
            y_train.append(find_label[imagePath])

    return tf.convert_to_tensor(np.array(X_train)), y_train


Decode_model = tf.keras.applications


def from_probability_get_label(model, image):
    image_probs = model.predict(image)
    # print('--------------')
    # print(np.argmax(image_probs))

    return Decode_model.densenet.decode_predictions(image_probs, top=1)[0][0], np.argmax(image_probs)
    # return Decode_model.nasnet.decode_predictions(image_probs, top=1)[0][0], np.argmax(image_probs)
    # return Decode_model.mobilenet.decode_predictions(image_probs, top=1)[0][0], np.argmax(image_probs)


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
            if tf.norm(gradient, ord=2) != 0:
                input_image = alpha * (0.5 * (tf.sign(gradient) + 1)) + input_image
            else:
                break
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

            g = g + gradient

            input_image = input_image + alpha * (0.5 * (tf.sign(g) + 1))

        return input_image
    else:
        return input_image


def create_adversarial_sample(model, input_image, input_label, epsilon, times):
    if epsilon != 0:

        T = 5
        Tmin = 1.1
        t = 0
        alpha = epsilon / times
        g = 0
        below = input_image - epsilon
        above = input_image + epsilon

        while T >= Tmin:
            for i in range(times):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(input_image)
                    prediction = model(input_image)
                    loss = loss_temp(input_label, prediction)

                gradient = tape.gradient(loss, input_image)
                g = g + gradient

                if tf.norm(gradient, ord=2).numpy() != 0:

                    input_image = input_image + alpha * (0.5 * (tf.sign(g) + 1))

                else:
                    p = math.exp(-(1 / T))
                    r = np.random.uniform(low=0, high=1)
                    if r < p:

                        print('random')
                        if tf.norm(g, ord=2).numpy() != 0:

                            input_image = input_image - 2 * alpha * (g / tf.norm(g, ord=2))
                        else:

                            break

                    else:

                        if tf.norm(g, ord=2).numpy() != 0:

                            input_image = input_image + alpha * (0.5 * (tf.sign(g) + 1))
                        else:

                            break

            t = t + 1
            T = 5 / (1 + t)

            if T >= Tmin:
                input_image = tf.clip_by_value(input_image, below, above)

        return input_image
    else:
        return input_image


def pre_process_label(index, label_shape):
    return tf.reshape(tf.one_hot(index, label_shape), (1, label_shape))


def display_images(image, description):
    (class_name, label, confidence), class_number = from_probability_get_label(Decode_model.densenet.DenseNet201(include_top=True, weights='imagenet'), image)
    # (class_name, label, confidence), class_number = from_probability_get_label(Decode_model.nasnet.NASNetMobile(include_top=True, weights='imagenet'), image)
    # (class_name, label, confidence), class_number = from_probability_get_label(Decode_model.mobilenet.MobileNet(include_top=True, weights='imagenet'), image)

    plt.figure()
    plt.imshow(image[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence * 100))
    plt.show()
    return class_number


class myThread(threading.Thread):
    def __init__(self, threadID, name, model, input_image, input_label, epsilon, times, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model = model
        self.input_image = input_image
        self.input_label = input_label
        self.epsilon = epsilon
        self.times = times
        self.q = q

    def run(self):
        adv_x_iterative = create_adversarial_sample_iterative(self.model, self.input_image, self.input_label,
                                                              self.epsilon, self.times)
        self.q[self.name] = adv_x_iterative


class myThread_1(threading.Thread):
    def __init__(self, threadID, name, model, input_image, input_label, epsilon, times, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model = model
        self.input_image = input_image
        self.input_label = input_label
        self.epsilon = epsilon
        self.times = times
        self.q = q

    def run(self):
        adv_x_iterative = create_adversarial_sample_MI(self.model, self.input_image, self.input_label, self.epsilon,
                                                       self.times)
        self.q[self.name] = adv_x_iterative


class myThread_2(threading.Thread):
    def __init__(self, threadID, name, model, input_image, input_label, epsilon, times, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model = model
        self.input_image = input_image
        self.input_label = input_label
        self.epsilon = epsilon
        self.times = times
        self.q = q

    def run(self):
        adv_x_iterative = create_adversarial_sample(self.model, self.input_image, self.input_label, self.epsilon,
                                                    self.times)
        self.q[self.name] = adv_x_iterative


image, label = load_ng_data([r'C:\Users\Administrator\Desktop\dataset\images'],
                            r'C:\Users\Administrator\Desktop\dataset\labels')

# pretrained_model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet')
pretrained_model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet')
pretrained_model.trainable = False

epsilons = [0.03]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

I_times_total = []
MI_times_total = []
SAI_times_total = []
for iterative_times in range(2, 3):

    I_times = 0
    MI_times = 0
    SAI_times = 0


    for j in range(160, 233):
        current_img = image[j]
        current_label = pre_process_label(int(label[j]), 1000)
        current_img = current_img[None, ...]
        q = {}
        # plt.figure()
        # plt.imshow(image[j])
        # plt.show()

        for i, eps in enumerate(epsilons):

            """
            p1 = myThread(1, "Thread-iterative", pretrained_model, current_img, current_label, eps, iterative_times + 1, q)
            p1.start()
            p1.join()
            adv_x_iterative = q['Thread-iterative']

            adv_x_iterative = tf.clip_by_value(adv_x_iterative, 0, 1)
            if int(display_images(adv_x_iterative, descriptions[i])) != int(label[j]):
                # print(display_images(adv_x_iterative, descriptions[i]))
                # print(label[j])
                I_times = I_times + 1
            del adv_x_iterative
            # del current_img
            # del current_label
            gc.collect()
            """


            """
            p2 = myThread_1(2, "Thread-MI", pretrained_model, current_img, current_label, eps, iterative_times + 1, q)
            p2.start()
            p2.join()
            adv_x_MI = q['Thread-MI']

            adv_x_MI = tf.clip_by_value(adv_x_MI, 0, 1)
            if int(display_images(adv_x_MI, descriptions[i])) != int(label[j]):
                MI_times = MI_times + 1
                # print(display_images(adv_x_MI, descriptions[i]))
                # print(int(label[j]))
            del adv_x_MI
            gc.collect()
            """


            p3 = myThread_2(3, "Thread-SAI", pretrained_model, current_img, current_label, eps, iterative_times + 1, q)
            p3.start()
            p3.join()
            adv_x = q['Thread-SAI']

            adv_x = tf.clip_by_value(adv_x, 0, 1)
            if int(display_images(adv_x, descriptions[i])) != int(label[j]):
                SAI_times = SAI_times + 1
            del adv_x
            gc.collect()



    # print('===========')

    # I_times_total.append(I_times)
    # print(I_times)

    # MI_times_total.append(MI_times)
    # print(MI_times)

    SAI_times_total.append(SAI_times)
    print(SAI_times)
