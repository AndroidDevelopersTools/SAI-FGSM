import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gc

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


def create_adversarial_sample(model, input_image, input_label, epsilon, times, T_temp, Tmin_temp):
    if epsilon != 0:

        T = T_temp
        Tmin = Tmin_temp
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

            t = t + 2
            T = T_temp / (1 + t)

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


image, label = load_ng_data([r'C:\Users\Administrator\Desktop\dataset\images'],
                            r'C:\Users\Administrator\Desktop\dataset\labels')


pretrained_model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet')
pretrained_model.trainable = False




epsilons = [0.03]
T = [1]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

SAI_times_total = []
for p, k in enumerate(T):
    SAI_times = 0

    for iterative_times in range(2, 3):


        for j in range(160, 300):
            current_img = image[j]
            current_label = pre_process_label(int(label[j]), 1000)
            current_img = current_img[None, ...]


            for i, eps in enumerate(epsilons):

                adv_x = create_adversarial_sample(pretrained_model, current_img, current_label, eps, 3, k, 0.5)
                adv_x = tf.clip_by_value(adv_x, 0, 1)
                if int(display_images(adv_x, descriptions[i])) != int(label[j]):
                    SAI_times = SAI_times + 1

                del adv_x
                gc.collect()

    # SAI_times_total.append(SAI_times)
    print(SAI_times)


# plt.figure()
# plt.plot(T, SAI_times_total, 'bo-', label='SAI value')
# plt.show()
