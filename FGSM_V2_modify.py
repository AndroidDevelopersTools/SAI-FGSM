import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False


def pre_process_image(name, link, size):
    image_path = tf.keras.utils.get_file(name, link)
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, size)
    image = image[None, ...]
    return image


def from_probability_get_label(model, image):
    image_probs = model.predict(image)
    return tf.keras.applications.densenet.decode_predictions(image_probs, top=1)[0][0]


loss_temp = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_sample(model, input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_temp(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad



def pre_process_label(index, label_shape):
    label = tf.one_hot(index, label_shape)
    label = tf.reshape(label, (1, label_shape))
    return label



def display_images(image, description):
    _, label, confidence = from_probability_get_label(tf.keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet'), image)
    plt.figure()
    plt.imshow(image[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
    plt.show()




image = pre_process_image('YellowLabradorLooking_new.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg',
            (224, 224))


pretrained_model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet')
pretrained_model.trainable = False


_, image_class, class_confidence = from_probability_get_label(pretrained_model, image)

# samples = image * 255.
# samples = tf.cast(samples, tf.uint8)
plt.imshow(image[0])
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()



label = pre_process_label(208, 1000)




perturbations = create_adversarial_sample(pretrained_model, image, label)


# 扰动大小和图片大小完全一致，因为扰动是对图片的每一个像素求梯度
plt.imshow(perturbations[0])
plt.show()



epsilons = [0, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]


for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    display_images(adv_x, descriptions[i])

