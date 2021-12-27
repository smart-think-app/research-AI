import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
fashion_mnist = tf.keras.datasets.fashion_mnist

new_model = tf.keras.models.load_model('my_model')
img_md = tf.keras.preprocessing.image

(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()
print(train_images.shape)
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images /= 255
test_images /= 255


def display_sample(img_display):
    # img_display = img_md.load_img("img_3.png", target_size=(28, 28))
    plt.imshow(img_display, cmap=plt.get_cmap("gray_r"))
    plt.show()


# display_sample("img_3.png")
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# img = img_md.load_img("img_3.png", target_size=(28, 28),color_mode="grayscale")
# x = img_md.img_to_array(img)
# x = x.astype("float32")
# x /= 255
# x = x.reshape(1,784)
x = test_images[1200].reshape(1,784)
pre = new_model.predict(x)
print(pre.argmax())

# display_sample(img)