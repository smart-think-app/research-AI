import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.misc import imread

image_md = tf.keras.preprocessing.image

fashion_mnist = tf.keras.datasets.fashion_mnist

num_classes = 10

(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()
print(train_images.shape)
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

# train_images /= 255
# test_images /= 255

train_labels = keras.utils.to_categorical(train_label, num_classes)
test_labels = keras.utils.to_categorical(test_label, num_classes)


def display_sample(num):
    # print(train_labels[num])
    # print(train_images[num].shape)
    label = test_labels[num].argmax(axis=0)
    image = test_images[num, :].reshape([28, 28])

    plt.title("Sample: {} Label: {}".format(num, label))
    plt.imshow(image, cmap=plt.get_cmap("gray_r"))
    plt.show()


# display_sample(1200)

# model.add(keras.layers.Dense(512, activation="relu", input_shape=(784,)))
# model.add(keras.layers.Dense(10, activation="softmax"))
# model.summary()
#
# model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.RMSprop(), metrics=["accuracy"])
# history = model.fit(train_images, train_labels, batch_size=100, epochs=10, verbose=2,
#                     validation_data=(test_images, test_labels))
#
# model.save("my_model")
# x = test_images[1200].reshape(1,784)
img_md = tf.keras.preprocessing.image
# img = img_md.load_img("img_3.png", target_size=(28, 28),color_mode="grayscale")
# x2 = img_md.img_to_array(img)
# x2 = x2.reshape(1,784)
# pre = model.predict(x2)
# print(pre.argmax())
model = tf.keras.models.load_model('my_model')
x1 = test_images[1400, :]
img = img_md.load_img("img.png", target_size=(28, 28), color_mode="grayscale")
x = img_md.img_to_array(img)
x = x.reshape(1,784)
x = 255 - x
pre = model.predict(x)
print(pre.argmax())
# display_sample(1400)
# plt.imshow(x.reshape([28,28]), cmap=plt.get_cmap("gray_r"))
# plt.show()