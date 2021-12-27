import tensorflow as tf
import matplotlib.pyplot as plt

# img_md = tf.keras.preprocessing.image
# img = img_md.load_img("images_products/img_1.png")
# img = img_md.img_to_array(img)

ImageDataGen = tf.keras.preprocessing.image.ImageDataGenerator

train = ImageDataGen(rescale=1. / 255)
validation = ImageDataGen(rescale=1. / 255)

train_dataset = train.flow_from_directory("images_products/train/", target_size=(224, 224),
                                          shuffle=True)

validation_dataset = train.flow_from_directory("images_products/test/", target_size=(224, 224),
                                               shuffle=True)

a = train_dataset.class_indices
print(a)
classes = train_dataset.classes
print(classes)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
#                            input_shape=(224, 224, 3)),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#     tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=0.0001),
#     loss="categorical_crossentropy",
#     metrics=['accuracy']
# )
# print(model.summary())
#
# history = model.fit(train_dataset,
#                     steps_per_epoch=train_dataset.samples,
#                     epochs=10,
#                     validation_data=validation_dataset,
#                     validation_steps=validation_dataset.samples,
#                     verbose=1
#                     )


# model.save("my_model_custom")
img_md = tf.keras.preprocessing.image
model = tf.keras.models.load_model('my_model_custom')
img = img_md.load_img("img_4.png", target_size=(224, 224))
x = img_md.img_to_array(img)
x = x.reshape(1, 224, 224, 3)
pre = model.predict(x)
print(pre.argmax())
