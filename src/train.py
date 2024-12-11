import os

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython.display import clear_output


gpus = tf.config.list_physical_devices("GPU")
print(gpus)

if gpus:
    print("Configuring GPUs")

    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except Exception as e:
        print(e)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis("off")
    plt.savefig(f"./predictions/{len(os.listdir("./predictions"))}.jpeg")


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True,
        )

        self.offset = self.add_weight(
            name="offset", shape=input_shape[-1:], initializer="zeros", trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def upsample(filters, size, norm_type="batchnorm", apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if norm_type.lower() == "batchnorm":
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == "instancenorm":
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


dataset_path = "./processed_dataset"
train_data = []

for folder in os.listdir(dataset_path):
    folder_path = f"{dataset_path}/{folder}"

    for filename in os.listdir(folder_path):
        if filename.startswith("mask_"):
            train_data.append((f"{folder_path}/image.jpg", f"{folder_path}/{filename}"))


def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    return image


@tf.function
def read_train_data(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_jpeg(image, channels=3)
    mask = tf.image.decode_jpeg(mask, channels=3)

    image = normalize(image)
    mask = normalize(mask)

    return image, mask


train_images, train_masks = zip(*train_data)
dataset = tf.data.Dataset.from_tensor_slices((list(train_images), list(train_masks)))
train = dataset.map(read_train_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

TRAIN_LENGTH = len(train_data)
BATCH_SIZE = 32
BUFFER_SIZE = 64
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(
    input_shape=[768, 512, 3], include_top=False
)

layer_names = [
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu",
    "block_16_project",
]
layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

up_stack = [
    upsample(512, 3),
    upsample(256, 3),
    upsample(128, 3),
    upsample(64, 3),
]


def unet_model(output_channels):
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding="same", activation="softmax"
    )

    inputs = tf.keras.layers.Input(shape=[768, 512, 3])
    x = inputs

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# tf.keras.utils.plot_model(model, show_shapes=True)

for image, mask in train.take(2):
    sample_image, sample_mask = image, mask


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...])),
            ]
        )


# show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


EPOCHS = 40
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(train_data) // BATCH_SIZE // VAL_SUBSPLITS

model_history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[DisplayCallback()],
)
