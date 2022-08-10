from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import MobileNetV1
import tensorflow as tf
import json
import os
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    #         exit(-1)

    data_root = 'C:\\Users\\Lunatic Tear\\PycharmProjects\\GoogLeNet'
    image_path = data_root + "\\data_set\\flower_data\\"
    train_dir = image_path + "train"
    validation_dir = image_path + "val"
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 224
    im_width = 224
    batch_size = 10
    epochs = 65

    def pre_function(img):
        # img = im.open('test.jpg')
        # img = np.array(img).astype(np.float32)
        img = img / 255.
        img = (img - 0.5) * 2.0

        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                               horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    model = MobileNetV1(im_height=im_height, im_width=im_width, class_num=5)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myMobileNet.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    # # using keras low level api for training
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # val_loss = tf.keras.metrics.Mean(name='val_loss')
    # val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         output = model(images, training=True)
    #         loss = loss_object(labels, output)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, output)
    #
    # @tf.function
    # def val_step(images, labels):
    #     output = model(images, training=False)
    #     loss = loss_object(labels, output)
    #
    #     val_loss(loss)
    #     val_accuracy(labels, output)
    #
    # best_val_acc = 0.
    # for epoch in range(epochs):
    #     train_loss.reset_states()  # clear history info
    #     train_accuracy.reset_states()  # clear history info
    #     val_loss.reset_states()  # clear history info
    #     val_accuracy.reset_states()  # clear history info
    #
    #     # train
    #     train_bar = tqdm(range(total_train // batch_size), file=sys.stdout)
    #     for step in train_bar:
    #         images, labels = next(train_data_gen)
    #         train_step(images, labels)
    #
    #         # print train process
    #         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
    #                                                                              epochs,
    #                                                                              train_loss.result(),
    #                                                                              train_accuracy.result())
    #
    #     # validate
    #     val_bar = tqdm(range(total_val // batch_size), file=sys.stdout)
    #     for step in val_bar:
    #         val_images, val_labels = next(val_data_gen)
    #         val_step(val_images, val_labels)
    #
    #         # print val process
    #         val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
    #                                                                            epochs,
    #                                                                            val_loss.result(),
    #                                                                            val_accuracy.result())
    #
    #     # only save best weights
    #     if val_accuracy.result() > best_val_acc:
    #         best_val_acc = val_accuracy.result()
    #         model.save_weights("./save_weights/myMobileNet.ckpt")


if __name__ == '__main__':
    main()
