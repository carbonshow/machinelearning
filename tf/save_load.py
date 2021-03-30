import os

import tensorflow as tf
from tensorflow import keras

from abstract_model import AbstractModel

"""
主要目的：说明模型的存储和载入。

意义：模型在训练期间以及完成后进行保存，意味着可以从任意中断中恢复，避免耗费较长时间在训练上；也意味着可以分享模型，包括创建模型的代码
以及模型训练的权重(weight)和参数(parameters)

主要方式有：
1. 通过回调自动保存checkpoints。可以设置保存的路径，频率等
2. 手动保存
3. 仅保存权重，保存整个模型
"""


class SaveLoadExample(AbstractModel):
    def __init__(self):
        super(SaveLoadExample, self).__init__()
        self.RECORD_COUNT = 1000
        self.CHECKPOINT_PATH = "checkpoint/save_load.ckpt"

    def model_type(self):
        return "saveload"

    def model_name(self):
        return "default"

    def run(self):
        # 载入数据
        train_images, train_labels, test_images, test_labels = self.load_data()

        # 创建模型
        model = SaveLoadExample.create_model()

        # 初始化checkpoint
        _ = os.path.dirname(self.CHECKPOINT_PATH)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.CHECKPOINT_PATH, save_weights_only=True,
                                                         verbose=1)

        # 使用新的回调训练模型，并会出发checkpoint保存
        model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
                  callbacks=[cp_callback])

        # 创建新的模型，必须和之前的模型保持一致，但可以是不同的实例
        new_model = SaveLoadExample.create_model()
        loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
        print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        # 从checkpoint加载模型权重并重新评估
        new_model.load_weights(self.CHECKPOINT_PATH)
        loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        train_labels = train_labels[:self.RECORD_COUNT]
        test_labels = test_labels[:self.RECORD_COUNT]

        train_images = train_images[:self.RECORD_COUNT].reshape(-1, 28 * 28) / 255.0
        test_images = test_images[:self.RECORD_COUNT].reshape(-1, 28 * 28) / 255.0

        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def create_model():
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model
