import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from abstract_model import AbstractModel


class ClassificationDefault(AbstractModel):
    """
    基础的图像分类示例
    """

    def __init__(self):
        super(AbstractModel, self).__init__()

    def model_type(self):
        return "classification"

    def model_name(self):
        return "default"

    def run(self):
        mnist = tf.keras.datasets.mnist

        # 载入训练集和测试集
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # 构建神经网络
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将数据从2维拉平，从 28*28像素变为784的一维数据
            tf.keras.layers.Dense(128, activation='relu'),  # 密集连接神经层，将784维数据连接到128个神经元，使用relu激活函数：小于0为0否则保留
            tf.keras.layers.Dropout(0.2),  # Dropout包括在每次更新期间随机将输入单位的分数rate设置为0,这有助于防止过度拟合。
            # 保留的单位按比例1 / (1 - rate)进行缩放,以便在训练时间和推理时间内它们的总和不变.
            tf.keras.layers.Dense(10, activation='softmax')  # sotfmax将输入缩放到[0,1]区间且保证和为1，常与corssentropy结合用于分类
        ])

        # 编译模型，添加优化器、损失函数、指标
        # 损失函数也就是代价函数，是优化过程的目标，期望最小化。
        # 优化器，决定模型如何根据输入数据和损失函数更新。
        # 指标，用于监控训练和测试步骤。
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # 开始训练
        model.fit(x_train, y_train, epochs=5)

        # 预测
        model.evaluate(x_test, y_test, verbose=2)


class ClassificationFashion(AbstractModel):
    """
    使用更加复杂的数据集，增加预测以及预测结果展示说明
    """

    def __init__(self):
        super(AbstractModel, self).__init__()

    def model_type(self):
        return "classification"

    def model_name(self):
        return "fashion"

    def run(self):
        # 载入数据
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # 图片类别的定义
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # 训练数据展示
        # plt.figure()
        # plt.imshow(train_images[0])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

        # 构建模型
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(10, activation="softmax")
        ])

        # 编译模型：优化器，损失函数，指标
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # 训练
        model.fit(train_images, train_labels, epochs=10)

        # 测试
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        # 预测，输出的结果是一个序列，针对每个test_image输出的预测结果
        # 根据模型的定义，每个预测结果是一个包含10个元素的数组，代表10 个分类。每个元素代表输入数据属于当前分类的概率
        # 那么最终预测的分类应该选择10个分类中概率最高的。
        predictions = model.predict(test_images)

        # 展示预测结果
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            ClassificationFashion.__plot_image(i, predictions[i], test_labels, test_images, class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            ClassificationFashion.__plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
        plt.show()
        print('\nTest accuracy:', test_acc)

    @staticmethod
    def __plot_image(i, predictions_array, true_label, img, class_names):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    @staticmethod
    def __plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        this_plot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        this_plot[predicted_label].set_color('red')
        this_plot[true_label].set_color('blue')
