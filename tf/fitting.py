import pathlib
import shutil
import tempfile

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, regularizers

from abstract_model import AbstractModel
from epoch_dots import EpochDots
from plots import HistoryPlotter

"""
说明如何避免过拟合和欠拟合：

1. 在深度学习中，可学习的
   参数数量往往称之为模型的容量（model's capacity）。原因很简单，大容量的模型意味着记忆能力强，容易形成映射。对训练集体现的更多是
   查询，弱化了解决泛化问题的能力。这是深度学习面临的基本问题。
2. 另一方面，容量小一些，映射较难形成，将会要求模型更好学习压缩后的表达方式，相当于提高了抽象能力，更好的解决泛化问题，更好的基于未
   知数据进行预测。而如果若容量过小，不足以将模型抽象出来。
3. 所以本质问题变成了："容量不能过大也不能过小"。怎么把握平衡，这中间没有什么黑魔法，需要通过不同的架构的尝试以及参数调优。

具体方法有以下几种：
1. 架构上。从易到难：避免一开始就使用过于复杂的模型，可以尝试少量的学习参数，减少神经网络的层级和每层的单元数量。比如仅使用Dense Layer，观测
指标，根据需要逐渐使用更加复杂的架构。
2. Learning rate控制。在训练过程中，平滑的逐渐降低learning rate，往往会获得更好的效果，可以使用 optimizers.schedules来持续降低learning rate
3. 迭代次数控制。次数过长也会导致过拟合，建议使用EarlyStopping来避免无意义的训练。可以结合实际需要选择monitor，比如：val_binary_crossentropy,
   val_loss等等。可以通过回调方式嵌入到训练过程中
   
比较方法：
建立了Tiny，Small，Medium，Large四种模型，层数逐渐增多，每层的神经元数量逐渐提升。最终只有Tiny没有出现overfit，但同时也是收敛最慢的。

阻止overfitting的方法：
1. 基本原则——奥卡姆剃须刀。达到相同质量的前提下，模型越简单（容量越小）越合适。
2. 怎么衡量模型是 "简单" 的呢？
   - 参数值得分布具有较小的熵
   - 或者具有更少的参数
3. weight regularization。即强制约束神经网络的参数取较小值以保持较低的复杂度，也就是使权值的分布更加regular。基本思路就是给损失函数
   增加一个扰动，有两种方式：
   - L1 Regularization。正比于所有权重系数的绝对值
   - L2 Regularization。正比于所有权重系数的方差，也称为weight decay
4. dropout。这是神经网络中最优先最广泛使用Regularizaiton方法。随机选择输入数据中的一定比例设置为0，一般设置在 0.2和0.5之间。在测试阶段
   不丢弃数据，而是通过一个相当于dropout rate的因子缩放。
5. 一般而言L2 + Dropout组合是比较好的阻止overfitting的方法。
"""


class FitExample(AbstractModel):
    def __init__(self):
        super(FitExample, self).__init__()
        self.FEATURES = 28
        self.N_VALIDATION = int(1e3)
        self.N_TRAIN = int(1e4)
        self.BUFFER_SIZE = int(1e4)
        self.BATCH_SIZE = 500
        self.STEPS_PER_EPOCH = self.N_TRAIN // self.BATCH_SIZE

    def model_type(self):
        return "fit"

    def model_name(self):
        return "example"

    def run(self):
        # 载入数据作为训练集和验证集
        train_ds, validate_ds, logdir = self.load_data()

        # 建模和训练
        size_histories = {
            'Tiny': self.__compile_fit(self.__get_tiny_model(), logdir, "Tiny", train_ds, validate_ds),  # base line
            'Small': self.__compile_fit(self.__get_small_model(), logdir, "Small", train_ds, validate_ds),  # 增加一层
            'Medium': self.__compile_fit(self.__get_medium_model(), logdir, "Medium", train_ds, validate_ds),  # 再加一层
            'Large': self.__compile_fit(self.__get_large_model(), logdir, "Large", train_ds, validate_ds),
            # 再加一层，并增加每层神经元数
            'Combined': self.__compile_fit(self.__get_combined_model(), logdir, "Combined", train_ds, validate_ds)
        }

        # 展示统计结果
        plotter = HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
        plotter.plot(size_histories)
        a = plt.xscale('log')
        plt.xlim([5, max(plt.xlim())])
        plt.ylim([0.5, 0.7])
        plt.xlabel("Epochs [Log Scale]")
        plt.show()

    def load_data(self):
        """ 加载Higgs数据集
        1. 从网络下载HIGGS.csv，该数据以gzip的方式压缩
        2. 使用批量处理的方法，将feature和label重新整理
        """
        logdir = pathlib.Path(tempfile.mkdtemp())
        shutil.rmtree(logdir, ignore_errors=True)
        gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
        ds = tf.data.experimental.CsvDataset(gz, [float()] * (self.FEATURES + 1), compression_type="GZIP")
        packed_ds = ds.batch(10000).map(FitExample.__pack_row).unbatch()

        # 采样1000个数据展示分布情况
        # for features, label in packed_ds.batch(1000).take(1):
        #     print(features[0])
        #     plt.hist(features.numpy().flatten(), bins=101)
        #     plt.show()

        # 收集验证数据集和训练数据集
        validate_ds = packed_ds.take(self.N_VALIDATION).cache()
        train_ds = packed_ds.skip(self.N_VALIDATION).take(self.N_TRAIN).cache()

        validate_ds = validate_ds.batch(self.BATCH_SIZE)
        train_ds = train_ds.shuffle(self.BUFFER_SIZE).repeat().batch(self.BATCH_SIZE)

        return train_ds, validate_ds, logdir

    @staticmethod
    def __pack_row(*row):
        label = row[0]
        features = tf.stack(row[1:], 1)
        return features, label

    def __get_learning_rate_schedule(self):
        """ 返回schedule以平滑得减少learning rate
        主要目的是获得更好的学习结果，避免过拟合。
        :return:
        """
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,  # 初始learning rate
            decay_steps=self.STEPS_PER_EPOCH * 1000,  # 衰减步长，训练一定量数据后，衰减learning rate
            decay_rate=1,  # 衰减幅度，1表示衰减1倍
            staircase=False  # 阶跃离散式衰减，还是连续衰减，这里使用连续衰减
        )

        # step = np.linspace(0, 100000)
        # lr = lr_schedule(step)
        # plt.figure(figsize=(8, 6))
        # plt.plot(step / self.STEPS_PER_EPOCH, lr)
        # plt.ylim([0, max(plt.ylim())])
        # plt.xlabel('Epoch')
        # _ = plt.ylim('Learning Rate')
        # plt.show()

        return lr_schedule

    @staticmethod
    def __get_callbacks(logdir, name):
        return [
            EpochDots(),
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
            tf.keras.callbacks.TensorBoard(logdir / name)
        ]

    def __compile_fit(self, model, logdir, name, train_ds, validate_ds, optimizer=None, max_epochs=10000):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(self.__get_learning_rate_schedule())

        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[
            tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 'accuracy'
        ])

        model.summary()

        history = model.fit(
            train_ds,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            epochs=max_epochs,
            validation_data=validate_ds,
            callbacks=FitExample.__get_callbacks(logdir, name),
            verbose=0
        )

        return history

    def __get_tiny_model(self):
        return tf.keras.Sequential([layers.Dense(16, activation="elu", input_shape=(self.FEATURES,)),
                                    layers.Dense(1)])

    def __get_small_model(self):
        return tf.keras.Sequential([
            layers.Dense(16, activation='elu', input_shape=(self.FEATURES,)),
            layers.Dense(16, activation='elu'),
            layers.Dense(1)
        ]
        )

    def __get_medium_model(self):
        return tf.keras.Sequential([
            layers.Dense(16, activation='elu', input_shape=(self.FEATURES,)),
            layers.Dense(16, activation='elu'),
            layers.Dense(16, activation='elu'),
            layers.Dense(1)
        ])

    def __get_large_model(self):
        return tf.keras.Sequential([
            layers.Dense(512, activation='elu', input_shape=(self.FEATURES,)),
            layers.Dense(512, activation='elu'),
            layers.Dense(512, activation='elu'),
            layers.Dense(512, activation='elu'),
            layers.Dense(1)
        ])

    def __get_combined_model(self):
        """
        基于L2 Regularization + Dropout的防止overfitting的神经网络
        """
        return tf.keras.Sequential([
            layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu',
                         input_shape=(self.FEATURES,)),
            layers.Dropout(0.5),
            layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu',
                         input_shape=(self.FEATURES,)),
            layers.Dropout(0.5),
            layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu',
                         input_shape=(self.FEATURES,)),
            layers.Dropout(0.5),
            layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu',
                         input_shape=(self.FEATURES,)),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])
