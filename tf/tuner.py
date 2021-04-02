import kerastuner as kt
import tensorflow as tf
from tensorflow import keras

from abstract_model import AbstractModel

"""
依赖keras-tuner，这是一个协助hyper-parameters调优的库。超参数和模型参数不同，模型参数是在学习过程中得到的；超参数影响机器学习的整个
过程以及模型拓扑，这些参数在整个训练过程中是常量。keras-tuner则用来对这些参数调优。适合于：
- 模型参数。影响隐藏层（hidden layers）的数量，宽度（神经元数量）等
- 算法参数。影响学习的速度和质量

本例中对以下学习过程的参数进行调优：
- 神经层的宽度，即神经数量。在指定范围内调优。
- learning rate。在指定候选项中调优
- 训练迭代次数。即Epoch
"""


class TunerExample(AbstractModel):
    def __init__(self):
        super(TunerExample, self).__init__()

    def model_type(self):
        return "tuner"

    def model_name(self):
        return "default"

    def run(self):
        # 载入数据
        img_train, label_train, img_test, label_test = TunerExample.load_data()

        # 创建调优器，在create_model中需要设置基于hp要调优的目标
        tuner = kt.Hyperband(TunerExample.create_model, objective='val_accuracy', max_epochs=10, factor=3,
                             directory='tuner',
                             project_name='_'.join([self.model_type(), self.model_name()]
                                                   ))
        # 创建回调
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # 运行hyper-parameter搜索，寻找最优的超参数
        tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

        # 获得最优hyper-parameter
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""
        The hyper-parameter search is complete. The optimal number of units in the first densely-connected layer is
        {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}
        """)

        # 使用最优hyper-parameter训练模型，寻找最优的epoch，也就是验证精度最高
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # 使用得到的best_epoch继续训练
        hypermodel = tuner.hypermodel.build(best_hps)
        hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
        eval_result = hypermodel.evaluate(img_test, label_test)
        print("[test loss, test accuracy]:", eval_result)

    @staticmethod
    def load_data():
        (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
        img_train = img_train.astype('float32') / 255.0
        img_test = img_test.astype('float32') / 255.0
        return img_train, label_train, img_test, label_test

    @staticmethod
    def create_model(hp):
        """ 创建训练模型，指定需要调优的参数
        对Dense Layer的宽度，以及学习率设置调优
        :param hp: hyper parameters
        :return: 支持调优的模型
        """
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        # 调整第一全连接层的神经元数量，在32-512之间选择一个最优值
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(keras.layers.Dense(10))

        # 调整优化器的学习速率，从0.01，0.001，0.0001中选择一个最优值
        hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model
