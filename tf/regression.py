import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from abstract_model import AbstractModel


class PrintDot(keras.callbacks.Callback):
    """ 训练过程展示进度的回调
    通过为每个完成的时期打印一个点来显示训练进度
    """
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


class RegressionDefault(AbstractModel):

    def __init__(self, epochs=1000):
        """ 初始化，设置迭代次数
        :param epochs: 训练迭代次数
        """
        super(RegressionDefault, self).__init__()
        self.epochs = epochs

    def model_type(self):
        return "regression"

    def model_name(self):
        return "default"

    def run(self):
        # 载入训练数据集和测试数据集
        train_dataset, test_dataset = RegressionDefault.prepare_data()

        # 数据预处理
        normed_train_data, normed_test_data, train_labels, test_labels = RegressionDefault.preprocess_data(
            train_dataset, test_dataset)

        # 创建模型
        model = RegressionDefault.build_model(train_dataset)
        print("model summary\n{}".format(model.summary()))

        # patience 值用来检查改进 epochs 的数量.
        # 'val_loss' 表示关注cost function的值，也就是代价函数的值
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(normed_train_data, train_labels, epochs=self.epochs,
                            validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

        # 普通的训练迭代次数方法，在一定次数后可能导致误差越来越大，因此建议使用EarlyStopping方法
        # history = model.fit(
        #     normed_train_data, train_labels,
        #     epochs=EPOCHS, validation_split=0.2, verbose=0,
        #     callbacks=[PrintDot()])

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        RegressionDefault.plot_history(history)

        # 训练结果评估
        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

        # 预测
        RegressionDefault.predict(model, normed_test_data, test_labels)

    @staticmethod
    def prepare_data():
        """ 准备训练集和测试集
        1. 下载数据并导入pandas用dataframe表示
        2. 添加USA，Europe，Japan三列说明数据来源地
        3. 随机分拆数据，80%用于训练集，其余用于测试集
        4. 作图展示不同columns的关系
        """
        # 下载数据并导入到pandas中
        dataset_path = keras.utils.get_file("auto-mpg.data",
                                            "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

        # 列名
        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                        'Acceleration', 'Model Year', 'Origin']
        raw_dataset = pd.read_csv(dataset_path, names=column_names,
                                  na_values="?", comment='\t',
                                  sep=" ", skipinitialspace=True)
        dataset = raw_dataset.copy().dropna()  # 去除无效值

        # 根据Origin列的值，将USA，EUROPE，JAPAN作为单独列添加，便于处理，如果某一列非0表示就是这个地区的数据
        origin = dataset["Origin"]
        dataset['USA'] = (origin == 1) * 1.0
        dataset['Europe'] = (origin == 2) * 1.0
        dataset['Japan'] = (origin == 3) * 1.0

        # 拆分训练集和测试集，随机将原始数据集中的80%作为训练集，剩下的作为测试集
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        # 快速查看几个列的联合分布，也就是大致了解它们之间的关系，基本上就是根据叉乘几个列，展示不同列的数据对应关系
        # 对角线也就是同一列自己可以指定显示方式，比如kde，hist等。
        sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

        return train_dataset, test_dataset

    @staticmethod
    def preprocess_data(train_dataset, test_dataset):
        """ 预处理数据
        1. 对训练集进行z-score形变
        2. 将MPG列弹出作为标签列
        :param train_dataset: 训练集
        :param test_dataset: 测试集
        :return: 标准化之后的训练集、测试集，以及训练集标签、测试集标签
        """
        # 查看总体分布，主要是为了计算mean和dev，方便z-score计算
        train_stats = train_dataset.describe()
        print("describe train_stats")
        print(train_stats)
        train_stats.pop("MPG")  # 删除MPG这一列
        train_stats = train_stats.transpose()  # 转置
        print(train_stats)

        # 将特征值从目标值或者"标签"中分离。 这个标签是使用训练模型进行预测的值。
        train_labels = train_dataset.pop('MPG')
        test_labels = test_dataset.pop('MPG')
        normed_train_data = RegressionDefault.norm(train_dataset, train_stats)
        normed_test_data = RegressionDefault.norm(test_dataset, train_stats)

        return normed_train_data, normed_test_data, train_labels, test_labels

    @staticmethod
    def norm(x, train_stats):
        """ 数据标准化处理
        z-score方法
        :param x: 输入样本值
        :param train_stats: 训练结果
        :return: 规范化处理的结果
        """
        return (x - train_stats['mean']) / train_stats['std']

    @staticmethod
    def build_model(train_dataset):
        """
        根据训练数据集，返回训练模型
        :param train_dataset:  训练数据集
        :return: 模型
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    @staticmethod
    def plot_history(history):
        """ 展示训练过程
        1. 作图展示Train Error和Val Error
        2. Train Error，mae(Mean Absolute Error)，即预测值和实际标签的绝对误差的均值。MSE比MAE更快收敛，但MAE相对Outlier更健壮。
        3. Val Error，val_mae，Validation MAE，也就是验证的误差，同样使用了mae。将训练集中的一部分用来验证。
        :param history: 训练结果
        """
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
                 label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()

    @staticmethod
    def predict(model, normed_test_data, test_labels):
        """ 针对测试集预测
        图示展示
        :param model: 训练好的模型
        :param normed_test_data: 标准化之后的测试集
        :param test_labels: 测试标签，实际上是连续值
        """
        # 预测
        test_predictions = model.predict(normed_test_data).flatten()

        # 预测MPG
        plt.scatter(test_labels, test_predictions)
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])

        # 查看误差分布
        error = test_predictions - test_labels
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [MPG]")
        _ = plt.ylabel("Count")
        plt.show()
