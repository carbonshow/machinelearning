import tensorflow as tf


class EpochDots(tf.keras.callbacks.Callback):
    """A simple callback that prints a "." every epoch, with occasional reports.
    Args:
      report_every: How many epochs between full reports
      dot_every: How many epochs between dots.
    """

    def __init__(self, report_every=100, dot_every=1):
        super().__init__()
        self.report_every = report_every
        self.dot_every = dot_every

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_every == 0:
            print()
            print('Epoch: {:d}, '.format(epoch), end='')
            for name, value in sorted(logs.items()):
                print('{}:{:0.4f}'.format(name, value), end=',  ')
            print()

        if epoch % self.dot_every == 0:
            print('.', end='')
