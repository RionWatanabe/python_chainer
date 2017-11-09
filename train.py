import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets import cifar
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from Converter import make_dataset

from Model import MyModel


def train(model_object, batchsize=64, gpu_id=0, max_epoch=5, train_dataset=None, test_dataset=None):

    # define dataset
    if train_dataset is None and test_dataset is None:
        train, test = make_dataset()
    else:
        train, test = train_dataset, test_dataset

    # Iterator
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    # Model
    model = L.Classifier(model_object)
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    # Optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    #Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_result'.format(model_object.__class__.__name__))

    # Evaluator
    class TestModeEvaluator(extensions.Evaluator):

        def evaluate(self):
            model = self.get_target('main')
            model.train = False
            ret = super(TestModeEvaluator, self).evaluate()
            model.train = True
            return ret

    trainer.extend(extensions.LogReport())
    trainer.extend(TestModeEvaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.run()
    del trainer

    return model


model = train(MyModel(10), gpu_id=-1)
