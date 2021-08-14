import paddle
import paddle.vision.models as models
from paddle.static import InputSpec as Input
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from paddleslim.dygraph import L1NormFilterPruner

net = models.mobilenet_v1(pretrained=False, scale=1.0, num_classes=10)
inputs = [Input([None, 3, 32, 32], 'float32', name='image')]
labels = [Input([None, 1], 'int64', name='label')]
optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=net.parameters())
model = paddle.Model(net, inputs, labels)
model.prepare(
        optimizer,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

transform = T.Compose([
                    T.Transpose(),
                    T.Normalize([127.5], [127.5])
                ])

val_dataset = Cifar10(mode='test', transform=transform)
train_dataset = Cifar10(mode='train', transform=transform)

model.fit(train_dataset, epochs=2, batch_size=128, verbose=1)
model.evaluate(val_dataset, batch_size=128, verbose=1)


pruner = L1NormFilterPruner(net, [1, 3, 32, 32])
pruner.prune_vars({'conv2d_22.w_0':0.5, 'conv2d_20.w_0':0.6}, axis=0)

Prue_FLOPs = paddle.flops(net, input_size=[1, 3, 32, 32], print_detail=True)
model.fit(train_dataset, epochs=1, batch_size=128, verbose=1)
model.evaluate(val_dataset, batch_size=128, verbose=1)
