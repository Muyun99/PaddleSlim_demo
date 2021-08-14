import paddle
import paddle.vision.models as models
from paddle.static import InputSpec as Input
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from paddleslim.dygraph.quant import QAT

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
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = Cifar10(mode='train', backend='cv2', transform=transform)
val_dataset = Cifar10(mode='test', backend='cv2', transform=transform)

model.fit(train_dataset, epochs=5, batch_size=256, verbose=1)
model.evaluate(val_dataset, batch_size=256, verbose=1)

paddle.jit.save(net, "./fp32_inference_model", input_spec=[inputs])

paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir='./',
        model_filename='fp32_inference_model.pdmodel',
        params_filename='fp32_inference_model.pdiparams',
        quantize_model_path='./quant_post_static_model',
        sample_generator=train_dataset,
        batch_nums=10)