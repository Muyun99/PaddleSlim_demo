import paddle
import paddle.fluid as fluid
import paddleslim as slim
paddle.enable_static()

exe, train_program, val_program, inputs, outputs = slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=False)

FLOPs = slim.analysis.flops(train_program)
print("FLOPs: {}".format(FLOPs))

pruner = slim.prune.Pruner()
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2_1_sep_weights", "conv2_2_sep_weights"],
        ratios=[0.33] * 2,
        place=fluid.CPUPlace())

FLOPs = slim.analysis.flops(pruned_program)
print("FLOPs: {}".format(FLOPs))


import paddle.dataset.mnist as reader
train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())

for data in train_reader():
    acc1, acc5, loss, _ = exe.run(pruned_program, feed=train_feeder.feed(data), fetch_list=outputs)
    print(acc1, acc5, loss)