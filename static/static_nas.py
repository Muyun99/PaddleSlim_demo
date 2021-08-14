import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.static as static
import paddleslim as slim
import numpy as np

sanas = slim.nas.SANAS(configs=[('MobileNetV2Space')], server_addr=("", 8337), save_checkpoint=None)

paddle.enable_static()
def build_program(archs):
    train_program = static.Program()
    startup_program = static.Program()
    with static.program_guard(train_program, startup_program):
        data = static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
        label = static.data(name='label', shape=[None, 1], dtype='int64')
        gt = paddle.reshape(label, [-1, 1])
        output = archs(data)
        output = static.nn.fc(output, size=10)

        softmax_out = F.softmax(output)
        cost = F.cross_entropy(softmax_out, label=label)
        avg_cost = paddle.mean(cost)
        acc_top1 = paddle.metric.accuracy(input=softmax_out, label=gt, k=1)
        acc_top5 = paddle.metric.accuracy(input=softmax_out, label=gt, k=5)
        test_program = static.default_main_program().clone(for_test=True)

        optimizer = paddle.optimizer.Adam(learning_rate=0.1)
        optimizer.minimize(avg_cost)

        place = paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(startup_program)
    return exe, train_program, test_program, (data, label), avg_cost, acc_top1, acc_top5

import paddle.vision.transforms as T

def input_data(image, label):
    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    train_dataset = paddle.vision.datasets.Cifar10(mode="train", transform=transform, backend='cv2')
    train_loader = paddle.io.DataLoader(train_dataset,
                    places=paddle.CPUPlace(),
                    feed_list=[image, label],
                    drop_last=True,
                    batch_size=64,
                    return_list=False,
                    shuffle=True)
    eval_dataset = paddle.vision.datasets.Cifar10(mode="test", transform=transform, backend='cv2')
    eval_loader = paddle.io.DataLoader(eval_dataset,
                    places=paddle.CPUPlace(),
                    feed_list=[image, label],
                    drop_last=False,
                    batch_size=64,
                    return_list=False,
                    shuffle=False)
    return train_loader, eval_loader

def start_train(program, data_loader):
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_loader():
        batch_reward = exe.run(program, feed=data, fetch_list = outputs)
        print("TRAIN: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))

def start_eval(program, data_loader):
    reward = []
    outputs = [avg_cost.name, acc_top1.name, acc_top5.name]
    for data in data_loader():
        batch_reward = exe.run(program, feed=data, fetch_list = outputs)
        reward_avg = np.mean(np.array(batch_reward), axis=1)
        reward.append(reward_avg)
        print("TEST: loss: {}, acc1: {}, acc5:{}".format(batch_reward[0], batch_reward[1], batch_reward[2]))
    finally_reward = np.mean(np.array(reward), axis=0)
    print("FINAL TEST: avg_cost: {}, acc1: {}, acc5: {}".format(finally_reward[0], finally_reward[1], finally_reward[2]))
    return finally_reward


# archs = sanas.next_archs()[0]
# exe, train_program, eval_program, (image, label), avg_cost, acc_top1, acc_top5 = build_program(archs)
# train_loader, eval_loader = input_data(image, label)
# start_train(train_program, train_loader)
# finally_reward = start_eval(eval_program, eval_loader)
# sanas.reward(float(finally_reward[1]))

for step in range(3):
    archs = sanas.next_archs()[0]
    exe, train_program, eval_program, (image, label), avg_cost, acc_top1, acc_top5 = build_program(archs)
    train_loader, eval_loader = input_data(image, label)

    current_flops = slim.analysis.flops(train_program)
    if current_flops > 321208544:
        continue

    for epoch in range(7):
        start_train(train_program, train_loader)

    finally_reward = start_eval(eval_program, eval_loader)

    sanas.reward(float(finally_reward[1]))