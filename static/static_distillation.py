import paddle
import numpy as np
import paddleslim as slim
paddle.enable_static()

model = slim.models.MobileNet()
student_program = paddle.static.Program()
student_startup = paddle.static.Program()
with paddle.static.program_guard(student_program, student_startup):
    image = paddle.static.data(
        name='image', shape=[None, 3, 32, 32], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    gt = paddle.reshape(label, [-1, 1])
    out = model.net(input=image, class_dim=10)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=gt)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=gt, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=gt, k=5)

teacher_model = slim.models.ResNet50()
teacher_program = paddle.static.Program()
teacher_startup = paddle.static.Program()
with paddle.static.program_guard(teacher_program, teacher_startup):
    with paddle.utils.unique_name.guard():
        image = paddle.static.data(
            name='image', shape=[None, 3, 32, 32], dtype='float32')
        predict = teacher_model.net(image, class_dim=10)
exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(teacher_startup)


# get all student tensor
student_vars = []
for v in student_program.list_vars():
    student_vars.append((v.name, v.shape))
#uncomment the following lines to observe student's tensor for distillation
#print("="*50+"student_model_vars"+"="*50)
#print(student_vars)

# get all teacher tensor
teacher_vars = []
for v in teacher_program.list_vars():
    teacher_vars.append((v.name, v.shape))
#uncomment the following lines to observe teacher's tensor for distillation
#print("="*50+"teacher_model_vars"+"="*50)
#print(teacher_vars)

data_name_map = {'image': 'image'}
main = slim.dist.merge(teacher_program, student_program, data_name_map, paddle.CPUPlace())
with paddle.static.program_guard(student_program, student_startup):
    l2_loss = slim.dist.l2_loss('teacher_bn5c_branch2b.output.1.tmp_3', 'depthwise_conv2d_11.tmp_0', student_program)
    loss = l2_loss + avg_cost
    opt = paddle.optimizer.Momentum(0.01, 0.9)
    opt.minimize(loss)
exe.run(student_startup)

import paddle.vision.transforms as T
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = paddle.vision.datasets.Cifar10(
    mode="train", backend="cv2", transform=transform)

train_loader = paddle.io.DataLoader(
    train_dataset,
    places=paddle.CPUPlace(),
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=True)

for idx, data in enumerate(train_loader):
    acc1, acc5, loss_np = exe.run(student_program, feed=data, fetch_list=[acc_top1.name, acc_top5.name, loss.name])
    print("Acc1: {:.6f}, Acc5: {:.6f}, Loss: {:.6f}".format(acc1.mean(), acc5.mean(), loss_np.mean()))
