import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
from hwdb import HWDB
from mobilenetv3 import MobileNetV3

def valid(epoch, net, test_loarder, writer):
    '''
    测试程序
    :param epoch: 学习次数
    :param net: 网络模型
    :param test_loarder: 测试集
    :param writer: 可视化
    '''
    print("epoch %d 开始验证..." % epoch)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('correct number: ', correct)
        print('totol number:', total)
        acc = 100 * correct / total
        print('第%d个epoch的识别准确率为：%d%%' % (epoch, acc))
        writer.add_scalar('valid_acc', acc, global_step=epoch)


def train(epoch, net, criterion, optimizer, train_loader, writer, scheduler, save_iter=100):
    '''
    :param epoch: 第n次学习
    :param net: 网络模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param train_loader: 训练集
    :param writer: 可视化
    :param scheduler: 调整学习率
    :param save_iter: 每n次保存模型
    :return:
    '''
    print("epoch %d 开始训练..." % epoch)
    net.train()
    sum_loss = 0.0
    total = 0
    correct = 0
    # 数据读取
    for i, (inputs, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()

        # 每训练100个batch打印一次平均loss与acc
        sum_loss += loss.item()
        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            # 每跑完一次epoch测试一下准确率
            acc = 100 * correct / total
            print('epoch: %d, batch: %d loss: %.03f, acc: %.04f'
                  % (epoch, i + 1, batch_loss, acc))
            writer.add_scalar('train_loss', batch_loss, global_step=i + len(train_loader) * epoch)
            writer.add_scalar('train_acc', acc, global_step=i + len(train_loader) * epoch)
            for name, layer in net.named_parameters():
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
            total = 0
            correct = 0
            sum_loss = 0.0
        scheduler.step()



if __name__ == "__main__":
    # 超参数
    epochs = 100 # 学习次数
    batch_size = 128 # batchsize
    lr = 0.1 # 学习率
    data_path = r'/Handwritten_Chinese_character_recognition/data' # 数据集路径
    log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr) # 日志路径
    save_path = r'checkpoints/' # 模型保存路径
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 数据集加载
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = HWDB(path=data_path, transform=transform)
    print("训练集数据:", dataset.train_size)
    print("测试集数据:", dataset.test_size)
    trainloader, testloader = dataset.get_loader(batch_size)

    # 模型加载
    net = MobileNetV3()
    if torch.cuda.is_available():
        net = net.cuda()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    writer = SummaryWriter(log_path)
    for epoch in range(epochs):
        # 训练
        train(epoch, net, criterion, optimizer, trainloader, writer=writer, scheduler=scheduler)
        # 测试
        valid(epoch, net, testloader, writer=writer)
        print("epoch%d 结束, 正在保存模型..." % epoch)
        torch.save(net.state_dict(), save_path + 'handwriting_iter_%03d.pth' % epoch)
