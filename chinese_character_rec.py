import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from mobilenetv3 import MobileNetV3
# from tensorboardX import SummaryWriter
import logging
import argparse

parse = argparse.ArgumentParser(description='Params for training. ')
# 数据集路径
parse.add_argument('--root', type=str, default='/Handwritten_Chinese_character_recognition/data', help='path to data set')
# 保存模型的路径
parse.add_argument('--log_path', type=str, default=os.path.abspath('.') + '/log.pth', help='dir of checkpoints')
# 是否加载模型
parse.add_argument('--restore', type=bool, default=False, help='whether to restore checkpoints')
# batchsize
parse.add_argument('--batch_size', type=int, default=16, help='size of mini-batch')
# 图片尺寸
parse.add_argument('--image_size', type=int, default=32, help='resize image')
args = parse.parse_args()

# 读取图片和对应的标签
class MyDataset(Dataset):
    def __init__(self, txt_path, transforms=None):
        super(MyDataset, self).__init__()
        images = [] # 图片
        labels = [] # 标签
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)

# test dataset and loader
class test_dataset:
    def __init__(self, txt_path):
        self.images = [] # 图片
        self.labels = [] # 标签
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.images.append(line)
                self.labels.append(int(line.split('/')[-2]))
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = Image.open(self.images[self.index]).convert('RGB')
        label = self.labels[self.index]
        image = self.transform(image).unsqueeze(0)

        self.index += 1
        self.index = self.index % self.size

        return image, label

    def __len__(self):
        return self.size

def trainval():
    # 图片预处理
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor()])
    # 训练集加载
    train_set = MyDataset(args.root + '/train.txt', transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # 测试集加载
    test_set = test_dataset(args.root + '/test.txt')
    test_loader = DataLoader(test_set)
    # 模型加载
    model = MobileNetV3().cuda()
    model = torch.nn.DataParallel(model)
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化策略
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # 开始训练
    for epoch in range(200):
        # 训练
        model.train()
        loss_all = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data[0].cuda(), data[1].cuda()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            # 计算平均损失
            loss_all += loss.data
            if (i+1) % 50 == 0:  # every 200 steps
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, loss_all / 100))
                loss_all = 0
                model.eval()
                correct = 0
                for i in range(test_set.size):
                    inputs, labels = test_set.load_data()
                    with torch.no_grad():
                        outputs = model(inputs)
                    _, predict = outputs.max(1)
                    correct += predict.eq(labels).sum()
                print('Accuracy: %.2f%%' % (correct / test_set.size * 100))
        # 每个epoch保存模型
        print('Save checkpoint...')
        torch.save(model.state_dict(), '/Handwritten_Chinese_character_recognition/res' + 'Net_epoch_{}.pth'.format(epoch + 1))

        # 测试程序
        model.eval()
        with torch.no_grad():
            correct = 0
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].cuda(), data[1]
                outputs = model(inputs)
                _, predict = outputs.cpu().max(1)
                correct += predict.eq(labels).sum()
        print('Accuracy: %.2f%%' % (correct / len(test_loader.dataset) * 100))

        scheduler.step()

# def inference():
#     print('Start inference...')
#     transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
#                                     transforms.Grayscale(),
#                                     transforms.ToTensor()])
#
#     f = open(args.root + '/test.txt')
#     num_line = sum(line.count('\n') for line in f)
#     f.seek(0, 0)
#     line = int(torch.rand(1).data * num_line - 10) # -10 for '\n's are more than lines
#     while line > 0:
#         f.readline()
#         line -= 1
#     img_path = f.readline().rstrip('\n')
#     f.close()
#     label = int(img_path.split('/')[-2])
#     print('label:\t%4d' % label)
#     input = Image.open(img_path).convert('RGB')
#     input = transform(input)
#     input = input.unsqueeze(0)
#     model = NetSmall()
#     model.eval()
#     checkpoint = torch.load(args.log_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     output = model(input)
#     _, pred = torch.max(output.data, 1)
#
#     print('predict:\t%4d' % pred)


# 创建读取数据集路径的txt文件
def classes_txt(root, out_path):
    dirs = os.listdir(root)
    num_class = len(dirs) # 类别数量
    # 创建.txt文件
    if not os.path.exists(out_path):
        f = open(out_path, 'w')
        f.close()
    # 写入txt文件
    with open(out_path, 'r+') as f:
        for dir in dirs:
            files = os.listdir(os.path.join(root, dir))
            for file in files:
                f.write(os.path.join(root, dir, file) + '\n')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    classes_txt(args.root + '/train', args.root + '/train.txt')
    classes_txt(args.root + '/test', args.root + '/test.txt')
    trainval()
    # validation()
    # inference()
