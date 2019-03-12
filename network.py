import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

"""
=================================================
        CREATE NETWORK CNN + LSTM
=================================================
"""


class PretrainedCNN(nn.Module):
    def __init__(self, in_img_dim, out_dim):
        super(PretrainedCNN, self).__init__()

        self.in_dim = in_img_dim

        # input image size: (224,224)
        self.net = torchvision.models.resnet50(pretrained=True)  # 18, 34, 50

        n_features = self.net.fc.in_features
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        x = self.resize(x)
        return self.net(x)

    def resize(self,frame):
        return F.interpolate(frame, size=(224, 224), mode='nearest')


class CustomCNN(nn.Module):
    def __init__(self, in_image_dim, characteristic_dim):
        super(CustomCNN, self).__init__()

        self.in_img_width = in_image_dim
        now_dim = self.in_img_width

        # output_dim
        self.output_dim = characteristic_dim

        # feature vec dim
        self.i_t_dim = characteristic_dim - 5

        # feature and location combo dim
        self.o_t_dim = characteristic_dim

        # hidden dimension
        self.hidden_dim = characteristic_dim

        # convolution 1
        conv_channels_1 = 16
        conv_kernel_1 = 5
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=conv_channels_1,
                              kernel_size=conv_kernel_1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        now_dim -= (conv_kernel_1 - 1)

        # pooling 1
        pool_kernel_1 = 2
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_1)
        now_dim = now_dim // 2

        # convolution 2
        conv_channels_2 = 32
        conv_kernel_2 = 3
        self.cnn2 = nn.Conv2d(in_channels=conv_channels_1, out_channels=conv_channels_2,
                              kernel_size=conv_kernel_2, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        now_dim -= (conv_kernel_2 - 1)

        # polling 2
        pool_kernel_2 = 2
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_2)
        now_dim = now_dim // 2

        # fully connected layer
        now_dim = conv_channels_2 * (now_dim ** 2)
        self.fc_cnn = nn.Linear(now_dim, self.i_t_dim)

    def forward(self, x):

        # convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # pooling 1
        out = self.pool1(out)

        # convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # pooling 2
        out = self.pool2(out)

        # Resize
        # Original size: (100, 32, 5, 5)
        # New out size: (100, 32*5*5)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        return self.fc_cnn(out)


class CNN_LSTM(nn.Module):
    def __init__(self, in_image_dim, characteristic_dim, use_gpu):
        super(CNN_LSTM, self).__init__()

        self.use_gpu = use_gpu

        #self.cnn = CustomCNN(in_image_dim, characteristic_dim)
        self.cnn = PretrainedCNN(in_image_dim, characteristic_dim - 5)

        if use_gpu:
            self.cnn = self.cnn.cuda()

        self.hidden_dim = characteristic_dim

        """
        Input: seq_length x batch_size x input_size
        Output: seq_length x batch_size x hidden_size
        """
        # batch_first=True causes input/output tensors
        # to be of shape (batch_dim, seq_dim, feature_dim)
        # layer_dim : number of lstm layers
        self.lstm = nn.LSTM(characteristic_dim, self.hidden_dim, 1, batch_first=True)

        self.clear_memory()

    def forward(self, x, gt):
        # initialize hidden/cell state with zeros
        # since x is a vec, size(0) yield his len
        # 1 is batch size

        self.h = self.h.detach()
        self.c = self.c.detach()

        out = self.cnn(x)

        # concatenates ground_truth
        out = torch.cat((out, gt), 1)

        # batch_size = 1 in our case
        out = out.unsqueeze(0)

        # lstm
        out, (self.h, self.c) = self.lstm(out, (self.h, self.c))

        # rid of first batch_dim
        # out = self.fc_rnn(out[-1, :])
        out = out[-1, :]

        # slice last four elements
        out = out[:, -5:]

        return out

    def clear_memory(self):
        if self.use_gpu:
            self.h = torch.zeros(1, 1, self.hidden_dim).cuda()
            self.c = torch.zeros(1, 1, self.hidden_dim).cuda()
        else:
            self.h = torch.zeros(1, 1, self.hidden_dim)
            self.c = torch.zeros(1, 1, self.hidden_dim)


if __name__=="__main__":
    im_size = 64
    net = PretrainedCNN(im_size, 100)
    x = torch.randn(5, 3, 64, 64)
    ans = net(x)
    print(ans.size())

