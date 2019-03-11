import torch.nn as nn
import torch


"""
=================================================
        CREATE NETWORK CNN + LSTM
=================================================
"""


class CNN_LSTM(nn.Module):
    def __init__(self, in_image_dim, characteristic_dim):
        super(CNN_LSTM, self).__init__()

        # we further assume that they are equal
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

        # number of lstm layers
        self.layer_dim = 1

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
        self.fc_cnn = nn.Linear(now_dim, self.i_t_dim, bias=True)

        """
        Input: seq_length x batch_size x input_size
        Output: seq_length x batch_size x hidden_size
        """
        # batch_first=True causes input/output tensors
        # to be of shape (batch_dim, seq_dim, feature_dim)
        # layer_dim : number of lstm layers
        self.lstm = nn.LSTM(self.o_t_dim, self.hidden_dim,
                            self.layer_dim, batch_first=True)

        self.h_0 = self.c_0 = None

        # Readout layer
        # self.fc_rnn = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, x, ground_truth, is_new, use_gpu):
        # initialize hidden/cell state with zeros
        # since x is a vec, size(0) yield his len
        # 1 is batch size

        if is_new:
            if use_gpu:
                h_0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda()
                c_0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda()
            else:
                h_0 = torch.zeros(self.layer_dim, 1, self.hidden_dim)
                c_0 = torch.zeros(self.layer_dim, 1, self.hidden_dim)
        else:
            h_0 = self.h_0.detach()
            c_0 = self.c_0.detach()

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
        out = self.fc_cnn(out)

        # concatenates ground_truth
        out = torch.cat((out, ground_truth), 1)

        # batch_size = 1 in our case
        out = out.unsqueeze(0)

        # lstm
        out, (self.h_0, self.c_0) = self.lstm(out, (h_0, c_0))

        # rid of first batch_dim
        # out = self.fc_rnn(out[-1, :])
        out = out[-1, :]

        # slice last four elements
        out = out[:, -5:]

        return out


if __name__=="__main__":
    im_size = (64, 64)
    net = CNN_LSTM(in_image_dim=im_size, characteristic_dim=100)
    print(net)

