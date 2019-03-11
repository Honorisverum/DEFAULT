import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os

"""
=================================================
        LOADING FRAMES AND GROUND TRUTH
=================================================
"""

# get current path
CWD = os.getcwd()


# normalization by image height and weight
def gt_normalize(gt, w, h):
    gt[:, 0] /= w
    gt[:, 1] /= h
    gt[:, 2] /= w
    gt[:, 3] /= h
    return gt


# class for store each video
class VideoBuffer(object):

    def __init__(self, title, height, width, gt, frames, len, T):
        self.title = title
        self.height = height
        self.width = width
        self.len = len
        self.ground_truth = gt
        self.frames = frames
        self.test_fails = 0
        self.test_predictions = []
        self.non_fail_frames = 0

    def create_gt_iter(self, T):
        return data_iter(self.ground_truth, T)

    def create_frames_iter(self, T):
        return data_iter(self.frames, T)


def load_videos(titles_list, T, img_size, use_gpu, vid_dir):

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    DIR = CWD if vid_dir is None else vid_dir

    # all roots to all videos is list
    roots_list = [DIR + "/videos/" + x for x in titles_list]

    # Resize    : to image_size
    # Transform : to torch tensor
    # Normalize : mean and std for 3 channels
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # set of videos, could be Training or Test
    videos_list = []

    for video_title, root in zip(titles_list, roots_list):

        print("video: {}...".format(video_title))

        # load frames to torch dataset
        frame_dataset = datasets.ImageFolder(root=root, transform=transform)

        # extract full set as holistic tensor
        all_frames = to_tensor(frame_dataset)

        # gpu
        if use_gpu:
            all_frames = all_frames.cuda()

        # load_info
        info_file = root + "/{}.txt".format(video_title)
        with open(info_file) as f:
            info_lines = f.readlines()
            # delete '\n'
            for char in '\n':
                info_lines[0] = info_lines[0].replace(char, "")
                info_lines[1] = info_lines[1].replace(char, "")
            sizes = list(info_lines[0].split(" "))
            x_size = int(sizes[0])
            y_size = int(sizes[1])
            frame_num = int(info_lines[1])
            # gt_info = info_lines[2]

        # load ground truth
        gt_path = root + "/groundtruth.txt"
        gt_txt = np.loadtxt(gt_path, delimiter=',', dtype=np.float32)
        gt_tens = torch.from_numpy(gt_txt)
        if use_gpu:
            gt_tens = gt_tens.cuda()
        gt_tens = gt_normalize(gt_tens, x_size, y_size)


        # add to set
        vid = VideoBuffer(title=video_title, height=y_size,
                          width=x_size, gt=gt_tens,
                          frames=all_frames, len=frame_num, T=T)
        videos_list.append(vid)

    return videos_list


def to_tensor(dataset):
    lst = []
    for i in range(dataset.__len__()):
        lst.append(dataset[i][0].unsqueeze(0))
    return torch.cat(lst, dim=0)


def data_iter(tensor, T):
    len = tensor.size(0)
    i = 0
    while i < len:
        yield tensor[i:i+T]
        i += T


if __name__ == "__main__":
    """
    Some tests:
    """
    use_gpu = False
    T = 25

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    titles_list = ['Vid_B_cup']
    img_size = (64, 64)

    roots_list = [CWD + "/videos/" + x for x in titles_list]

    print(roots_list)

    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    frame_set = datasets.ImageFolder(root=roots_list[0], transform=transform)

    full_set = to_tensor(frame_set)

    if use_gpu:
        full_set = full_set.cuda()

    print(full_set.size())

    it = data_iter(full_set, T)

    for i in it:
        print(i.size())


