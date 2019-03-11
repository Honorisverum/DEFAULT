"""
=================================================
                TEST PHASE
    COMPUTE FORWARD PASS ON TEST VIDEOS
=================================================
"""

import utils
import torch


REINITIALIZE_GAP = 4
ACCURACY_CALC_GAP = 10


def compute_tests(testing_set_videos, net, pad_len, use_gpu):

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net.eval()

    for video in testing_set_videos:

        print("testing {}".format(video.title))

        is_initialize = True
        reinitialize_wait = False
        wait_frame = 0
        accuracy_wait_frames = 0

        gt_loader = video.create_gt_iter(1)
        frames_loader = video.create_frames_iter(1)

        for gt, frames in zip(gt_loader, frames_loader):

            # check if it is still waiting time
            if reinitialize_wait:
                wait_frame -= 1
                accuracy_wait_frames -= 1
                if wait_frame == 0:
                    reinitialize_wait = False
                continue

            # calculate output
            if is_initialize:
                init_frame_pad = torch.cat([frames]*pad_len, 0)
                init_gt_pad = torch.cat([gt]*pad_len, 0)
                _ = net(init_frame_pad, init_gt_pad, True, use_gpu)
                output = net(frames, gt, False, use_gpu)
            else:
                output = net(frames, torch.zeros(1, 5), False, use_gpu)

            # calculate overlap
            overlap = utils.reward2(output.view(-1), gt.view(-1))

            # check if overlap is 0, else add to the answer
            if overlap == 0:
                video.test_fails += 1
                reinitialize_wait = True
                wait_frame = REINITIALIZE_GAP
                accuracy_wait_frames = ACCURACY_CALC_GAP
            else:
                if accuracy_wait_frames != 0:
                    accuracy_wait_frames -= 1
                    continue
                video.test_predictions.append(overlap)
                video.non_fail_frames += 1


