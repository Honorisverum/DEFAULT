"""
=================================================
                TEST PHASE
    COMPUTE FORWARD PASS ON TEST VIDEOS
=================================================
"""

import utils
import torch
import numpy as np


REINITIALIZE_GAP = 4
ACCURACY_CALC_GAP = 9
INFO_STRING = "Title: {video_title} | " \
              "Mean Reward: {mean_reward} | " \
              "Number of frames: {n_frames} | " \
              "Number of fails: {n_fails} |"
OVERALL_INFO_STRING = "Total Mean overlap: " \
                      "{overlap} |" \
                      "Total Robustness: " \
                      "{robustness}"


def compute_tests(testing_set_videos, net, pad_len, use_gpu):

    print("Validation:")

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net.eval()

    all_overlaps, all_fails, all_frames = [], 0, 0

    for video in testing_set_videos:

        is_initialize = True
        reinitialize_wait = False
        wait_frame = 0
        accuracy_wait_frames = 0

        for gt, frames in video.get_dataloader(1):

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
                video.n_fails += 1
                reinitialize_wait = True
                wait_frame = REINITIALIZE_GAP
                accuracy_wait_frames = ACCURACY_CALC_GAP
            else:
                if accuracy_wait_frames != 0:
                    accuracy_wait_frames -= 1
                    continue
                video.predictions.append(overlap)

        all_fails += video.n_fails
        all_frames += video.len
        all_overlaps += video.predictions
        ep_video_overlap = np.mean(video.predictions).item() if video.predictions != [] else 0

        # print info
        info_format = {
            'video_title': video.title,
            'mean_reward': round(ep_video_overlap, 4),
            'n_frames': video.len,
            'n_fails': video.n_fails
        }; print(INFO_STRING.format(**info_format))

        # clear for next episode
        video.n_fails = 0
        video.predictions = []

    ep_overlap = np.mean(all_overlaps).item() if all_overlaps != [] else 0
    ep_robustness = 100.0 * all_fails / all_frames

    # print overall info
    overall_info_format = {
        'overlap': round(ep_overlap, 4),
        'robustness': round(ep_robustness, 4)
    }; print(OVERALL_INFO_STRING.format(**overall_info_format))

    print("***********************************************\n\n")

    return net

