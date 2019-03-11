"""
=================================================
                TRAINING PHASE
TRAIN MODEL ON TRAIN VIDEOS USING DEEP RL ALGORITHM
=================================================
"""


import utils
import torch
import numpy as np
import os


def train(training_set_videos, net, optimizer, save_filename,
          save_every, T, N, sigma, epochs, initial_stage, use_gpu):

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    save_path = os.path.join(os.getcwd(), 'weights', save_filename)

    rewards_curve = []
    weights_curve = []
    weights_grad_curve = []

    net.train()

    for epoch in range(1, epochs + 1):

        print("Epoch: {}".format(epoch))

        # training stage
        reward_func = utils.compute_rewards1 if epoch <= initial_stage else utils.compute_rewards2

        # weight/reward info for episode
        ep_weights = []
        ep_weights_grad = []
        ep_reward = []

        for i, video in enumerate(training_set_videos):

            # not to throw cell/hidden state from prev sequence
            is_new = True

            # accumulate reward/weights for info
            ep_video_reward = 0
            ep_video_weights = 0
            ep_video_weights_grad = 0

            gt_loader = video.create_gt_iter(T)
            frames_loader = video.create_frames_iter(T)

            for gt, images in zip(gt_loader, frames_loader):

                # compute location vec
                s_t = utils.loc_vec(gt)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass to get output
                outputs = net(images, s_t, is_new, use_gpu)
                is_new = False

                # sample predictions
                predictions = utils.sample_predictions(outputs, sigma, N).detach()

                # calculate rewards
                rewards = reward_func(gt, predictions)

                # calculate baseline
                baseline = utils.compute_baseline(rewards)

                # calculate diff and loss
                differences = utils.calculate_diff(outputs, predictions, sigma).detach()
                loss = utils.compute_loss(rewards, baseline, outputs, differences)

                # Getting gradients
                loss.backward()

                # Updating parameters
                optimizer.step()

                ep_video_reward += baseline.item()
                ep_video_weights += utils.compute_weights(net)
                ep_video_weights_grad += utils.compute_weights_grad(net)

            ep_video_reward /= video.len

            # print info for ep for this video
            iteration_info_format = {
                'video_title': video.title,
                'mean_reward': round(ep_video_reward, 3)
            }

            train_info_string = "Title {video_title} |" \
                                " Mean Reward: {mean_reward} |"

            print(train_info_string.format(**iteration_info_format))

            # for curves
            ep_weights.append(ep_video_weights / video.len)
            ep_weights_grad.append(ep_video_weights_grad / video.len)
            ep_reward.append(ep_video_reward)

        if save_every is not None:
            if epoch % save_every:
                torch.save(net, save_path)

        weights_curve.append(np.mean(ep_weights))
        weights_grad_curve.append(np.mean(ep_weights_grad))
        rewards_curve.append(np.mean(ep_reward))

        print("Mean epoch reward:", round(rewards_curve[-1], 3))

        # for divide info on ep blocks
        print("================================================\n")

    # final save
    if save_every is not None:
        torch.save(net, save_path)

    return weights_curve, weights_grad_curve, rewards_curve, net

