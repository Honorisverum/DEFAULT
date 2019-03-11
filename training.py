"""
=================================================
                TRAINING PHASE
TRAIN MODEL ON TRAIN VIDEOS USING DEEP RL ALGORITHM
=================================================
"""


import utils
import torch
import os


def train(training_set_videos, net, optimizer, save_every,
          T, N, sigma, epochs1, epochs2, use_gpu):

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    save_path = os.path.join(os.getcwd(), 'weights', 'last.pt')

    net.train()

    for epoch in range(1, epochs1 + epochs2 + 1):

        print(f"Epoch: {epoch}")

        # training stage
        reward_func = utils.compute_rewards1 if epoch <= epochs1 else utils.compute_rewards2

        ep_reward = 0

        for i, video in enumerate(training_set_videos):

            # accumulate reward/weights for info
            ep_video_reward = 0

            for gt, images in video.get_dataloader(T):

                if use_gpu:
                    gt = gt.cuda()
                    images = images.cuda()

                # compute location vec
                s_t = utils.loc_vec(gt)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass to get output
                outputs = net(images, s_t)

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

            ep_video_reward /= video.len

            ep_reward += ep_video_reward

            # print info for ep for this video
            iteration_info_format = {
                'video_title': video.title,
                'mean_reward': round(ep_video_reward, 4)
            }

            train_info_string = "Title {video_title} |" \
                                " Mean Reward: {mean_reward} |"

            print(train_info_string.format(**iteration_info_format))

            net.clear_memory()

        if save_every is not None:
            if not epoch % save_every:
                torch.save(net, save_path)

        print("Mean epoch reward:", round(ep_reward / len(training_set_videos), 4))

        # for divide info on ep blocks
        print("================================================\n")

    # final save
    if save_every is not None:
        torch.save(net, save_path)

    return net

