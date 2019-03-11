"""
=================================================
                QUALITY CONTROL
            COMPUTE METRICS AND PLOTS
=================================================
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import math

CWD = os.getcwd()


def compute_overlap_threshold_curve(pred, num_frames):
    pt = list(np.linspace(0, 1, 50))
    ans = []
    for threshold in pt:
        val = len([x for x in pred if x > threshold])
        ans.append(val / num_frames)
    return ans, pt


def compute_metrics(test_set_videos):
    print("RESULTS:")

    all_predictions = []
    all_fails = 0
    all_frames = 0

    with open(CWD + "/results/report.txt", "w+") as f:

        for video in test_set_videos:

            all_predictions += video.test_predictions
            all_fails += video.test_fails
            all_frames += video.len

            video_report_format = {
                'video_title': video.title,
                'frames_num': video.len,
                'mean_overlap': round(np.mean(video.test_predictions).item(), 3) if video.test_predictions != [] else 0.0,
                'fails': video.test_fails
            }

            f.write("Test for {video_title} with "
                    "{frames_num} frames |"
                    " Mean overlap: {mean_overlap} |"
                    " Number of fails: {fails} \n".format(**video_report_format))

            print("Test for {video_title} with "
                  "{frames_num} frames |"
                  " Mean overlap: {mean_overlap} |"
                  " Number of fails: {fails}".format(**video_report_format))

        f.write("\n\n")

        final_report_format = {
            'average_accuracy': round(np.mean(all_predictions).item(), 4),
            'total_fails': all_fails,
            'robustness': round(5 * all_fails / all_frames, 4),
            'reliability': round(math.exp(- 100.0 * all_fails / all_frames), 4)
        }

        f.write("AVERAGE ACCURACY: {average_accuracy}\n"
                "TOTAL FAILS: {total_fails}\n"
                "ROBUSTNESS: {robustness} of total frames\n"
                "RELIABILITY(s=100): {reliability}\n".format(**final_report_format))

        print("AVERAGE ACCURACY: {average_accuracy}\n"
              "TOTAL FAILS: {total_fails}\n"
              "ROBUSTNESS: {robustness} of total frames\n"
              "RELIABILITY(s=100): {reliability}\n".format(**final_report_format))

    # plot overlap curve and save it values
    overlap_threshold_curve, points = compute_overlap_threshold_curve(all_predictions, all_frames)
    np.savetxt(CWD + "/results/overlap_threshold.txt", overlap_threshold_curve)


def draw_plots(rews, wghts, wghts_grad):

    #plt.plot(points, overlap_threshold_curve, color='green')
    #plt.xlabel('threshold')
    #plt.ylabel('success rate')
    #plt.savefig(CWD + '/results/overlap_threshold.png')
    #plt.close()

    # learning curve
    plt.plot(rews, color='orange')
    plt.xlabel('epoch')
    plt.ylabel('mean reward')
    plt.savefig(CWD + '/results/learning_curve.png')
    plt.close()

    # weights curve
    plt.plot(wghts, color='red')
    plt.xlabel('epoch')
    plt.ylabel('weights modulus')
    plt.savefig(CWD + '/results/weights_curve.png')
    plt.close()

    # weights grad curve
    plt.plot(wghts_grad, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('weights grad modulus')
    plt.savefig(CWD + '/results/weights_grad_curve.png')
    plt.close()

    np.savetxt(CWD + "/results/rewards.txt", rews)
    np.savetxt(CWD + "/results/weights.txt", wghts)
    np.savetxt(CWD + "/results/weights_grad.txt", wghts_grad)
