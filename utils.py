import math
import torch
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geos import TopologicalError

"""
=================================================
    SAMPLE PREDICTIONS AND CALCULATE REWARDS
=================================================
"""


def rotate_anticlockwise(vec, cos, sin):
    ans = np.zeros(2)
    ans[0] = vec[0] * cos - vec[1] * sin
    ans[1] = vec[0] * sin + vec[1] * cos
    return ans


def rotate_clockwise(vec, cos, sin):
    ans = np.zeros(2)
    ans[0] = vec[0] * cos + vec[1] * sin
    ans[1] = - vec[0] * sin + vec[1] * cos
    return ans


def extract_coord(tens):
    tens = tens.detach().cpu().numpy()
    v1 = tens[0:2]
    v3 = tens[2:4]
    a4 = (tens[4] + 1/2) * math.pi
    cos_a = math.cos(a4)
    sin_a = math.sin(a4)
    vc = (v1 + v3) / 2
    vc3 = v3 - vc
    v2 = vc + rotate_anticlockwise(vc3, -cos_a, sin_a)
    v4 = vc + rotate_clockwise(vc3, cos_a, sin_a)
    return [tuple(v1), tuple(v2), tuple(v3), tuple(v4)]


def sample_predictions(out, sig, N):
    """
    sigma * np.random.randn(...) + mu ~ N(mu, sigma ** 2)
    :param out: torch(T, 5)
    :param sig: variance
    :param T: sequence length
    :param N: number of samples
    :return: torch(T, N, 4)
    """
    ans = sig * torch.randn(out.size(0), N, 5)
    return ans + out.unsqueeze(1).expand_as(ans)


def reward2(pred, gt):
    if abs(pred[4].item()) >= 0.5:
        return 0
    pred_poly = Polygon(extract_coord(pred))
    gt_poly = Polygon(extract_coord(gt))
    try:
        inter_area = pred_poly.intersection(gt_poly).area
    except TopologicalError:
        inter_area = 0
    except ValueError:
        inter_area = 0

    if not inter_area:
        return 0
    else:
        return inter_area / (pred_poly.area + gt_poly.area - inter_area)


def compute_rewards2(gt, pred):
    """
    compute rewards at late training stage
    :param predictions: torch(T, 5)
    :param ground_truth: torch(T, N, 5)
    :return: torch(T, N)
    """
    out_rewards = torch.zeros(gt.size(0), pred.size(1))
    for i in range(gt.size(0)):
        for j in range(pred.size(1)):
            out_rewards[i][j] = reward2(pred[i][j], gt[i])
    return out_rewards


def compute_rewards1(gt, pred):
    """
    compute rewards at early training stage
    :param gt: torch(T, 5)
    :param pred: torch(T, N, 5)
    :return: torch(T, N)
    """
    gt = gt.unsqueeze(1).expand_as(pred)
    ans = torch.abs(pred - gt)
    return - (ans.mean(dim=2) + ans.max(dim=2)[0]) / 2


"""
=================================================
        BASELINES, LOSS AND SIGMA
=================================================
"""


# location vector
def loc_vec(gt):
    ans = torch.zeros(gt.size())
    ans[0] = gt[0].clone()
    return ans


# baseline
# rew : (T, N)
def compute_baseline(rew):
    rew = torch.sum(rew, dim=0)
    return torch.mean(rew, dim=0)


# log baseline
# pred : (T, N, 4)
# rew : (T, N)
# out : (T, 4)
def compute_log_baseline(rew, out, pred, sig):
    rew = torch.sum(rew, dim=0)
    out = out.unsqueeze(1).expand_as(pred)
    df = (out - pred) / (sig ** 2)
    df = torch.sum(df, dim=2)
    df = torch.sum(df, dim=0) ** 2
    numerator = torch.sum(df * rew).item()
    denominator = torch.sum(df).item()
    return numerator / denominator


# diff : (T, N, 4)
# out : (T, 4)
# rew : (T, N)
def compute_loss(rew, bs, out, diff):
    rew = torch.sum(rew, dim=0) - bs
    rew = rew.unsqueeze(0).unsqueeze(2)
    rew = rew.expand_as(diff)
    out = out.unsqueeze(1).expand_as(diff)
    return torch.sum(diff * out * rew)


def calculate_diff(out, pred, sig):
    out = out.unsqueeze(1).expand_as(pred)
    df = (out - pred) / (sig ** 2)
    return df


# weights for weight change curve
def compute_weights(net):
    ans = 0
    for param in net.parameters():
        ans += math.sqrt(torch.sum(param ** 2).item())
    return ans


# weights grad for weight grad change curve
def compute_weights_grad(net):
    ans = 0
    for param in net.parameters():
        ans += math.sqrt(torch.sum(param.grad ** 2).item())
    return ans


if __name__=='__main__':
    """
    part for testing functions above
    """
    a = torch.tensor([0, 0, 1, 5, 0])
    print(a)
    ext = extract_coord(a)
    print(ext)

