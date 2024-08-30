import numpy as np
import math
import torch
import torch.nn.functional as F


def angle_between_points(point1, point2, point3, ignore_direction=False, in_degree=False):
    vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vec2 = [point3[0] - point2[0], point3[1] - point2[1]]
    cos = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    angle = math.atan2(sin, cos)
    if ignore_direction:
        angle = abs(angle)

    if in_degree:
        angle = math.degrees(angle)
    return angle


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr=thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def distance_between(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def distance_points(pred, target, scale_factor=4):
    pred = F.interpolate(pred, scale_factor=scale_factor, mode='bilinear')
    pred = np.array(pred.detach().cpu())
    target = np.array(target.detach().cpu())
    batch, num_points, _, _ = pred.shape
    avg_dist = 0
    for batch_idx in range(batch):
        for i in range(num_points):
            pred_point = np.where(pred[batch_idx, i, :, :] == pred[batch_idx, i, :, :].max())
            pred_point = np.mean(pred_point, axis=1)
            pred_point = np.array(pred_point[1]), np.array(pred_point[0])
            target_point = np.array(target[batch_idx, i][0]), np.array(target[batch_idx, i][1])
            avg_dist += distance_between(pred_point, target_point)
    return avg_dist / (batch * num_points)


def offset_distance(pred, offset_x, offset_y, target, radius, scale_factor=4):
    pred = np.array(pred.detach().cpu())
    target = np.array(target.detach().cpu())
    offset_x = np.array(offset_x.detach().cpu())
    offset_y = np.array(offset_y.detach().cpu())
    batch, num_points, _, _ = pred.shape
    avg_dist = 0
    for batch_idx in range(batch):
        for i in range(num_points):
            pred_point = np.where(pred[batch_idx, i, :, :] == pred[batch_idx, i, :, :].max())
            pred_point = np.mean(pred_point, axis=1)
            x, y = np.array(round(pred_point[1])), np.array(round(pred_point[0]))  # x, y
            # print(x * scale_factor, target[batch_idx, i][0], y * scale_factor, target[batch_idx, i][1])
            pred_point = ((x * scale_factor) + (offset_x[batch_idx, i, y, x] * radius),
                          (y * scale_factor) + (offset_y[batch_idx, i, y, x] * radius))
            # pred_point = (x * scale_factor, y * scale_factor)
            target_point = np.array(target[batch_idx, i][0]), np.array(target[batch_idx, i][1])
            avg_dist += distance_between(pred_point, target_point)
    return avg_dist / (batch * num_points)


def crl_mask_acc(pred, offset_x, offset_y, target, radius, scale_factor=4):
    pred = np.array(pred.detach().cpu())
    target = np.array(target.detach().cpu())
    offset_x = np.array(offset_x.detach().cpu())
    offset_y = np.array(offset_y.detach().cpu())
    batch, num_points, _, _ = pred.shape
    count_a1 = count_a2 = count_a3 = count_angle = 0
    pred_angle = target_angle = 0
    for batch_idx in range(batch):
        pred_head = np.where(pred[batch_idx, 0, :, :] == pred[batch_idx, 0, :, :].max())
        pred_head = np.mean(pred_head, axis=1)
        x, y = np.array(round(pred_head[1])), np.array(round(pred_head[0]))
        pred_head = ((x * scale_factor) + (offset_x[batch_idx, 0, y, x] * radius),
                     (y * scale_factor) + (offset_y[batch_idx, 0, y, x] * radius))

        pred_hip = np.where(pred[batch_idx, 1, :, :] == pred[batch_idx, 1, :, :].max())
        pred_hip = np.mean(pred_hip, axis=1)
        x, y = np.array(round(pred_hip[1])), np.array(round(pred_hip[0]))
        pred_hip = ((x * scale_factor) + (offset_x[batch_idx, 1, y, x] * radius),
                    (y * scale_factor) + (offset_y[batch_idx, 1, y, x] * radius))

        pred_rotate = np.where(pred[batch_idx, 2, :, :] == pred[batch_idx, 2, :, :].max())
        pred_rotate = np.mean(pred_rotate, axis=1)
        x, y = np.array(round(pred_rotate[1])), np.array(round(pred_rotate[0]))
        pred_rotate = ((x * scale_factor) + (offset_x[batch_idx, 2, y, x] * radius),
                       (y * scale_factor) + (offset_y[batch_idx, 2, y, x] * radius))

        target_head = np.array(target[batch_idx, 0][0]), np.array(target[batch_idx, 0][1])
        target_hip = np.array(target[batch_idx, 1][0]), np.array(target[batch_idx, 1][1])
        target_rotate = np.array(target[batch_idx, 2][0]), np.array(target[batch_idx, 2][1])
        # print(x * scale_factor, target_hip[0], y * scale_factor, target_hip[1])
        # print((x * scale_factor) + (offset_x[batch_idx, 1, y, x] * radius), target_hip[0],
        #       (y * scale_factor) + (offset_y[batch_idx, 1, y, x] * radius) * scale_factor, target_hip[1])
        # print('\n')
        pred_crl = distance_between(pred_head, pred_hip)
        target_crl = distance_between(target_head, target_hip)
        pred_angle = angle_between_points(pred_hip, pred_rotate, pred_head,
                                          ignore_direction=True, in_degree=True)
        target_angle = angle_between_points(target_hip, target_rotate, target_head,
                                            ignore_direction=True, in_degree=True)
        count_angle += abs(pred_angle-target_angle)
        if abs(pred_crl - target_crl) < target_crl * 0.03:
            count_a1 += 1
        if abs(pred_crl - target_crl) < target_crl * 0.02:
            count_a2 += 1
        if abs(pred_crl - target_crl) < target_crl * 0.01:
            count_a3 += 1
    a1 = count_a1 / batch
    a2 = count_a2 / batch
    a3 = count_a3 / batch
    angle = count_angle / batch
    return a1, a2, a3, angle


def offset_distance_points(pred, offset_x, offset_y, target, radius, scale_factor=4):
    pred = np.array(pred.detach().cpu())
    target = np.array(target.detach().cpu())
    offset_x = np.array(offset_x.detach().cpu())
    offset_y = np.array(offset_y.detach().cpu())
    batch, num_points, _ = pred.shape
    avg_dist = 0
    for batch_idx in range(batch):
        for i in range(num_points):
            pred_point = pred[batch_idx, i, :]
            x, y = min(int(pred_point[0]), 159), min(int(pred_point[1]), 119)  # x, y
            pred_point = ((x * scale_factor) + (offset_x[batch_idx, i, y, x] * radius),
                          (y * scale_factor) + (offset_y[batch_idx, i, y, x] * radius))
            target_point = np.array(target[batch_idx, i][0]), np.array(target[batch_idx, i][1])
            avg_dist += distance_between(pred_point, target_point)
    return avg_dist / (batch * num_points)


def crl_point_acc(pred, offset_x, offset_y, target, radius, scale_factor=4):
    pred = np.array(pred.detach().cpu())
    target = np.array(target.detach().cpu())
    offset_x = np.array(offset_x.detach().cpu())
    offset_y = np.array(offset_y.detach().cpu())
    batch, num_points, _ = pred.shape
    count_a1 = count_a2 = count_a3 = 0
    for batch_idx in range(batch):
        pred_head = pred[batch_idx, 0, :]
        x, y = min(int(pred_head[0]), 159), min(int(pred_head[1]), 119)
        pred_head = ((x * scale_factor) + (offset_x[batch_idx, 0, y, x] * radius),
                     (y * scale_factor) + (offset_y[batch_idx, 0, y, x] * radius))

        pred_hip = pred[batch_idx, 1, :]
        x, y = min(int(pred_hip[0]), 159), min(int(pred_hip[1]), 119)
        pred_hip = ((x * scale_factor) + (offset_x[batch_idx, 1, y, x] * radius),
                    (y * scale_factor) + (offset_y[batch_idx, 1, y, x] * radius))

        target_head = np.array(target[batch_idx, 0][0]), np.array(target[batch_idx, 0][1])
        target_hip = np.array(target[batch_idx, 1][0]), np.array(target[batch_idx, 1][1])

        pred_crl = distance_between(pred_head, pred_hip)
        target_crl = distance_between(target_head, target_hip)

        if abs(pred_crl - target_crl) < target_crl * 0.03:
            count_a1 += 1
        if abs(pred_crl - target_crl) < target_crl * 0.02:
            count_a2 += 1
        if abs(pred_crl - target_crl) < target_crl * 0.01:
            count_a3 += 1
    a1 = count_a1 / batch
    a2 = count_a2 / batch
    a3 = count_a3 / batch
    return [a1, a2, a3]


def crl_heatmap_acc(pred, target, scale_factor=4):
    pred = F.interpolate(pred, scale_factor=scale_factor, mode='bilinear')
    pred = np.array(pred.detach().cpu())
    target = np.array(target.detach().cpu())
    batch, num_points, _, _ = pred.shape
    count_a1 = count_a2 = count_a3 = count_angle = 0
    for batch_idx in range(batch):
        pred_head = np.where(pred[batch_idx, 0, :, :] == pred[batch_idx, 0, :, :].max())
        pred_head = np.mean(pred_head, axis=1)
        x, y = np.array(round(pred_head[1])), np.array(round(pred_head[0]))
        pred_head = (x, y)

        pred_hip = np.where(pred[batch_idx, 1, :, :] == pred[batch_idx, 1, :, :].max())
        pred_hip = np.mean(pred_hip, axis=1)
        x, y = np.array(round(pred_hip[1])), np.array(round(pred_hip[0]))
        pred_hip = (x, y)

        target_head = np.array(target[batch_idx, 0][0]), np.array(target[batch_idx, 0][1])
        target_hip = np.array(target[batch_idx, 1][0]), np.array(target[batch_idx, 1][1])

        pred_crl = distance_between(pred_head, pred_hip)
        target_crl = distance_between(target_head, target_hip)

        pred_rotate = np.where(pred[batch_idx, 2, :, :] == pred[batch_idx, 2, :, :].max())
        pred_rotate = np.mean(pred_rotate, axis=1)
        x, y = np.array(round(pred_rotate[1])), np.array(round(pred_rotate[0]))
        pred_rotate = (x, y)
        target_rotate = np.array(target[batch_idx, 2][0]), np.array(target[batch_idx, 2][1])

        pred_angle = angle_between_points(pred_hip, pred_rotate, pred_head,
                                          ignore_direction=True, in_degree=True)
        target_angle = angle_between_points(target_hip, target_rotate, target_head,
                                            ignore_direction=True, in_degree=True)
        count_angle += abs(pred_angle-target_angle)

        if abs(pred_crl - target_crl) < target_crl * 0.03:
            count_a1 += 1
        if abs(pred_crl - target_crl) < target_crl * 0.02:
            count_a2 += 1
        if abs(pred_crl - target_crl) < target_crl * 0.01:
            count_a3 += 1

    a1 = count_a1 / batch
    a2 = count_a2 / batch
    a3 = count_a3 / batch
    angle = count_angle / batch
    return a1, a2, a3, angle
