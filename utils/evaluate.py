import copy

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from torch import Tensor
from tqdm import tqdm


def adjust_pred(pred, label):
    adjusted_pred = copy.deepcopy(pred)

    anomaly_state = False
    anomaly_count = 0
    for i in range(len(adjusted_pred)):
        if label[i] and adjusted_pred[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                if not adjusted_pred[j]:
                    adjusted_pred[j] = True
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = True
    return adjusted_pred


def get_total_error_score(result: tuple[Tensor, Tensor, Tensor], epsilon: float = 1e-2) -> Tensor:
    predict_result, actual_result, _ = result

    delta = (predict_result - actual_result).abs()
    iqr = delta.quantile(0.75, dim=0) - delta.quantile(0.25, dim=0)
    score = (delta - delta.quantile(0.5, dim=0)) / (iqr + epsilon)  # [*, num_nodes]

    return score.max(dim=-1)[0]


def get_metrics(test_result: tuple[Tensor, Tensor, Tensor], point_adjustment: bool) -> tuple[float, float, float, float, float]:
    test_error_score = get_total_error_score(test_result)

    actual_labels = test_result[2].cpu()

    test_error_score = test_error_score.cpu()

    if not point_adjustment:
        precision, recall, thresholds = precision_recall_curve(actual_labels, test_error_score)
        f1_score_list = 2 * precision * recall / (precision + recall + 1e-8)

        best_index = f1_score_list.argmax()
        best_threshold = thresholds[best_index]
        predict_labels = (test_error_score >= best_threshold)
    else:
        thresholds = np.linspace(test_error_score.min(), test_error_score.max(), 10000)
        f1_list = []
        for threshold in tqdm(thresholds):
            predict_labels = (test_error_score > threshold)
            predict_labels = adjust_pred(predict_labels, actual_labels)
            f1_list.append(f1_score(actual_labels, predict_labels))
        best_threshold = thresholds[np.argmax(f1_list)]
        predict_labels = (test_error_score > best_threshold)
        predict_labels = adjust_pred(predict_labels, actual_labels)

    tn, fp, fn, tp = confusion_matrix(actual_labels, predict_labels).ravel()

    f1 = f1_score(actual_labels, predict_labels)
    precision = precision_score(actual_labels, predict_labels)
    recall = recall_score(actual_labels, predict_labels)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    return precision, recall, fpr, fnr, f1
