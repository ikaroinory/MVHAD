import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_confusion_matrix, binary_f1_score, binary_precision, binary_precision_recall_curve, binary_recall


def adjust_predict_labels(predict_labels: Tensor, actual_labels: Tensor) -> Tensor:
    adjusted_pred = predict_labels.clone()

    zeros = torch.tensor([0], device=actual_labels.device)
    pad_labels = torch.cat([zeros, actual_labels, zeros])
    starts = (pad_labels[1:] & ~pad_labels[:-1]).nonzero().squeeze(dim=-1)
    ends = (~pad_labels[1:] & pad_labels[:-1]).nonzero().squeeze(dim=-1)

    for start, end in zip(starts, ends):
        adjusted_pred[:, start:end] = torch.where(
            predict_labels[:, start:end].any(dim=-1).unsqueeze(-1),
            torch.tensor(1, device=predict_labels.device),
            adjusted_pred[:, start:end]
        )
    return adjusted_pred


def get_total_error_score(result: tuple[Tensor, Tensor, Tensor], epsilon: float = 1e-2) -> Tensor:
    predict_result, actual_result, _ = result

    delta = (predict_result - actual_result).abs()
    iqr = delta.quantile(0.75, dim=0) - delta.quantile(0.25, dim=0)
    score = (delta - delta.quantile(0.5, dim=0)) / (iqr + epsilon)  # [*, num_nodes]

    return score.max(dim=-1)[0]


def get_metrics(test_result: tuple[Tensor, Tensor, Tensor], point_adjustment: bool) -> tuple[float, float, float, float, float]:
    test_error_score = get_total_error_score(test_result)

    actual_labels = test_result[2]

    test_error_score = (test_error_score - test_error_score.min()) / (test_error_score.max() - test_error_score.min() + 1e-8)

    if not point_adjustment:
        precision, recall, thresholds = binary_precision_recall_curve(test_error_score, actual_labels, validate_args=False)
        f1_score_list = 2 * precision * recall / (precision + recall + 1e-8)
        predict_labels = (test_error_score >= thresholds[f1_score_list.argmax()])
    else:
        thresholds = torch.linspace(test_error_score.min(), test_error_score.max(), 10000)
        predict_labels_batch = (test_error_score.unsqueeze(0) > thresholds.unsqueeze(-1))
        predict_labels_batch = adjust_predict_labels(predict_labels_batch, actual_labels)
        f1s = binary_f1_score(
            predict_labels_batch,
            actual_labels.unsqueeze(0).expand(predict_labels_batch.shape[0], -1),
            multidim_average='samplewise',
            validate_args=False
        )
        predict_labels = predict_labels_batch[f1s.argmax()]

    tn, fp, fn, tp = binary_confusion_matrix(predict_labels, actual_labels).ravel()

    f1 = binary_f1_score(predict_labels, actual_labels, validate_args=False).item()
    precision = binary_precision(predict_labels, actual_labels, validate_args=False).item()
    recall = binary_recall(predict_labels, actual_labels, validate_args=False).item()
    fnr = (fn / (fn + tp)).item()
    fpr = (fp / (fp + tn)).item()

    return precision, recall, fpr, fnr, f1
