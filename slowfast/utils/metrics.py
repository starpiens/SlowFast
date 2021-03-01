import torch


def num_topK_correct(preds, labels, k_list=(1, 5)):
    _, top_k_indices = torch.topk(preds, max(k_list))
    expanded_labels = labels.view(-1, 1).expand_as(top_k_indices)
    match_matrix = ((expanded_labels - top_k_indices) == 0)
    num_correct = [match_matrix[:, :k].sum() for k in k_list]
    return num_correct
