import torch

__all__ = ['pdist']


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def pdist_batched(e, squared=False, eps=1e-12):
    """
    Computes pairwise distances for a batch of embedding matrices.
    
    Args:
        e: Tensor of shape (B, N, D)
        squared: If True, return squared distances
        eps: Small epsilon to prevent sqrt of zero

    Returns:
        A distance matrix of shape (B, N, N)
    """
    B, N, D = e.shape
    
    # Compute squared norms for each embedding
    e_square = e.pow(2).sum(dim=2)  # (B, N)
    
    # Compute dot product between all pairs in the batch
    prod = torch.bmm(e, e.transpose(1, 2))  # (B, N, N)

    # Apply the pairwise distance formula
    res = (e_square.unsqueeze(2) + e_square.unsqueeze(1) - 2 * prod).clamp(min=eps)

    # if not squared:
    #     res = res.sqrt()

    # Set diagonal to 0 (distance to self = 0)
    idx = torch.arange(N, device=e.device)
    res[:, idx, idx] = 0

    return res

def recall(embeddings, labels, K=[]):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []

    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k

