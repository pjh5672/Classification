def compute_accuracy(output, target, topk=(1,)):
    """computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (_type_): _description_
        target (_type_): _description_
        topk (tuple, optional): _description_. Defaults to (1,).

    Returns:
        _type_: _description_
    """
    _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res