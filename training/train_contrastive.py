def contrastive_loss(x, y, temp=0.07):
    logits = x @ y.T / temp
    labels = torch.arange(len(x)).to(x.device)
    return nn.CrossEntropyLoss()(logits, labels)
