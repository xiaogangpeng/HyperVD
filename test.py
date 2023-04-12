from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch


def test(dataloader, model, gt, args):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(args.device)
        vanilla_input = torch.zeros(0).to(args.device)
        trained_output = torch.zeros(0).to(args.device)
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(args.device)
            _, logits, = model(inputs, None)  # (bs, len)
            sig = logits
            sig = torch.sigmoid(sig)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        # precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        # pr_auc2 = auc(recall, precision)

        return pr_auc, 0.0



