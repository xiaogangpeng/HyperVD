from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random

from model import Model
from dataset import Dataset
from train import train
from test import test
import option
import copy
from tqdm import tqdm

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args = option.parser.parse_args()
    setup_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

    train_data = Dataset(args, test_mode=False)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    test_data = Dataset(args, test_mode=True)
    test_loader = DataLoader(test_data,
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(args)
    if torch.cuda.is_available():
        model = model.to(args.device)
    test_tensor = (torch.rand(128, 200, 1024),)
    flops = FlopCountAnalysis(model, test_tensor)

    print(">>> training params: {:.3f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
    # print(">>> FLOPs: {:.3f}G".format(flops.total() / 1000000000.0))
    print("==========================================\n")

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    is_topk = True
    gt = np.load(args.gt)
    random_ap, online_ap = test(test_loader, model, gt, args)
    print('Random initialized AP: {:.4f}   on_line AP:  {:.4f} \n'.format(random_ap, online_ap))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_ap = 0.0

    st = time.time()




    for epoch in range(1, args.max_epoch+1):

        # lamda = min(args.lamda, args.lamda_cof * (epoch+1))
        total_loss, mil_loss, loss2, loss3 = train(train_loader, model, optimizer, args, criterion)

        scheduler.step()

        pr_auc, pr_auc_online = test(test_loader, model, gt, args)
        if pr_auc > best_ap:
            best_ap = pr_auc
            best_model_wts = copy.deepcopy(model.state_dict())

        print('[Epoch {0}/{1}]: total_loss: {2}  | offline pr_auc:{3:.4} | online pr_auc:{4:.4}\n'.format(epoch + 1, args.max_epoch, total_loss, pr_auc, pr_auc_online))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')

    time_elapsed = time.time() - st
    print('Training completes in {:.0f}m {:.0f}s | '
          'best test AP: {:.4f}\n'.format(time_elapsed // 60, time_elapsed % 60, best_ap))

