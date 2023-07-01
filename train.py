import torch
from tqdm import tqdm


def CENTROPY(logits, logits2, seq_len,):
    instance_logits = torch.tensor(0)  # tensor([])
    for i in range(logits.shape[0]):
        tmp1 = torch.sigmoid(logits[i, :seq_len[i]]).squeeze()
        tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()
        loss = torch.mean(-tmp1.detach() * torch.log(tmp2))
        instance_logits = instance_logits + loss
    instance_logits = instance_logits/logits.shape[0]
    return instance_logits


def train(dataloader, model, optimizer, args, criterion):
    t_loss = []

    with torch.set_grad_enabled(True):
        model.train()
        # for i, (n_inputs, a_inputs, n_labels, a_labels) in tqdm(enumerate(dataloader)):
        #     inputs = torch.cat([n_inputs, a_inputs], dim=0).cuda().float()
        #     labels = torch.cat([n_labels, a_labels], dim=0).cuda().float()
        for i, (inputs, labels) in tqdm(enumerate(dataloader)):
            seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
            inputs = inputs[:, :torch.max(seq_len), :]
            inputs = inputs.float().to(args.device)
            labels = labels.float().to(args.device)

            # features, v_logits, a_logits, av_logits = model(inputs)
            mil_logits, logits = model(inputs, seq_len)
            # logits = logits.squeeze()
            # audio_logits = audio_logits.squeeze()
            # visual_logits = visual_logits.squeeze()
            #
            # cmaloss_v2a_a2n, cmaloss_a2v_v2n = CMAL(mmil_logits, audio_logits, visual_logits, seq_len, audio_rep,
            #                                         visual_rep)

            # cmaloss1, cmaloss2 = CMAL2(logits, clip_feat, seq_feat, seq_len, idx)

            clsloss = criterion(mil_logits, labels)
            # clsloss2 = criterion(seq_logtis, labels)
            # total_loss = clsloss + args.lamda * cmaloss_v2a_a2n + args.lamda * cmaloss_a2v_v2n

            loss = clsloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # unit = dataloader.__len__() // 2
            # if i % unit == 0:
            #     print(f"Current Lambda_a2n: {args.lamda:.4f}")
            #     print(
            #         f"MIL Loss: {clsloss:.4f},  loss1: { args.lamda * cmaloss1:.4f},  loss2: { args.lamda * cmaloss2:.4f} ")

            t_loss.append(loss)

    # return sum(t_loss)/len(t_loss), sum(loss1)/len(loss1), sum(loss2)/len(loss2), sum(loss3)/len(loss3)
    return sum(t_loss)/len(t_loss), 0, 0, 0

