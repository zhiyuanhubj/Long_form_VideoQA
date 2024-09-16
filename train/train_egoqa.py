import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json
import clip
from tqdm import tqdm
from thop import profile

def eval(model, data_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(tqdm(data_loader)):
            answer_id, answer, video, question, question_clip = (
                batch["answer_id"],
                batch["answer"],
                batch["video"].cuda(),
                batch["question"].cuda(),
                batch['question_clip'].cuda()
            )
            video_f = video
            video_len = batch["video_len"]
            seq_len = batch["seq_len"]
            question_mask = (question > 0).float()
            answer_mask = (answer > 0).float()
            video_mask = get_mask(video_len, video_f.size(1)).cuda()
            count += answer_id.size(0)
            question_id = batch['question_id']
            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                fusion_proj, answer_proj = model(
                    video,
                    question,
                    question_clip= question_clip,
                    text_mask=question_mask,
                    # text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer.cuda(),
                    seq_len = seq_len
                )
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}

    step = "val" if not test else "test"
    
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )

    def convert_models_to_fp32(model):
        for param in model.parameters():
            try:
                param.data = param.data.float()
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()
            except:
                from IPython.core.debugger import Pdb
                dbg = Pdb()
                dbg.set_trace()
    
    convert_models_to_fp32(model.clip)
    for i, batch in enumerate(tqdm(train_loader)):
        answer_id, answer, video, question, question_clip = (
            batch["answer_id"],
            batch["answer"],
            batch["video"].cuda(),
            batch["question"].cuda(),
            batch['question_clip'].cuda()
        )
        video_len = batch["video_len"]
        question_mask = (question > 0).float()
        answer_mask = (answer>0).float()
        video_f = video
        video_mask = (
            get_mask(video_len, video_f.size(1)).cuda() if args.max_feats > 0 else None
        )
        N = answer_id.size(0)
        seq_len = batch["seq_len"]

        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                video_mask=video_mask,
                seq_len = seq_len
            )
        else:
            fusion_proj, answer_proj, c3_loss = model(
                video,
                question,
                question_clip= question_clip,
                text_mask=question_mask,
                video_mask=video_mask,
                answer=answer.cuda(),
                seq_len = seq_len,
                labels=answer_id.cuda()
            )

            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        vqa_loss = criterion(predicts, answer_id.cuda())
        predicted = torch.max(predicts, dim=1).indices.cpu()
        running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            max_seq_len = args.qmax_words
            if args.mc > 0:
                tmp_id = [aid+(args.mc*i) for i, aid in enumerate(answer_id)]
                inputs = answer.view(N*args.mc, -1)[tmp_id,:]
                question_mask = (inputs>0).float()
                max_seq_len = args.amax_words
            else:
                inputs = batch["question"]
            inputs, labels = mask_tokens(
                inputs, tokenizer, mlm_probability=args.mlm_prob
            )
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                max_seq_len=max_seq_len,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        else:
            loss = vqa_loss

        loss = loss + c3_loss * 0.005
        optimizer.zero_grad()
        loss.backward()

        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
