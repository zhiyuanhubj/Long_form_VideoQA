import os
import sys
sys.path.insert(0, '../')
from util import tokenize, transform_bb, load_file
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
import os.path as osp
import h5py
import random as rd
import numpy as np
import clip
from tqdm import tqdm, trange
from transformers import CLIPTextModel


class VideoQADataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        feature_dir,
        qmax_words=20,
        amax_words=5,
        bert_tokenizer=None,
        a2id=None,
        bnum=10,
        topk_selector_dataloading=0,
        num_frames_in_feature_file=32,
    ):
        self.data = pd.read_csv(f'./gsmt_data/datasets/egoqa/{split}.csv')
        self.dset = 'egoqa'
        self.all_answers = list(self.data['answer'])
        self.video_feature_path = feature_dir
        self.bbox_num = 16
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.bert_tokenizer = bert_tokenizer
        self.mc = 5
        self.topk_selector_dataloading = topk_selector_dataloading
        self.num_frames_in_feature_file = num_frames_in_feature_file

        app_feat_file = osp.join(self.video_feature_path, f'feats.h5')
        
        print('Load {}...'.format(app_feat_file))
        encoding = 'utf-8'
        self.frame_feats = {}
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        with h5py.File(app_feat_file, 'r') as fp:
            for i, key in enumerate(tqdm(fp.keys())):
                vid = key
                self.frame_feats[vid] = np.array(fp[vid])


    def __len__(self):
        return len(self.data)
    
    def get_video_feature(self, raw_vid_id, width=1, height=1):
        vid_id = raw_vid_id if raw_vid_id in self.frame_feats else raw_vid_id.strip('.mp4')

        frame_feat = self.frame_feats[vid_id][:, :]
        frame_feat = torch.from_numpy(frame_feat).type(torch.float32)

        return frame_feat


    def __getitem__(self, index):
        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video_id"]
        vid_id = str(vid_id)
        qid = str(cur_sample['qid'])

        video_f = self.get_video_feature(vid_id)
        
        vid_duration = video_f.shape[0]
        
        question_txt = cur_sample['question']

        question_embd = torch.tensor(
            self.bert_tokenizer.encode(
                question_txt,
                add_special_tokens=True,
                padding="longest",
                max_length=self.qmax_words,
                truncation=True,
            ),
            dtype=torch.long,
        )

        question_clip = clip.tokenize(question_txt, truncate=True)
        if self.topk_selector_dataloading:
            question_clip_hidden_state = self.clip_text_model(question_clip).pooler_output
            frame_question_similarity = torch.matmul(video_f, question_clip_hidden_state.T)[:,0]
            topk_frames = torch.topk(frame_question_similarity, self.num_frames_in_feature_file).indices
            video_f = video_f[topk_frames.sort().values, :]

        type, answer = 0, 0
        question_id = vid_id+'_'+qid
        answer_id = int(cur_sample["answer"])

        answer_txts = [question_txt +' [SEP] '+ self.data["a" + str(i)][index] for i in range(self.mc)]

        answer = tokenize(
            answer_txts,
            self.bert_tokenizer,
            add_special_tokens=True,
            max_length=self.amax_words,
            dynamic_padding=True,
            truncation=True,
        )

        seq_len = torch.tensor([len(ans) for ans in answer], dtype=torch.long)

        return {
            "video_id": vid_id,
            "video": video_f,
            "video_f": video_f,
            "video_len": vid_duration,
            "question": question_embd,
            "question_clip": question_clip,
            "question_txt": question_txt,
            "type": type,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": answer,
            "seq_len": seq_len,
            "question_id": question_id
        }


def videoqa_collate_fn(batch):

    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, features, a2id, bert_tokenizer, test_mode):
    data_dir = os.path.join(args.dataset_dir, args.dataset)
    if test_mode:
        test_dataset = VideoQADataset(
            data_dir=data_dir,
            split='test',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            bnum=args.bnum,
            topk_selector_dataloading=args.topk_selector_dataloading,
            num_frames_in_feature_file=args.num_frames_in_feature_file
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:

        train_dataset = VideoQADataset(
            data_dir=data_dir,
            split='train',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            bnum=args.bnum,
            topk_selector_dataloading=args.topk_selector_dataloading,
            num_frames_in_feature_file=args.num_frames_in_feature_file
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            drop_last=True,
            collate_fn=videoqa_collate_fn,
        )

        if args.dataset.split('/')[0] in ['tgifqa', 'tgifqa2', 'msrvttmc']:
            args.val_csv_path = args.test_csv_path
        val_dataset = VideoQADataset(
            data_dir=data_dir,
            split='val',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            bnum=args.bnum,
            topk_selector_dataloading=args.topk_selector_dataloading,
            num_frames_in_feature_file=args.num_frames_in_feature_file
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )
        test_loader = None

    return train_loader, val_loader, test_loader
