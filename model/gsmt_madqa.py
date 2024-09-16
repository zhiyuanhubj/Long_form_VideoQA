import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
import math
import copy
from transformers.activations import gelu
from transformers.modeling_outputs import BaseModelOutput
from transformers import BertConfig
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
from util import get_mask
from model.gssl_pytorch import GSS


class AModel(nn.Module):
    def __init__(self, bert_tokenizer, word_dim=768, out_dim=512):
        super(AModel, self).__init__()
        self.bert = Bert(bert_tokenizer)
        self.linear_text = nn.Linear(word_dim, out_dim)

    def forward(self, answer):
        if len(answer.shape) == 3:
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer, intra_text_attention = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)
            answer_g = answer_g.view(bs, nans, -1)
        else:
            answer, intra_text_attention = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)

        return answer_g, answer, intra_text_attention


class Bert(nn.Module):
    def __init__(self, bert_tokenizer):
        super(Bert, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.bert.resize_token_embeddings(len(bert_tokenizer))


    def forward(self, tokens):
        attention_mask = (tokens > 0).float()
        outs = self.bert(tokens, attention_mask=attention_mask, output_attentions=True)
        embds = outs[0]

        return embds, outs[-1][-1]


class EncoderVid(nn.Module):
    def __init__(self, feat_dim, bbox_dim, feat_hidden, pos_hidden, input_dropout_p=0.3):
        super(EncoderVid, self).__init__()
        self.dim_feat = feat_dim
        self.dim_bbox = bbox_dim
        self.dim_hidden = feat_hidden
        self.input_dropout_p = input_dropout_p

        input_dim = feat_dim

        input_dim += pos_hidden
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(self.dim_bbox, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
            nn.Conv2d(pos_hidden, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),

        )

        self.tohid = nn.Sequential(
            nn.Linear(feat_dim + pos_hidden, feat_hidden),
            nn.ELU(inplace=True))

        # self.roi_conv = nn.Sequential(
        #     nn.Conv1d(feat_dim, feat_hidden, kernel_size=3, padding=1),
        #     nn.ELU(inplace=True)
        # )

        # self.roi_conv = nn.Sequential(
        #     nn.Conv2d(4, 4, kernel_size=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        # )

    def forward(self, video_o):
        bsize, numc, numf, numr, fdim = video_o.shape

        video_o = video_o.view(bsize, numc * numf, numr, fdim)
        roi_feat = video_o[:, :, :, :self.dim_feat]
        roi_bbox = video_o[:, :, :, self.dim_feat:(self.dim_feat + self.dim_bbox)]
        bbox_pos = self.bbox_conv(roi_bbox.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        bbox_features = torch.cat([roi_feat, bbox_pos], dim=-1)

        bbox_feat = self.tohid(bbox_features)

        return bbox_feat

def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.num_attention_heads  # config.n_heads
        self.dim = config.hidden_size  # config.dim
        dp_rate = config.attention_probs_dropout_prob  # config.attention_dropout
        self.dropout = nn.Dropout(p=dp_rate)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  
        k = shape(self.k_lin(key))  
        v = shape(self.v_lin(value)) 

        q = q / math.sqrt(dim_per_head) 
        scores = torch.matmul(q, k.transpose(2, 3))  
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  
        scores.masked_fill_(mask, -float("inf")) 

        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)  

        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  
        context = unshape(context)  
        context = self.out_lin(context)  

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout, dim, hidden_dim = config.attention_probs_dropout_prob, config.hidden_size, config.intermediate_size
        activation = config.hidden_act

        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # dim = config.dim
        dim = config.hidden_size
        # assert config.dim % config.n_heads == 0
        assert dim % config.num_attention_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  
        else:  
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x) 

        ffn_output = self.ffn(sa_output)  
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.num_hidden_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.n_layers)]
        )

    def forward(
            self,
            x,
            attn_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
    ):
      
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Embeddings(nn.Module):
    def __init__(
            self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds, d_pos=128
    ):
        super().__init__()
        max_position_embeddings = language_len + 2 * vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  

        position_embeddings = self.position_embeddings(
            position_ids
        )  
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)  
        embeddings = self.dropout(embeddings)  

        return embeddings


class PositionEmbeddings(nn.Module):
    def __init__(
        self, d_model, max_position_embeddings, sinusoidal_pos_embds
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        if len(embeddings.size()) == 4:
            bsize, numf, numr, fdim = embeddings.size()
            position_ids = torch.arange(numf, dtype=torch.long, device=embeddings.device)  # (max_seq_length)
            position_ids = position_ids.view(1, -1, 1).expand(bsize, -1, numr)  # (bs, max_seq_length， num_obj)
        elif len(embeddings.size()) == 3:
            bsize, numf, fdim = embeddings.size()
            position_ids = torch.arange(numf, dtype=torch.long, device=embeddings.device)  # (max_seq_length)
            position_ids = position_ids.view(1, -1).expand(bsize, -1)  # (bs, max_seq_length， num_obj)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        return position_embeddings

class TokenTypeEmbeddings(nn.Module):
    def __init__(
        self, d_model, token_type_num
    ):
        super().__init__()
        self.modality_embedding = nn.Embedding(token_type_num, d_model)
        self.type2id = {'object': 0,
                        'segment': 1,
                        'question': 2}
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, token_type):
        seq_length = embeddings.size(1)
        token_type_id = self.type2id[token_type]
        modality_embeddings = self.modality_embedding(
            torch.tensor(
                [token_type_id] * seq_length, dtype=torch.long
            ).to(embeddings.device)
        )
        return modality_embeddings

class POSEmbeddings(nn.Module):
    def __init__(
            self, d_model, max_seq_len, dropout, sinusoidal_pos_embds, d_pos=128
    ):
        super().__init__()
        max_position_embeddings = max_seq_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_pos)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_pos,
                out=self.position_embeddings.weight,
            )
        self.merge_pos = nn.Sequential(
            nn.Linear(d_model + d_pos, d_model),
            nn.ELU(inplace=True))

    def forward(self, embeddings, cid):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids += cid * seq_length

        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        # print(position_embeddings.shape)
        embeddings = self.merge_pos(torch.cat([embeddings, position_embeddings], dim=-1))  # (bs, max_seq_length, dim)

        cpos_embed = position_embeddings.mean(dim=1)  # (bs, dim)
        return embeddings, cpos_embed

class Selector(nn.Module):
    def __init__(self, topk, selection_method='gumbel', q_dim=512, dim=512):
        super(Selector, self).__init__()
        self.linear_Q = nn.Linear(q_dim, dim)
        self.norm_Q = nn.LayerNorm(dim, eps=1e-12)

        self.linear_K = nn.Linear(dim, dim)
        self.norm_K = nn.LayerNorm(dim, eps=1e-12)

        self.topk = topk
        self.selection_method = selection_method

    @staticmethod
    def sample_gumbel(n, k):
        unif = torch.distributions.Uniform(0, 1).sample((n, k))
        g = -torch.log(-torch.log(unif))
        return g

    # @staticmethod
    def sample_gumbel_softmax(self, pi, temperature):
        n, k = pi.shape
        # dbg.set_trace()
        g = self.sample_gumbel(n, k).to(pi.device)
        h = (g + torch.log(pi)) / temperature
        h_max = h.max(dim=1, keepdim=True)[0]
        h = h - h_max
        cache = torch.exp(h)
        #     print(pi, torch.log(pi), intmdt)
        y = cache / cache.sum(dim=-1, keepdim=True)
        return y

    def forward(self, Q, K, V):
        '''
        Q: (bs, q_dim, 1)
        K: (bs, n_select, dim), n_select could be num_obj or num_seg
        V: (bs, n_select, n_frame_per_clip, obj_num, obj_dim)
        '''
        bs, n_select, _ = K.shape
        obj_num, obj_dim = V.shape[-2:]
        # from IPython.core.debugger import set_trace;
        # set_trace()
        v_shape = V.shape
        # V = V.view(bs, n_select, -1)

        # dbg.set_trace()

        Q = self.norm_Q(self.linear_Q(Q.squeeze(dim=-1)))  # [bs, dim, 1] -> [bs, dim]
        K = self.norm_K(self.linear_K(K))  # [bs, numc, dim]

        logit_scale = 1
        x_logits = logit_scale * K @ Q.unsqueeze(dim=-1)
        x_logits = torch.softmax(x_logits.squeeze(dim=-1), dim=-1)

        # selection_index_argmax = x_logits.topk(dim=1, k=1)[1]
        # selection_mask = torch.zeros_like(x_logits, memory_format=torch.contiguous_format).scatter_(
        #     dim=1, index=selection_index_argmax, value=1.0)
        #
        # selected_clip = torch.matmul(selection_mask.permute(0, 2, 1), video_proj.view(bs, n_clip, -1)).view(bs, -1,
        #                                                                                                     512)

        selected_segs = []
        for _ in range(self.topk):
            # print(x_logits.shape)
            # selection_mask = self.sample_gumbel_softmax(x_logits, 1)
            selection_mask = F.gumbel_softmax(x_logits, tau=100, dim=-1)
            if torch.isnan(selection_mask).sum() or torch.isinf(selection_mask).sum():
                from IPython.core.debugger import Pdb
                Pdb().set_trace()
                # dbg.set_trace()
            selection_mask = selection_mask.unsqueeze(dim=1)
            if V.dim() == 3:
                selected_segs.append(
                    torch.matmul(selection_mask, V.view(bs, n_select, -1)))
            else:
                selected_segs.append(
                    torch.matmul(selection_mask, V.view(bs, n_select, -1)).view(bs, -1, obj_num, obj_dim))

        selected_segs = torch.cat(selected_segs, dim=1)  # [bs, topk * num_obj, CLIP_dim]

        return selected_segs


class GatedSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = 2
        layer = GSS(
            dim=config.hidden_size,
            dim_expansion_factor=4,
            dss_kernel_N=512,
            dss_kernel_H=256
        )
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.n_layers)]
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x
    

class ISTA(nn.Module):
    def __init__(self, feature_dim, word_dim, Q, N, d_model, dropout, d_ff, h, topk=6, topj=12, num_frames_in_feature_file=128):
        super(ISTA, self).__init__()
        self.topk = topk
        self.numc = 32
        self.numf = int(32 / self.numc)
        self.topj = topj  # max self.numr is 16
        T = self.numc + (self.topj) * self.topk * self.numf + num_frames_in_feature_file
        self.position = Embeddings(d_model, Q, T, dropout, True)

        self.config = BertConfig.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=N,
            hidden_size=d_model,
            attention_probs_dropout_prob=dropout,
            intermediate_size=d_ff,
            num_attention_heads=h,
        )

        self.mmt = Transformer(self.config)
        self.seg_selector = Selector(topk=self.topk)
        self.reg_selector = Selector(topk=self.topj)

        # segment embedding
        self.linear_video = nn.Linear(feature_dim, d_model)
        self.norm_video = nn.LayerNorm(d_model, eps=1e-12)

        # patch embedding
        self.linear_patch = nn.Linear(feature_dim, d_model)
        self.norm_patch = nn.LayerNorm(d_model, eps=1e-12)

        # question post bert modules
        self.linear_question = nn.Linear(word_dim, d_model)
        self.norm_question = nn.LayerNorm(d_model, eps=1e-12)

        self.d_model = d_model

        self.apply(self._init_weights)


    def get_segment_embedding(self, video):
        video = self.linear_video(video)
        video = gelu(video)
        video = self.norm_video(video)
        return video

    def get_patch_embedding(self, patch):
        patch = self.linear_patch(patch)
        patch = gelu(patch)
        patch = self.norm_patch(patch)
        return patch

    def get_question_embedding(self, question):
        question = self.linear_question(question)
        question = gelu(question)
        question = self.norm_question(question)
        return question

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, q_feat, q_mask, question, seg_feat, video_o):
        bsize, q_len, _ = question.shape
        seg_len = seg_feat.shape[1]
        feat_dim = seg_feat.shape[-1]

        selected_patches = self.seg_selector(q_feat, seg_feat, video_o)  # [bs, topk * numf, num_obj, dim]

        q_feat_tmp = q_feat.unsqueeze(dim=1).repeat(1, selected_patches.shape[1], 1, 1)  # [bs, topk * numf, num_obj, dim]
        q_feat_tmp = q_feat_tmp.view(-1, q_feat_tmp.shape[-2], q_feat_tmp.shape[-1])  # [bs * topk * numf, num_obj, dim]
        selected_patches = selected_patches.view(-1, selected_patches.shape[-2], selected_patches.shape[-1]) # [bs * topk * numf, num_obj, dim]

        selected_patches = self.reg_selector(q_feat_tmp, selected_patches, selected_patches)  # [bs * topk * numf, topj, dim]
        selected_patches = selected_patches.view(bsize, -1, selected_patches.shape[-1])  # [bs, topk * numf * topj, dim]

        # Position and Token Type Embedding
        seg_feat = self.get_segment_embedding(seg_feat)
        patch_feat = self.get_patch_embedding(selected_patches).view(bsize, -1, self.d_model)

        vq_cat = torch.cat([seg_feat, patch_feat], dim=1)

        video_mask = torch.ones([bsize, seg_len + patch_feat.size(1)], dtype=torch.long, device=patch_feat.device)
        mask = torch.cat([video_mask], dim=1)
        vq_cat = self.position(vq_cat)

        attended_vq, intra_video_attention = self.mmt(x=vq_cat, attn_mask=mask, output_attentions=True)
        intra_video_attention = intra_video_attention[-1]
        intra_video_attention = torch.max(intra_video_attention, 1).values
        out_seg_feat = attended_vq[:, :seg_len]

        return attended_vq, out_seg_feat, intra_video_attention


class GSMT_VideoQA(nn.Module):
    def __init__(
            self,
            bert_tokenizer,
            feature_dim=512,
            word_dim=768,
            N=2,
            h=8,
            d_model=512,
            d_ff=2048,
            dropout=0.1,
            Q=20,
            T=20,
            vocab_size=30522,
            baseline="",
            n_negs=1,
            probe=False,
            topk=2,
            numc=8,
            topj=12,
            bnum=5,
            CM_PT=False,
            dataset="",
            clip_dim=512,
            num_frames_in_feature_file=32,
    ):
        super(GSMT_VideoQA, self).__init__()
        self.baseline = baseline
        self.Q = Q
        self.T = T
        self.n_negs = n_negs
        self.numc = num_frames_in_feature_file
        self.numf = 1
        self.num_ista = 1
        d_pos = 128
        feature_dim = 512

        self.encode_vid = EncoderVid(feat_dim=feature_dim,
                                     bbox_dim=5,
                                     feat_hidden=d_model,
                                     pos_hidden=d_pos)

        self.config = BertConfig.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=N,
            hidden_size=d_model,
            attention_probs_dropout_prob=dropout,
            intermediate_size=d_ff,
            num_attention_heads=h,
        )

        self.vqproj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.ttrans = Transformer(self.config)

        self.topk=1
        self.topj=16

        # question post bert modules
        self.linear_question = nn.Linear(word_dim, d_model)
        self.norm_question = nn.LayerNorm(d_model, eps=1e-12)
        self.num_frames_in_feature_file = num_frames_in_feature_file

        self.ISTA = [ISTA(feature_dim=feature_dim, word_dim=word_dim, Q=Q, N=N,
                          d_model=d_model, dropout=dropout, d_ff=d_ff, h=h, topk=self.topk, topj=self.topj, num_frames_in_feature_file=num_frames_in_feature_file)]
        for _ in range(self.num_ista - 1):
            self.ISTA.append(
                ISTA(feature_dim=d_model, word_dim=d_model, Q=Q, N=N,
                     d_model=d_model, dropout=dropout, d_ff=d_ff, h=h, topk=self.topk, topj=self.topj)
            )
        self.ISTA = nn.ModuleList(self.ISTA)

        # # weight initialization
        self.apply(self._init_weights)
        self.answer_embeddings = None

        self.amodel = AModel(bert_tokenizer, out_dim=d_model)
        self.clip, _ = clip.load("ViT-B/32")
        self.bert = Bert(bert_tokenizer)

        self.gss = GSS(
            dim=d_model,
            dim_expansion_factor=4,
            dss_kernel_N=4,
            dss_kernel_H=4
        )
     

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_answer_embedding(self, a2v):
        self.answer_embeddings = self.get_answer_embedding(a2v)

    def get_answer_embedding(self, answer):
        answer_g, answer, intra_text_attention = self.amodel(answer)
        return answer_g, answer, intra_text_attention

    def get_question_embedding(self, question, text_mask):
        question, _ = self.bert(question)
        if question.shape[1] < self.Q:
            question = torch.cat(
                [
                    question,
                    torch.zeros(
                        question.shape[0],
                        self.Q - question.shape[1],
                        question.shape[2],
                    ).cuda(),
                ],
                1,
            )
            text_mask = torch.cat(
                [
                    text_mask,
                    torch.zeros(
                        text_mask.shape[0], self.Q - text_mask.shape[1]
                    ).cuda(),
                ],
                1,
            )
        return question

    def get_vqa_embedding_simplify(self, video_f, language=None, language_lens=None):
        if language is not None:
            bsize_lan, len_lan, dim_lan = language.size()
            ans_n = bsize_lan // bsize

        bsize, numf, fdim = video_f.size()
        short_mask = get_mask(torch.tensor([1] * bsize * numf, dtype=torch.long), 1).cuda()
        X = self.ttrans(video_f, short_mask)[0]

        X = self.gss(X)
        try:
            video = X.view(bsize, numf, 1, 1, -1)
        except:
            from IPython.core.debugger import Pdb
            dbg = Pdb()
            dbg.set_trace()

        return video

    def get_clip_txt_embedding(self, question):
        bsize = question.size(0)
        question_clip, word_clip = self.clip.encode_text(question.squeeze(dim=1))

        question_clip = question_clip / question_clip.norm(dim=-1, keepdim=True)   
        question_clip = question_clip.view(bsize, -1, 1).float()  

        word_clip = word_clip / word_clip.norm(dim=-1, keepdim=True)  
        word_clip = word_clip.view(bsize, -1, 1).float()  
        return question_clip, word_clip

    def forward(
            self,
            video,
            question=None,
            question_clip=None,
            labels=None,
            answer=None,
            seq_len=None,
            video_mask=None,
            text_mask=None,
            max_seq_len=0,
            mode="vqa"
    ):
        video_f = video
        bsize = video.shape[0]
        numc, numf = self.numc, self.numf

        answer_g, answer_w, intra_text_attention = (
            self.get_answer_embedding(answer)
            if answer is not None
            else self.answer_embeddings
        )

        video_proj = self.get_vqa_embedding_simplify(video_f)  
        bs, n_clip, n_frame, n_object, dim = video_proj.shape

        q_feat, w_feat = self.get_clip_txt_embedding(question_clip)
        question = self.get_question_embedding(question, text_mask)

        video_f = video_f.view(bsize, numc, numf, -1)
        seg_feat = torch.mean(video_f, dim=-2)
        seg_feat = seg_feat / seg_feat.norm(dim=1, keepdim=True)   

        out_list = []
        for ista in self.ISTA:
            attended_vq, seg_feat, intra_video_attention = ista(q_feat, text_mask, question, seg_feat, video_proj)
            out_list.append(attended_vq)

        fusion_proj = torch.sum(torch.stack([out.mean(dim=1) for out in out_list], dim=-1), dim=-1)
        fusion_proj = self.vqproj(fusion_proj)

        if labels is not None:
            intra_text_attention = torch.max(intra_text_attention, 1).values
            intra_text_attention = intra_text_attention.view(bsize, -1, intra_text_attention.shape[1], intra_text_attention.shape[2])
            intra_text_attention = torch.index_select(intra_text_attention, 1, labels)[:,0,:,:]

            text_hidden_states = torch.index_select(answer_w.view(bsize, -1, answer_w.shape[1], answer_w.shape[2]), 1, labels)[:,0,:]
            video_hidden_states = attended_vq
            text_video_attention = torch.softmax(torch.matmul(text_hidden_states, video_hidden_states.transpose(2,1)), -1)
            video_text_attention = torch.softmax(torch.matmul(video_hidden_states, text_hidden_states.transpose(2,1)), -1)
            intra_video_attention = torch.softmax(intra_video_attention, -1)
            intra_text_attention = torch.softmax(intra_text_attention, -1)

            c3_loss = -F.kl_div(torch.softmax(torch.matmul(torch.matmul(video_text_attention, intra_text_attention), video_text_attention.transpose(2, 1))/1e-2, -1), intra_video_attention, reduction='batchmean')

        if fusion_proj is not None and answer_g.device != fusion_proj.device:
            answer_g = answer_g.to(fusion_proj.device)
        if answer is not None:
            if labels is not None:
                return fusion_proj, answer_g, c3_loss
            else:
                return fusion_proj, answer_g
        else:
            pred = (fusion_proj @ answer_g.t()) * (question_g @ answer_g.t())
            return pred

