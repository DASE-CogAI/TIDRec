import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, GCN, AttentiveSequenceFusion, ItemEmbeddingLayer, TagEmbeddingLayer
import numpy as np


class Intrec(nn.Module):
    def __init__(self, args, mat=None) -> None:
        super(Intrec, self).__init__()
        # self.item_embeddings = ItemEmbeddingLayer(args.item_dict, model_name=args.bert, embedding_dim=args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 0为空
        self.item_position_embeddings = nn.Embedding(args.max_seq_length_item, args.hidden_size)
        self.item_sub_encoder = Encoder(args)
        self.item_gen_encoder = Encoder(args)

        # self.tag_embeddings = TagEmbeddingLayer(args.tag_dict, model_name=args.bert, embedding_dim=args.hidden_size)
        self.tag_embeddings = nn.Embedding(args.tag_size, args.hidden_size, padding_idx=0)
        self.tag_position_embeddings = nn.Embedding(args.max_seq_length_tag, args.hidden_size)
        self.tag_encoder = Encoder(args)

        self.subject_embeddings = nn.Embedding(args.subject_size, args.hidden_size, padding_idx=0)

        self.fusion = AttentiveSequenceFusion(args.hidden_size, args.attention_size)

        self.item_user_mlp = nn.Sequential(
            nn.Linear(5 * args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        # for CL
        self.gen_emb = None
        self.sub_emb = None

        ###### this part is for GCN ######
        idx = torch.LongTensor(mat[0]).transpose(0, 1)
        val = torch.FloatTensor(mat[1])
        self.mat = torch.sparse_coo_tensor(idx, val, mat[2]).to('cuda')
        self.gcn = GCN(args.gnn_layer)
        ##################################

        self.item_layernorm = LayerNorm(args.hidden_size)
        self.item_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.tag_layernorm = LayerNorm(args.hidden_size)
        self.tag_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        ##apply(fn)会递归地将函数fn应用到父模块的每个子模块submodule，也包括model这个父模块自身。
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # pass
            module.weight.data.normal_(mean=0.0, std=0.02)  # 初始化可以考虑调整
            # nn.init.xavier_normal_(module.weight)  # 收敛快，但性能不一定好

        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_gcn_embedding(self):
        ### this is for gcn, 用于计算candidate items表示
        mat = self.mat
        embeddings_gcn = self.gcn(self.item_embeddings, mat)
        self.embeddings_gcn = embeddings_gcn
        return embeddings_gcn

    def item_add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        embeddings_gcn = self.get_gcn_embedding()
        item_embeddings = embeddings_gcn[sequence]

        item_position_embeddings = self.item_position_embeddings(position_ids)
        item_sequence_emb = item_embeddings + item_position_embeddings
        item_sequence_emb = self.item_layernorm(item_sequence_emb)
        item_sequence_emb = self.item_dropout(item_sequence_emb)

        return item_sequence_emb

    def tag_add_position_embedding(self, tags):
        seq_length = tags.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tags.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tags)

        tag_embeddings = self.tag_embeddings(tags)
        tag_position_embeddings = self.tag_position_embeddings(position_ids)
        tag_sequence_emb = tag_embeddings + tag_position_embeddings
        tag_sequence_emb = self.tag_layernorm(tag_sequence_emb)
        tag_sequence_emb = self.tag_dropout(tag_sequence_emb)

        return tag_sequence_emb

    def intent_learning(self, input_id, encoder):
        # User Sequence Modeling
        attention_mask = (input_id > 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2)  # torch.int64 [batch_size, 1, 1, max_seq_length]

        extended_attention_mask = torch.tril(
            extended_attention_mask.expand((-1, -1, input_id.size(-1), -1))
        ) # torch.int64 [batch_size, 1, max_seq_length, max_seq_length] 三角矩阵

        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)

        # GCN + Position Embeddings
        item_sequence_emb = self.item_add_position_embedding(input_id)

        # Item Transformer
        item_seq_all_output = encoder(item_sequence_emb, extended_attention_mask) # [layer_num, batch_size, max_seq_length, hidden_size]
        item_sequence_output = item_seq_all_output[-1] # [batch_size, max_seq_length, hidden_size]
        # item_sequence_output = item_sequence_output[:, -1, :]  # [batch_size, hidden_size]

        return item_sequence_output

    def tag_seq_encoder(self, tags, encoder, subject_embeddings):
        attention_mask = (tags > 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2)  # torch.int64 [batch_size, 1, 1, max_seq_length]

        extended_attention_mask = torch.tril(
            extended_attention_mask.expand((-1, -1, tags.size(-1), -1))
        ) # torch.int64 [batch_size, 1, max_seq_length, max_seq_length] 三角矩阵

        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)

        # GCN + Position Embeddings
        tag_sequence_emb = self.tag_add_position_embedding(tags)

        # Item Transformer
        tag_seq_all_output = encoder(tag_sequence_emb, extended_attention_mask) # [layer_num, batch_size, max_seq_length, hidden_size]
        tag_sequence_output = tag_seq_all_output[-1] # [batch_size, max_seq_length, hidden_size]

        if not self.args.ablation_tag: # 加入消融实验验证长短期的有效性
            long_term_tag = tag_sequence_output.mean(dim=1) # 全部聚合 [batch_size, hidden_size]
            short_term_tag = tag_sequence_output[:, -1, :] # 取最后1个作为short_term [batch_size, hidden_size]
            return torch.cat([long_term_tag, short_term_tag], dim=-1)
        return torch.cat([tag_sequence_output.mean(dim=1), tag_sequence_output.mean(dim=1)], dim=-1)

    def forward(self, items_sub, items_gen, tag_ids, subject_ids):
        item_sub_intent = self.intent_learning(items_sub, self.item_sub_encoder)
        item_gen_intent = self.intent_learning(items_gen, self.item_gen_encoder)
        gen_intent_fin = item_gen_intent[:, -1, :]  # [batch_size, hidden_size]
        self.gen_emb = gen_intent_fin

        subject_embeddings = self.subject_embeddings(subject_ids) # [batch_size, hidden_size]
        if not self.args.ablation_sub:  # 消融实验 验证attentive sequence fusion的有效性
            sub_intent_fin = self.fusion(item_sub_intent, subject_embeddings)  # [batch_size, hidden_size]
        else:
            # sub_intent_fin = item_sub_intent[:, -1, :]
            sub_intent_fin = item_sub_intent.mean(dim=1)  # [batch_size, hidden_size]
        self.sub_emb = sub_intent_fin

        tag_output_fin = self.tag_seq_encoder(tag_ids, self.tag_encoder, subject_embeddings) # [batch_size, hidden_size*2]

        user_features = torch.cat([subject_embeddings, sub_intent_fin, gen_intent_fin, tag_output_fin], dim=-1)
        output = self.item_user_mlp(user_features) # [batch_size, hidden_size]

        return output

    # only test online
    def predict(self, input_ids, tag_ids, school_ids, area_ids, subject_ids):
        # 1. 获取用户的最后一个物品序列的表示
        sequence_output = self.forward(input_ids, tag_ids, school_ids, area_ids,
                                       subject_ids)  # [batch_size, hidden_size]

        # 2. 获取所有物品的嵌入
        item_embeddings = self.get_gcn_embedding()  # [item_size, hidden_size]

        # 3. 计算用户表示与物品嵌入的点积，得到物品得分
        scores = torch.matmul(sequence_output, item_embeddings.transpose(0, 1))  # [batch_size, item_size]
        scores[:, 0] = -np.inf  # 避免推荐第一个填充元素
        return scores  # 返回每个用户对所有物品的得分






