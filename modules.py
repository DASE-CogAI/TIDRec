import numpy as np
import json
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

###斜坡函数
def swish(x):
    return x*torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Embeddings(nn.Module):
    def __init__(self, args) -> None:
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size,args.hidden_size,padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length,args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size,eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        #[batch_size, len]
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #[batch_size,len,hidden_size]
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 原始sasrec中会根据长度调整
        # items_embeddings *= items_embeddings.shape[1] ** 0.5

        #这里直接加是个隐患
        embeddings = items_embeddings+position_embeddings

        embeddings = self.LayerNorm(self.dropout(embeddings))

        return embeddings

class SelfAttention(nn.Module):
    def __init__(self,args) -> None:
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = args.hidden_size

        self.query = nn.Linear(args.hidden_size,self.all_head_size)
        self.key = nn.Linear(args.hidden_size,self.all_head_size)
        self.value = nn.Linear(args.hidden_size,self.all_head_size)

        self.atten_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def multi_head_trans(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        ####reshape也可以，但是torch不推荐
        x = x.view(*new_shape)
        return x.permute(0,2,1,3)

    def forward(self,input_tensor,atten_mask):
        #之所以为mix是因为embedding被加一起了
        mix_query = self.query(input_tensor)
        mix_key = self.key(input_tensor)
        mix_value = self.value(input_tensor)

        query_matrix = self.multi_head_trans(mix_query)
        key_matrix = self.multi_head_trans(mix_key)
        value_matrix = self.multi_head_trans(mix_value)

        atten_scores = torch.matmul(query_matrix,key_matrix.transpose(-1,-2))
        atten_scores = atten_scores / math.sqrt(self.attention_head_size)
        atten_scores = atten_scores + atten_mask
        atten_probs = nn.Softmax(dim=-1)(atten_scores)

        atten_probs = self.atten_dropout(atten_probs)
        atten_output = torch.matmul(atten_probs,value_matrix)
        #####使用了view之前，如果用过permute和transpose就一定要contiguous
        atten_output = atten_output.permute(0,2,1,3).contiguous()
        new_context_layer_shape = atten_output.size()[:-2] + (self.all_head_size,)
        ####之前是使用cat和split进行处理的
        context_layer = atten_output.view(*new_context_layer_shape)

        #####不是forward layer，只是有做了个变化，不知何意,此外，多了一次残差#####
        hidden_state = self.dense(context_layer)
        hidden_state = self.out_dropout(hidden_state)
        hidden_state = self.LayerNorm(hidden_state+input_tensor)
        ####没有第二次的timeline mask，加上试试####

        return hidden_state

class Pointwise_layer(nn.Module):
    def __init__(self,args) -> None:
        super(Pointwise_layer,self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_state = self.dense_1(input_tensor)
        hidden_state = self.intermediate_act_fn(hidden_state)
        hidden_state = self.dense_2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.LayerNorm(hidden_state+input_tensor)

        return hidden_state


class Layer(nn.Module):
    def __init__(self,args) -> None:
        super(Layer,self).__init__()
        self.attention_layer = SelfAttention(args)
        self.pointwise = Pointwise_layer(args)

    def forward(self, hidden_state, attention_mask):
        attention_output = self.attention_layer(hidden_state, attention_mask)
        intermediate_output = self.pointwise(attention_output)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self,args) -> None:
        super(Encoder,self).__init__()

        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask,output_all_encoded_layers = True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states,attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class AttentiveSequenceFusion(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(AttentiveSequenceFusion, self).__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_size, attention_size))
        self.b1 = nn.Parameter(torch.randn(attention_size))
        self.w2 = nn.Parameter(torch.randn(hidden_size, attention_size))
        self.b2 = nn.Parameter(torch.randn(attention_size))
        self.attention_size = attention_size
        self.init_param()

    def init_param(self):
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, inputs, query):
        '''
        inputs: (batch_size, seq_len, hidden_size)
        query: (batch_size, hidden_size)
        out: (batch_size, hidden_size)
        '''
        k = torch.einsum('bsh, ha -> bsa', inputs, self.w1) + self.b1 # (batch_size, seq_len, attention_size)
        q = torch.einsum('bh, ha -> ba', query, self.w2) + self.b2 # (batch_size, attention_size)
        e = torch.bmm(k, q.unsqueeze(2)) / math.sqrt(self.attention_size) # (batch_size, seq_len, 1)
        a = torch.softmax(e, dim=1) # (batch_size, seq_len, 1)
        out = torch.sum(inputs * a, dim=1) # (batch_size, hidden_size)

        return out

def sparse_dropout(x,rate):

    random_tensor = 1-rate
    # print(x)
    noise_shape = x._indices()
    noise_shape = noise_shape.shape[1]
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    # print(dropout_mask.shape)
    i = x._indices()
    v = x._values()
    # print(i.shape)

    i = i[:,dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i,v,x.shape).to(x.device)

    out = out*(1./(1-rate))

    return out


def attention_dropout(x,rate):
    random_tensor = torch.sigmoid(rate)
    noise_shape = x._indices()
    noise_shape = noise_shape.shape[1]
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()

    i = x._indices()
    v = x._values()
    i = i[:,dropout_mask]
    v = v[dropout_mask]
    out = torch.sparse.FloatTensor(i,v,x.shape).to(x.device)

    # out = out*(1./(1-rate))
    return out


class GCN(nn.Module):
    def __init__(self, gnn_layer):
        super(GCN,self).__init__()
        self.gnn_layer = gnn_layer

    def forward(self,inputs,support):
        x = inputs.weight[1:,:]

        support = support

        ### nn.module类本就具有self.training参数，由model.train()和model.eval()控制
        if self.training:
            #和tensorflow中一样，输入的是稀疏矩阵，dropout率和矩阵中非0元素个数
            support = sparse_dropout(support, 0.2)

        x_fin = [x]
        layer = x
        for f in range(self.gnn_layer):
            layer = torch.sparse.mm(support,layer)
            layer = torch.tanh(layer) # 可调整
            x_fin+=[layer]
        x_fin = torch.stack(x_fin,dim=1)
        out = torch.sum(x_fin,dim=1)
        # out = layer
        ######################
        fin_out = torch.cat([inputs.weight[0, :].unsqueeze(dim=0),out], dim=0)

        return fin_out


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, item_dict, model_name='./bert-base-chinese', embedding_dim=128, args=None):
        super(ItemEmbeddingLayer, self).__init__()
        # self.args = args
        with open(item_dict, 'r', encoding='utf-8') as file:
            self.item_dict = json.load(file)
        # self.path = self.args.data_dir + model_name
        self.path =model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.path)
        self.model = BertModel.from_pretrained(self.path)

        # 线性层，用于将BERT输出的768维向量转换为指定的embedding维度
        self.fc = nn.Linear(768, embedding_dim)  # 默认使用768维，转换为目标维度

    def forward(self, item_ids):
        # 获取物品ID对应的物品名称
        item_names = [self.item_dict.get(item_id, None) for item_id in item_ids]

        # 处理物品名称列表，逐个tokenize
        inputs = self.tokenizer(item_names, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)

        # 获取BERT的输出
        with torch.no_grad():  # 不需要计算梯度
            outputs = self.model(**inputs)

        # 获取[CLS]标记的embedding (表示整个句子的嵌入)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # 使用线性层将768维嵌入转换为目标维度
        embeddings = self.fc(cls_embeddings)  # (batch_size, embedding_dim)

        return embeddings

class TagEmbeddingLayer(nn.Module):
    def __init__(self, tag_dict, model_name='./bert-base-chinese', embedding_dim=128, args=None):
        super(ItemEmbeddingLayer, self).__init__()
        # self.args = args
        with open(tag_dict, 'r', encoding='utf-8') as file:
            self.tag_dict = json.load(file)
        # self.path = self.args.data_dir + model_name
        self.path =model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.path)
        self.model = BertModel.from_pretrained(self.path)

        # 线性层，用于将BERT输出的768维向量转换为指定的embedding维度
        self.fc = nn.Linear(768, embedding_dim)  # 默认使用768维，转换为目标维度

    def forward(self, tag_ids):
        # 获取物品ID对应的物品名称
        tag_names = [self.tag_dict.get(tag_id, None) for tag_id in tag_ids]

        # 处理物品名称列表，逐个tokenize
        inputs = self.tokenizer(tag_names, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)

        # 获取BERT的输出
        with torch.no_grad():  # 不需要计算梯度
            outputs = self.model(**inputs)

        # 获取[CLS]标记的embedding (表示整个句子的嵌入)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # 使用线性层将768维嵌入转换为目标维度
        embeddings = self.fc(cls_embeddings)  # (batch_size, embedding_dim)

        return embeddings
if __name__ == "__main__":
    # inputs = torch.randn(2,3,4)
    # query = torch.randn(2,4)
    # fusion = AttentiveSequenceFusion(4,5)
    # out = fusion(inputs,query)
    # print(out)


    # 物品ID与物品名称映射
    item_dict = {
        1: "Laptop",
        2: "Smartphone",
        3: "Headphones",
        4: "Smartwatch"
    }

    # 创建Embedding层，目标嵌入维度为128
    embedding_layer = ItemEmbeddingLayer(item_dict, embedding_dim=128)

    # 获取物品ID 1, 2, 3的嵌入向量（批量处理）
    item_ids = [1, 2, 3]
    embeddings = embedding_layer(item_ids)

    # 打印嵌入向量
    print(embeddings)

