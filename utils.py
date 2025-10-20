import numpy as np
import math
import random
import os
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch

# def gather_nd(params, indices):
#     '''
#     example:
#     # batch_size = 2, 每个batch有3个元素，每个元素有2个特征
#     params = torch.tensor([
#         [[1, 2], [3, 4], [5, 6]],   # batch 0
#         [[7, 8], [9, 10], [11, 12]]  # batch 1
#     ])
#
#     # indices也是三维，用于选择特定位置
#     indices = torch.tensor([
#         [[0, 1], [1, 0]],   # batch 0的选择
#         [[1, 2], [0, 1]]    # batch 1的选择
#     ])
#     return:
#     '''
#     out_shape = indices.shape[:-1]  # 输出形状为：[user_num, 100]
#     indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fringe
#     ndim = indices.shape[0]
#     indices = indices.long() - 1 # 下标从1开始的，不减一会出错
#
#     idx = torch.zeros_like(indices[0], device=indices.device).long()  # 初始化索引为0，形状为：(user_num, 100)
#     m = 1
#     for i in range(ndim)[::-1]:
#         idx += indices[i] * m  # 计算每个维度的索引
#         m *= params.size(i)  # params.size(0)的长度为user_num，params.size(1)的长度为kc_num
#
#     out = torch.take(params, idx)
#     return out.view(out_shape)

def gather_nd(rate, negative):
    batch_size = rate.shape[0]

    # Create a batch index tensor
    batch_index = torch.arange(batch_size, device=rate.device)[:, None]

    # Combine batch index and item indices
    # indices = torch.cat([batch_index, negative[:, :, 1].long() - 1], dim=1)
    indices = negative[:, :, 1].long()
    # Use torch.gather() to extract the values
    test = torch.gather(rate, 1, indices)

    return test

def hr(rate, negative, length, k=5, ans=999):
    test = gather_nd(rate, negative) # 100个测试集的预测评分
    topk = torch.topk(test, k).indices # 返回前k个最大元素的索引数组
    isIn = (topk == ans).float() # 判断99号索引是否在最大元素索引数组中，得到判断数组
    row = torch.sum(isIn, dim=1) # 对每个用户的判断数组求和，若命中则和为1
    all_ = torch.sum(row) # 求所有用户的命中个数
    # hr = all_ / length # 所有用户的命中个数/用户数
    try:
        return all_.item()
    except RuntimeError:
        return 0  # 如果发生错误，返回 0 或其他合理的默认值


def mrr(rate, negative, length, ans=999):
    test = gather_nd(rate, negative)
    topk = torch.topk(test, 100).indices
    n = torch.where(topk == ans)[1] # 查找99号元素是否在索引数组中，并返回True所在位置
    new_n = torch.add(n, 1)
    mrr_ = torch.sum(torch.reciprocal(new_n.float())) # 位置倒数的求和
    mrr = mrr_ / length
    try:
        return mrr_.item()
    except RuntimeError:
        return 0  # 如果发生错误，返回 0 或其他合理的默认值


def ndcg(rate, negative, length, k=5, ans=999):
    test = gather_nd(rate, negative) # 测试样本在rate中的预测评分
    topk = torch.topk(test, k).indices # 返回前K个最大元素的索引数组
    n = torch.where(topk == ans)[1] # 查找正例是否在索引数组中，若在则返回正例所在的位置
    ndcg_ = torch.sum(torch.log2(torch.tensor(2.0).to(n.device)) / torch.log2(torch.add(n, 2).float()))
    # ndcg = ndcg_ / length
    try:
        return ndcg_.item()
    except RuntimeError:
        return 0  # 如果发生错误，返回 0 或其他合理的默认值

def env(rate, negative):
    length = negative.shape[0] # length == batch_size
    hrat5 = hr(rate, negative, length, k=5) # 5\10\20
    hrat10 = hr(rate, negative, length, k=10)
    hrat20 = hr(rate, negative, length, k=20)
    ndcg5 = ndcg(rate, negative, length, k=5)
    ndcg10 = ndcg(rate, negative, length, k=10)
    ndcg20 = ndcg(rate, negative, length, k=20)
    mr = mrr(rate, negative, length)
    return hrat5, hrat10, hrat20, ndcg5, ndcg10, ndcg20, mr

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ',1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set.update(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 1

    return user_seq, max_item

def get_train_seqs(train_seq):
    ### train_seq: list[list]
    ans = []
    for seq in train_seq:
        if len(seq)<=2:
            continue
        ###recbole中最短序列是2，也就是[1,2]用1预测2
        for i in range(2,len(seq)+1):
            ans.append(seq[:i])

    return ans

def get_user_seqs_split_org(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ',1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        # item_set = item_set | set(items)
        item_set.update(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items_with_padding = max_item + 1
    train_seq, vaild_seq = [x[:-2] for x in user_seq],[x[:-1] for x in user_seq]
    # print(len(train_seq))
    train_seq = get_train_seqs(train_seq)
    test_seq = user_seq
    return train_seq, vaild_seq, test_seq, max_item


def split_train_seqs(train_seq, train_tag, train_profile):
    ### train_seq: list[list]
    seqs_aug, tags_aug, profile_aug = [],[],[]
    for seq, tag, profile in zip(train_seq, train_tag, train_profile):
        if len(seq) <= 2:
            continue
        for i in range(2, len(seq) + 1):
            seqs_aug.append(seq[:i])
            tags_aug.append(tag[:i])
            profile_aug.append(profile) # 子序列的profile保持一致

    return seqs_aug, tags_aug, profile_aug

def split_seqs(train_seq, train_tag, train_profile):
    ### train_seq: list[list]
    seqs_aug, tags_aug, profile_aug = [],[],[]
    for seq, tag, profile in zip(train_seq, train_tag, train_profile):
        if len(seq) <= 2:
            continue
        for i in range(2, len(seq) + 1):
            seqs_aug.append(seq[:i])
            tags_aug.append(tag[:i])
            profile_aug.append(profile) # 子序列的profile保持一致

    return seqs_aug, tags_aug, profile_aug


def get_user_seqs_split(item_seq_file, tag_seq_file, profile_file, aug=False):
    item_lines = open(item_seq_file).readlines()
    user_indice = []
    user_seq = []
    for line in item_lines:
        user, items = line.strip().split(' ',1)
        user_indice.append(int(user)-1) # 用户下标从1开始
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)

    tag_lines = open(tag_seq_file).readlines()
    tag_seq = []
    for line in tag_lines:
        user, tags = line.strip().split(' ',1)
        tags = tags.split(' ')
        tags = [int(tag) for tag in tags]
        tag_seq.append(tags)

    user_attri_data = pd.read_csv(profile_file)
    # user_profiles = []
    # for index, user_i in user_attri_data.iterrows():
    #     user_profiles.append([int(user_i['school_id']),int(user_i['area_id']),int(user_i['subject'])])
    user_profiles = user_attri_data.loc[user_indice, ['school_id', 'area_id', 'subject']].apply(pd.to_numeric).values.tolist()
    # 不用分割训练集和验证集
    # num_users = len(item_lines)
    # total_idx = np.random.permutation(num_users)
    # train_idx = total_idx[ : int(split_ratio[0]*num_users)]
    # test_idx = total_idx[int((split_ratio[0])*num_users) : ]
    #
    # train_seq= [user_seq[x] for x in train_idx]
    # train_tag= [tag_seq[x] for x in train_idx]
    # train_profile= [user_profiles[x] for x in train_idx]
    if aug:
        train_seq_aug, train_tag_aug, train_profile_aug  = split_train_seqs(user_seq, tag_seq, user_profiles)
        return train_seq_aug, train_tag_aug, train_profile_aug
    # test_seq = [user_seq[x] for x in test_idx]
    # test_tag = [tag_seq[x] for x in test_idx]
    # test_profile = [user_profiles[x] for x in test_idx]
    return user_seq, tag_seq, user_profiles

def get_user_seqs_split_atr_seq(item_seq_file, tag_seq_file, profile_file, item_atr_file):
    '''
    input:
        item_seq_file: 用户序列文件
            line: [user_id item1 item2 ...]
        tag_seq_file: 用户tag序列文件
            line: [user_id tag1 tag2 ...]
        profile_file: 用户属性文件
            line: [school_id area_id subject] 只使用subject
        item_atr_file: [gen_item, gen_tag] 存储general属性对应item和tag的id
    output:
        user_seq_sub: [num_user, num_seq_sub, num_item_sub]
        user_seq_gen: [num_user, num_seq_gen, num_item_gen]
        tag_seq: [num_user, num_tag]
        user_sub: [num_user, subject_id]
    '''
    gen_item = set(np.load(item_atr_file)["gen_item"])
    gen_tag = set(np.load(item_atr_file)["gen_tag"])
    item_lines = open(item_seq_file).readlines()
    user_seq_sub, user_seq_gen = [], []
    user_indice = []
    for line in item_lines:
        user, items = line.strip().split(' ',1)
        user_indice.append(int(user))
        items = items.split(' ')
        items_sub = []
        items_gen = []
        user_sub, user_gen = [], []
        flag = -1 # 用于判断item类型是否连续
        for item in items:
            if int(item) in gen_item:
                if flag == 1:
                    user_sub.append(items_sub)
                    items_sub = []
                flag = 0
                items_gen.append(int(item))
            else:
                if flag == 0:
                    user_gen.append(items_gen)
                    items_gen = []
                flag = 1
                items_sub.append(int(item))
        if len(items_gen):
            user_gen.append(items_gen)
        if len(items_sub):
            user_sub.append(items_sub)

        user_seq_sub.append(user_sub)
        user_seq_gen.append(user_gen)


    tag_lines = open(tag_seq_file).readlines()
    tag_seq = []
    for line in tag_lines:
        user, tags = line.strip().split(' ',1)
        tags = tags.split(' ')
        tags = [int(tag) for tag in tags]
        tag_seq.append(tags)


    user_attri_data = pd.read_csv(profile_file)
    user_sub = user_attri_data.loc[user_indice, 'subject'].tolist()

    return items_sub, items_gen, tag_seq, user_sub

def get_user_seqs_split_atr(item_seq_file, tag_seq_file, profile_file, item_atr_file):
    '''
    output:
        user_seq_sub: [num_user, num_item_sub]
        user_seq_gen: [num_user, num_item_gen]
        tag_seq: [num_user, num_tag]
        num_item_sub + num_item_gen + 1 = num_tag 最后一个item作为target，最后一个tag在dataset里处理掉
        user_sub: [num_user]
        item_target: [num_user]
    '''
    gen_item = set(np.load(item_atr_file)["gen_item"])
    item_lines = open(item_seq_file).readlines()
    user_seq_sub, user_seq_gen = [], []
    user_indice = []
    item_target = []
    for line in item_lines:
        user, items = line.strip().split(' ',1)
        user_indice.append(int(user) - 1) # 用户下标从1开始
        items = items.split(' ')
        items_sub = []
        items_gen = []
        for idx, item in enumerate(items):
            if idx == len(items) - 1:
                item_target.append(int(item))
                continue
            if int(item) in gen_item:
                items_gen.append(int(item))
            else:
                items_sub.append(int(item))
        user_seq_sub.append(items_sub)
        user_seq_gen.append(items_gen)

    tag_lines = open(tag_seq_file).readlines()
    tag_seq = []
    for line in tag_lines:
        user, tags = line.strip().split(' ',1)
        tags = tags.split(' ')
        tags = [int(tag) for tag in tags]
        tag_seq.append(tags)

    user_attri_data = pd.read_csv(profile_file)
    user_sub = user_attri_data.loc[user_indice, 'subject'].tolist()
    return user_seq_sub, user_seq_gen, tag_seq, user_sub, item_target

def get_user_seqs_split_atr_aug(item_seq_file, tag_seq_file, profile_file, item_atr_file):
    item_lines = open(item_seq_file).readlines()
    user_indice = []
    user_seq = []
    for line in item_lines:
        user, items = line.strip().split(' ',1)
        user_indice.append(int(user)-1) # 用户下标从1开始
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)

    tag_lines = open(tag_seq_file).readlines()
    tag_seq = []
    for line in tag_lines:
        user, tags = line.strip().split(' ',1)
        tags = tags.split(' ')
        tags = [int(tag) for tag in tags]
        tag_seq.append(tags)

    user_attri_data = pd.read_csv(profile_file)
    user_sub = user_attri_data.loc[user_indice, 'subject'].tolist()
    train_seq_aug, train_tag_aug, train_profile_aug = split_seqs(user_seq, tag_seq, user_sub)

    gen_item = set(np.load(item_atr_file)["gen_item"])
    user_seq_sub, user_seq_gen = [], []
    item_target = []
    for items in train_seq_aug:
        items_sub = []
        items_gen = []
        for idx, item in enumerate(items):
            if idx == len(items) - 1:
                item_target.append(int(item))
                continue
            if int(item) in gen_item:
                items_gen.append(int(item))
            else:
                items_sub.append(int(item))
        user_seq_sub.append(items_sub)
        user_seq_gen.append(items_gen)

    return user_seq_sub, user_seq_gen, train_tag_aug, train_profile_aug, item_target


def get_user_seqs_test(item_seq_file, tag_seq_file, profile_file):
    item_lines = open(item_seq_file).readlines()
    user_seq = []
    for line in item_lines:
        user, items = line.strip().split(' ',1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)

    tag_lines = open(tag_seq_file).readlines()
    tag_seq = []
    for line in tag_lines:
        user, tags = line.strip().split(' ',1)
        tags = tags.split(' ')
        tags = [int(tag) for tag in tags]
        tag_seq.append(tags)

    user_attri_data = pd.read_csv(profile_file)
    user_profiles = []
    for index, user_i in user_attri_data.iterrows():
        user_profiles.append([int(user_i['school_id']),int(user_i['area_id']),int(user_i['subject'])])

    return user_seq, tag_seq, user_profiles

def get_user_seqs_split_without_aug(data_file, intention_file, split_ratio = [0.8,0.1,0.1]):
    lines = open(data_file).readlines()
    intention_lines = open(intention_file).readlines()
    user_seq = []
    intention_seq = []
    item_set = set()
    inten_set = set()
    max_len_inten = 0
    for line, inten_line in zip(lines, intention_lines):
        user, items = line.strip().split(' ',1)
        items = items.split(' ')
        inten_line = inten_line.split(' ')
        items = [int(item) for item in items]
        inten_line = [int(inten) for inten in inten_line]
        max_len_inten = max(max_len_inten,len(inten_line))
        user_seq.append(items)
        intention_seq.append(inten_line)
        # item_set = item_set | set(items)
        item_set.update(items)
        inten_set.update(inten_line)
    max_item = max(item_set)
    max_inten = max(inten_set)
    # print(max_inten)

    num_users = len(lines)
    total_idx = np.random.permutation(num_users)
    train_idx = total_idx[ : int(split_ratio[0]*num_users)]
    vaild_idx = total_idx[int(split_ratio[0]*num_users) : int((split_ratio[0]+split_ratio[1])*num_users)]
    test_idx = total_idx[int((split_ratio[0]+split_ratio[1])*num_users) : ]
    train_seq, vaild_seq = [user_seq[x] for x in train_idx],[user_seq[x] for x in vaild_idx]
    train_inten_seq, vaild_inten_seq = [intention_seq[x] for x in train_idx], [intention_seq[x] for x in vaild_idx]
    # print(len(train_seq))
    # train_seq = get_train_seqs(train_seq)
    test_seq = [user_seq[x] for x in test_idx]
    test_inten_seq = [intention_seq[x] for x in test_idx]
    return train_seq, vaild_seq, test_seq, max_item, train_inten_seq, vaild_inten_seq, test_inten_seq, max_inten

def cross_entropy(seq_out, pos_ids, neg_ids,model,args):
        # [batch seq_len hidden_size]
        pos_emb = model.item_embeddings(pos_ids)
        neg_emb = model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def mrr_at_k(actual, predicted, topk):
    mrr_sum = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        for j in range(topk):
            if j < len(predicted[i]) and predicted[i][j] in actual[i]:
                mrr_sum += 1.0 / (j + 1)
                break
    return mrr_sum / num_users

def ndcg_k(actual, predicted, topk): #归一化折损累积增益，对排名靠前的推荐项给予更高的权重，衡量推荐系统在前topk个推荐项中的排序质量
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        # res += dcg_k
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i]) # 转换为集合类型以便进行集合运算
        pred_set = set(predicted[i][:topk]) # 当前用户的前k个推荐项目集合
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set)) # len(act_set & pred_set)：实际感兴趣项目和推荐项目集合的交集数量
            true_users += 1
    return sum_recall / true_users

def get_sp_adj_matrix(matrix):
    row_sum = np.array(matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())  # 开根号对角阵算D^{-1/2}
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(
        matrix.dot(degree_mat_inv_sqrt)).tocoo()  # D^{-1/2}AD^{-1/2}，tocoo变为稀疏形式
    # return rel_matrix_normalized
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    # return indices
    values = rel_matrix_normalized.data.astype(np.float32)  # D^{-1/2}AD^{-1/2}的值
    shape = rel_matrix_normalized.shape
    return indices, values, shape

# def get_torch_adj_matrix(idx,val,num_node):
#     ###beauty:12101,sports:18357,toys:11924,yelp:20033
#     deg = scatter_add(val, idx[1], dim=0, dim_size=num_node)+1e-24
#     deg_inv_sqrt = deg.pow_(-0.5)
#     val_norm = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]
#     return val_norm

def get_orign_adj_matrix(matrix):
    mat = matrix.tocoo()
    indices = np.vstack((mat.row, mat.col))
    idx = torch.LongTensor(indices)
    val = torch.FloatTensor(mat.data)
    shape = mat.shape
    # torch_mat = torch.sparse_coo_tensor(idx,val,mat.shape).to('cuda')
    return idx, val, shape

# trans_mat = sp.load_npz('mat_noweight.npz')
# _,vall,_ = get_sp_adj_matrix(trans_mat)
# print(vall)


# mat = trans_mat.tocoo()
# indices = np.vstack((mat.row, mat.col))
# idx = torch.LongTensor(indices)
# val = torch.FloatTensor(mat.data)
# torch_mat = torch.sparse_coo_tensor(idx,val,mat.shape)
# deg = scatter_add(val, idx[1], dim=0, dim_size=12101)+1e-24
# deg_inv_sqrt = deg.pow_(-0.5)
# a = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]
# print(a)

class EarlyStopping(object):
    def __init__(self,output_dir,patience=10,verbose=False) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter1 = 0
        self.counter2 = 0
        self.best_score = None
        self.best_score_neg = None
        self.early_stop = False
        self.output_dir = output_dir

    def compare(self, score):
        # 多个指标
        # for i in range(len(score)):
        #     if score[i]>self.best_score[i]:
        #         return False

        # 单个指标
        if score > self.best_score:
            return False
        return True

    def __call__(self, score1, score2, model):
        if self.best_score is None:
            self.best_score = score1
            self.save_checkpoint(score1, model)
        elif score1 <= self.best_score:
            self.counter1 +=1
            print(f'EarlyStopping counter: {self.counter1} out of {self.patience}')
            if self.counter1 >= self.patience and self.counter2 >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score1
            self.save_checkpoint(score1, model)
            self.counter1 = 0
        if self.best_score_neg is None:
            self.best_score_neg = score2
            self.save_checkpoint(score2, model)
        elif score2 <= self.best_score_neg:
            self.counter2 +=1
            print(f'EarlyStopping counter: {self.counter2} out of {self.patience}')
            if self.counter1 >= self.patience and self.counter2 >= self.patience:
                self.early_stop = True
        else:
            self.best_score_neg = score2
            self.save_checkpoint(score2, model)
            self.counter2 = 0

    # def __call__(self, model):
    #     self.save_checkpoint(model)

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ... but no saving')
        # np.savetxt(os.path.join(self.output_dir, 'best_k_prediction.csv'), best_pred, delimiter=',', fmt='%d')
        # np.save(os.path.join(self.output_dir, 'all_prediction_mrr%s.npy'%str(score)[:3]), best_pred)
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'best_model_mrr%s.pt'%str(score)[:3]))

