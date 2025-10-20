import random
import torch
import numpy as np
import os
from torch.utils.data import Dataset


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class SASRecDataset(Dataset):
    def __init__(self,args,user_seq, tag_seq=None, user_profiles = None, test_neg_items = None, data_type = 'train') -> None:
        self.args = args
        self.user_seq = user_seq
        self.tag_seq = tag_seq  # 新增 tag_seq 作为标签序列
        self.user_profile = user_profiles # 新增user_profile
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        user_index = index
        school_id = self.user_profile[index][0]
        area_id = self.user_profile[index][1]
        subject_id = self.user_profile[index][2]
        items = self.user_seq[index]
        tags = self.tag_seq[index]

        assert self.data_type in {"train","valid","test","test_online"}

        if self.data_type == 'train':
            input_ids = items[:-1]
            target_pos = items[1:]
            tag_ids = tags[:-1]  # 对应的 tag_ids
            answer = [items[-1]]
        elif self.data_type == 'valid':
            input_ids = items[:-1]
            target_pos = items[1:]
            tag_ids = tags[:-1]  # 对应的 tag_ids
            answer = [items[-1]]
        elif self.data_type == 'test':
            input_ids = items[:-1]
            target_pos = items[1:]
            tag_ids = tags[:-1]  # 对应的 tag_ids
            answer = [items[-1]]
        elif self.data_type == 'test_online':
            input_ids = items  # 需要输入全部test
            target_pos = items
            tag_ids = tags  # 对应的 tag_ids
            answer = [items[-1]]

        target_neg = []
        seq_set = set(items)

        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size)) # 训练阶段负采样

        # 不足填0
        pad_len = self.max_len-len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # 多余截断
        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        tag_ids = [0] * pad_len + tag_ids  # 用 0 填充
        tag_ids = tag_ids[-self.max_len:]  # 截断到 max_len

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        assert len(tag_ids) == self.max_len  # 确保 tag_ids 长度一致

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_index, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),  # tag_ids 添加到输出
                torch.tensor(school_id, dtype=torch.long),  # school_id 添加到输出
                torch.tensor(area_id, dtype=torch.long),  # area_id 添加到输出
                torch.tensor(subject_id, dtype=torch.long),  # subject_id 添加到输出
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),  # tag_ids 添加到输出
                torch.tensor(school_id, dtype=torch.long),  # school_id 添加到输出
                torch.tensor(area_id, dtype=torch.long),  # area_id 添加到输出
                torch.tensor(subject_id, dtype=torch.long),  # subject_id 添加到输出
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)


class Userdataset(Dataset):
    def __init__(self,args, user_seq_sub, user_seq_gen, tag_seq=None, user_sub = None, target = None,test_neg_items = None, data_type = 'train'):
        self.args = args
        self.user_seq_sub = user_seq_sub
        self.user_seq_gen = user_seq_gen
        self.tag_seq = tag_seq
        self.user_sub = user_sub
        self.target = target
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len_item = args.max_seq_length_item
        self.max_len_tag = args.max_seq_length_tag
        self.device = "cpu"

        # self._pad_and_truncate()

        # if self.args.contrast and not os.path.exists(self.args.data_dir + '/gen_sim_matrix.pt'):
        #     self.__dataprocess__(self.user_seq_gen, self.user_seq_sub, self.user_sub, self.args.data_dir)

    def _pad_and_truncate(self):
        # 填充和截断数据，确保序列的长度符合最大长度要求
        self.user_seq_sub = self._pad_sequences(self.user_seq_sub, self.max_len_item)
        self.user_seq_gen = self._pad_sequences(self.user_seq_gen, self.max_len_item)
        self.tag_seq = self._pad_sequences(self.tag_seq, self.max_len_tag, padding_value=0)

    def _pad_sequences(self, sequences, max_len, padding_value=0):
        # 对每个序列进行填充和截断
        padded_sequences = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded_seq = [padding_value] * pad_len + seq
            padded_seq = padded_seq[-max_len:]  # 截断
            padded_sequences.append(padded_seq)
        return torch.tensor(padded_sequences, dtype=torch.long, device=self.device)


    def compute_jaccard_matrix(self, sequences):
        """
        计算所有序列之间的Jaccard相似度矩阵(交并比)
        sequences : [batch_size, hidden_size]
        """
        num_sequences = len(sequences)
        # sim_matrix = torch.zeros((num_sequences, num_sequences), dtype=torch.float32, device=self.device)

        # 计算交集
        # 使用广播机制批量计算 seq1 == seq2 的交集（相等且非零）
        non_zero_mask = (sequences != 0)  # 非零掩码
        intersection_matrix = torch.eq(sequences.unsqueeze(1), sequences.unsqueeze(0))  # 计算相等的位置
        intersection_matrix = intersection_matrix & non_zero_mask.unsqueeze(1) & non_zero_mask.unsqueeze(
            0)  # 交集：元素相等且非零
        intersection_matrix = intersection_matrix.float().sum(dim=2)  # 计算交集的大小

        # 计算并集：去除零元素并计算唯一的非零元素数量
        union_matrix = torch.cat([sequences.unsqueeze(1).expand(-1, num_sequences, -1),
                                  sequences.unsqueeze(0).expand(num_sequences, -1, -1)], dim=2)  # 合并两个序列
        union_matrix = torch.unique(union_matrix, dim=2)  # 获取每对序列的唯一非零元素

        # 计算并集的大小
        union_matrix = (union_matrix != 0).sum(dim=2).float()  # 非零元素的数量

        # 计算 Jaccard 相似度
        sim_matrix = intersection_matrix / union_matrix
        sim_matrix[union_matrix == 0] = 0  # 如果并集为 0（两个序列都全是填充值），设置相似度为 0

        return sim_matrix

    def __dataprocess__(self, gen_seq, sub_seq, subject_ids, data_dir):
        # 计算general Top-N相似序列索引
        gen_sim_matrix = self.compute_jaccard_matrix(gen_seq)

        # 计算subject Top-N相似序列索引
        # 初始化全局相似度矩阵
        sub_sim_matrix = torch.zeros((len(sub_seq), len(sub_seq)), dtype=torch.float, device=self.device)

        # 按subject_id分组计算相似度
        unique_subjects = torch.unique(subject_ids)

        for sub_id in unique_subjects:
            # 获取当前subject的所有样本索引
            mask = (subject_ids == sub_id).nonzero().squeeze(1).tolist()
            group_size = len(mask)

            if group_size > 1:
                # 计算组内Jaccard相似度
                group_seqs = sub_seq[mask]
                group_sim_matrix = self.compute_jaccard_matrix(group_seqs)

                # 将组内相似度填充到全局相似度矩阵中
                for i, idx1 in enumerate(mask):
                    for j, idx2 in enumerate(mask):
                        sub_sim_matrix[idx1, idx2] = group_sim_matrix[i, j]

        torch.save(gen_sim_matrix, data_dir + '/gen_sim_matrix.pt')
        torch.save(sub_sim_matrix, data_dir + '/sub_sim_matrix.pt')

    def __len__(self):
        assert len(self.user_seq_sub) == len(self.user_seq_gen) # sub和gen的长度必须是一致的
        return len(self.user_seq_sub)

    def __getitem__(self, index):
        user_index = index
        subject_id = self.user_sub[index]
        items_sub = self.user_seq_sub[index]
        items_gen = self.user_seq_gen[index]
        tags = self.tag_seq[index]
        answer = [self.target[index]]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == 'train':
            tag_ids = tags[:-1]  # 对应的 tag_ids
        elif self.data_type == 'valid':
            tag_ids = tags[:-1]  # 对应的 tag_ids
        elif self.data_type == 'test':
            tag_ids = tags[:-1]  # 对应的 tag_ids

        # 不足填0
        pad_len = self.max_len_item - len(items_sub)
        items_sub = [0] * pad_len + items_sub
        pad_len = self.max_len_item - len(items_gen)
        items_gen = [0] * pad_len + items_gen
        # target_pos = [0] * pad_len + target_pos
        # target_neg = [0] * pad_len + target_neg

        # 多余截断
        items_sub = items_sub[-self.max_len_item:]
        items_gen = items_gen[-self.max_len_item:]
        # target_pos = target_pos[-self.max_len:]
        # target_neg = target_neg[-self.max_len:]

        pad_len = self.max_len_tag - len(tag_ids)
        tag_ids = [0] * pad_len + tag_ids  # 用 0 填充
        tag_ids = tag_ids[-self.max_len_tag:]  # 截断到 max_len

        assert len(items_sub) == self.max_len_item
        assert len(items_gen) == self.max_len_item
        assert len(tag_ids) == self.max_len_tag  # 确保 tag_ids 长度一致

        # target_neg = []
        # seq_set = set(items_sub.extend(items_gen))

        # 没有用到
        # for _ in tag_ids:
        #     target_neg.append(neg_sample(seq_set, self.args.item_size))  # 训练阶段负采样

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_index, dtype=torch.long),  # user_id for testing
                torch.tensor(items_sub, dtype=torch.long),
                torch.tensor(items_gen, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),  # tag_ids 添加到输出
                torch.tensor(subject_id, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_index, dtype=torch.long),  # user_id for testing
                torch.tensor(items_sub, dtype=torch.long),
                torch.tensor(items_gen, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),  # tag_ids 添加到输出
                torch.tensor(subject_id, dtype=torch.long),  # subject_id 添加到输出
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors


class BatchIndexCollate:
    def __init__(self, top_n: int = 1, pad_idx: int = -1, device='cuda'):
        """
        自定义collate_fn，用于计算基于Jaccard相似度的Top-N相似序列。

        参数:
            top_n: 每组返回的相似序列数量
            pad_idx: 用于填充的无效索引值(默认-1)
            device: 设备配置（默认为'cuda'）
        """
        # self.top_n = top_n
        self.pad_idx = pad_idx
        self.device = device
        # self.sub_sim = torch.load(load_path + '/sub_sim_matrix.pt')
        # self.gen_sim = torch.load(load_path + '/gen_sim_matrix.pt')

    def compute_jaccard_matrix(self, sequences):
        """
        计算所有序列之间的Jaccard相似度矩阵(交并比)
        sequences : [batch_size, hidden_size]
        """
        num_sequences = len(sequences)
        # sim_matrix = torch.zeros((num_sequences, num_sequences), dtype=torch.float32, device=self.device)

        # 计算交集
        # 使用广播机制批量计算 seq1 == seq2 的交集（相等且非零）
        non_zero_mask = (sequences != 0)  # 非零掩码
        intersection_matrix = torch.eq(sequences.unsqueeze(1), sequences.unsqueeze(0))  # 计算相等的位置
        intersection_matrix = intersection_matrix & non_zero_mask.unsqueeze(1) & non_zero_mask.unsqueeze(
            0)  # 交集：元素相等且非零
        intersection_matrix = intersection_matrix.float().sum(dim=2)  # 计算交集的大小

        # 计算并集：去除零元素并计算唯一的非零元素数量
        union_matrix = (sequences.unsqueeze(1) != 0) | (sequences.unsqueeze(0) != 0)  # 计算非零元素的并集
        union_matrix = union_matrix.sum(dim=2).float()  # 计算并集的大小

        # 计算 Jaccard 相似度
        sim_matrix = intersection_matrix / union_matrix
        sim_matrix[union_matrix == 0] = 0  # 如果并集为 0（两个序列都全是填充值），设置相似度为 0

        return sim_matrix

    def __dataprocess__(self, gen_seq, sub_seq, subject_ids):
        # 计算general Top-N相似序列索引
        gen_sim_matrix = self.compute_jaccard_matrix(gen_seq)
        gen_sim_matrix.fill_diagonal_(1)

        # 计算subject Top-N相似序列索引
        # 初始化全局相似度矩阵
        sub_sim_matrix = torch.zeros((len(sub_seq), len(sub_seq)), dtype=torch.float, device=self.device)

        # 按subject_id分组计算相似度
        unique_subjects = torch.unique(subject_ids)

        for sub_id in unique_subjects:
            # 获取当前subject的所有样本索引
            mask = (subject_ids == sub_id).nonzero().squeeze(1).tolist()
            group_size = len(mask)

            if group_size > 1:
                # 计算组内Jaccard相似度
                group_seqs = sub_seq[mask]
                group_sim_matrix = self.compute_jaccard_matrix(group_seqs)

                # 将组内相似度填充到全局相似度矩阵中
                for i, idx1 in enumerate(mask):
                    for j, idx2 in enumerate(mask):
                        sub_sim_matrix[idx1, idx2] = group_sim_matrix[i, j]
        sub_sim_matrix.fill_diagonal_(1)

        return gen_sim_matrix, sub_sim_matrix

    def get_idx(self, sim_matrix, top_n=2):
        """
        获取每行的Top-N索引
        """
        top_values, top_indices = torch.topk(sim_matrix, top_n, largest=True, dim=1)

        # idx是一个按行递增的张量，表示每个样本的索引
        idx = torch.arange(sim_matrix.size(0), device=self.device)

        # 获取top1和top2的索引
        top1 = top_indices[:, 0]
        top2 = top_indices[:, 1]

        # 获取top1和top2的相似度值
        top2_values = top_values[:, 1]

        # 根据条件返回top2的索引，或者返回top1的索引
        result = torch.where(
            (top1 == idx) & (top2_values != 0),  # 如果top1是自身索引且top2的相似度不为0
            top2,  # 返回top2的索引
            top1  # 否则返回top1的索引
        )

        return result

    def __call__(self, batch):
        """
        计算每个批次的Top-N相似序列索引。
        """
        # 解包原始数据
        userid, item_sub, item_gen, tag_ids, subject_ids, answers = zip(*batch)

        # 转换为tensor并转移到指定的设备上
        userid = torch.stack(userid).to(self.device)
        item_sub = torch.stack(item_sub).to(self.device)
        item_gen = torch.stack(item_gen).to(self.device)
        tag_ids = torch.stack(tag_ids).to(self.device)
        subject_ids = torch.stack(subject_ids).to(self.device)
        answers = torch.stack(answers).to(self.device)

        # sub_sim = self.sub_sim[userid, userid].to(self.device)  # 全量sub相似度矩阵
        # gen_sim = self.gen_sim[userid, userid].to(self.device)  # 全量gen相似度矩阵

        gen_sim, sub_sim = self.__dataprocess__(item_gen, item_sub, subject_ids)

        sub_top_n = self.get_idx(sub_sim, top_n=2)
        gen_top_n = self.get_idx(gen_sim, top_n=2)

        # 封装批次数据
        batch_data = {
            'userid': userid,
            'item_sub': item_sub, # [batch_size, hidden_size]
            'item_gen': item_gen,
            'tag_ids': tag_ids,
            'subject_ids': subject_ids,
            'answers': answers,
            'sub_top_n': sub_top_n,
            'gen_top_n': gen_top_n
        }

        return batch_data


