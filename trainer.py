import numpy as np
import tqdm
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, mrr_at_k, env

class Trainer(object):
    # def __init__(self,model, train_dataloader,eval_dataloader,
    #              test_dataloader,args) -> None:
    def __init__(self,model, train_dataloader,test_dataloader,args) -> None:
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device('cuda' if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        self.train_dataloader = train_dataloader
        # self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # ##### l2norm for weight not for bias#######
        # weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        # no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        # parameters = [{'params': weight_decay_list},
        #           {'params': no_decay_list, 'weight_decay': 0.}]
        ###########################################

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        # self.optim = Adam(parameters, lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self,epoch):
        self.iteration(epoch,self.train_dataloader, train=True)

    def vaild(self,epoch,full_sort = False):
        return self.iteration(epoch,self.eval_dataloader,full_sort,train=False)

    def test(self,epoch,full_sort = False):
        return self.iteration(epoch,self.test_dataloader,full_sort,train=False)

    def iteration(self,epoch, dataloader,full_sort=False,train = False):
        raise NotImplementedError

    def get_full_sort_score(self,epoch,answers, pred_list, valid_metric=None, all_predictions=None):
        valid_metric = np.array(valid_metric)
        valid = np.sum(valid_metric, axis=0) / self.args.test_num # 数据增强后需要修改
        metric = {
            "Epoch": epoch,
            "MRR": '{:.4f}'.format(valid[0]),
            "hr@5": '{:.4f}'.format(valid[1]),
            "hr@10": '{:.4f}'.format(valid[2]),
            "hr@20": '{:.4f}'.format(valid[3]),
            "NDCG@5": '{:.4f}'.format(valid[4]),
            "NDCG@10": '{:.4f}'.format(valid[5]),
            "NDCG@20": '{:.4f}'.format(valid[6]),
        }
        mrr_all = valid[0]
        recall,ndcg = [], []
        for k in [1,3,5]:
            recall.append(recall_at_k(answers,pred_list,k))
            ndcg.append(ndcg_k(answers,pred_list,k))
        mrr = mrr_at_k(answers,pred_list, 50)
        post_fix = {
                "Epoch":epoch,
                "MRR@50": '{:.4f}'.format(mrr),
                "Recall@1":'{:.4f}'.format(recall[0]),"NDCG@1": '{:.4f}'.format(ndcg[0]),
                "Recall@3":'{:.4f}'.format(recall[1]),"NDCG@3": '{:.4f}'.format(ndcg[1]),
                "Recall@5": '{:.4f}'.format(recall[2]), "NDCG@5": '{:.4f}'.format(ndcg[2]),
                # "Recall@10":'{:.4f}'.format(recall[2]),"NDCG@10": '{:.4f}'.format(ndcg[2]),
                # "Recall@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
                # "Recall@100": '{:.4f}'.format(recall[4]), "NDCG@100": '{:.4f}'.format(ndcg[4]),
        }
        return [mrr, recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix), str(metric), all_predictions, mrr_all, pred_list
        #return post_fix

    def save(self,file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def binary_cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        embedding = self.model.get_gcn_embedding()
        # embedding = self.model.item_embeddings.weight
        pos_emb = embedding[pos_ids]
        neg_emb = embedding[neg_ids]
        # pos_emb = self.model.item_embeddings(pos_ids)
        # neg_emb = self.model.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def cross_entropy(self, seq_out, pos_ids):
        embedding = self.model.get_gcn_embedding()  # GCN之后所有item的embeddings

        logits = torch.matmul(seq_out, embedding.transpose(0,1))

        index = pos_ids.reshape(-1)

        ce_loss = nn.CrossEntropyLoss()
        loss = ce_loss(logits, index)
        return loss


    def predict_full(self, seq_out): # 得到每个用户对所有项目的评分预测矩阵
        # [item_num hidden_size]
        test_item_emb = self.model.get_gcn_embedding()  # 存疑？ 是否需要结合tag embedding一起计算

        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1)) # 计算用户序列表示与项目嵌入矩阵的点积，得到评分预测矩阵
        return rating_pred

class SASRecTrainer(Trainer):
    # def __init__(self, model,
    #              train_dataloader,
    #              eval_dataloader, test_dataloader, args) -> None:
    #     super(SASRecTrainer,self).__init__(model, train_dataloader,
    #                                        eval_dataloader,
    #                                        test_dataloader, args)
    def __init__(self, model,
                 train_dataloader, test_dataloader, args) -> None:
        super(SASRecTrainer,self).__init__(model, train_dataloader, test_dataloader, args)

    def iteration(self, epoch, dataloader, full_sort=False, train=False):
        str_code = "train" if train else "test"

        # train和test过程的进度条
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i,batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)  # 数据放在GPU
                _, input_ids, tag_ids, school_ids, area_ids, subject_ids, target_pos, target_neg, answers = batch #

                sequence_out = self.model(input_ids, tag_ids, school_ids, area_ids, subject_ids)

                loss = self.cross_entropy(sequence_out,answers)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval() # 将模型设置为评估模式（torch的内置函数），这意味着模型中的 dropout 和 batch normalization 层会使用固定的参数，而不是在训练过程中更新
            pred_list = None
            # 预先分配一个numpy数组来存储所有预测结果
            all_predictions = np.zeros((20000, 53435), dtype=np.float32)  # 测试数据2w条
            if full_sort: #test和valid时
                answer_list = None
                valid_metric = []
                for i, batch in rec_data_iter:
                    # 计算当前批次的起始和结束索引
                    start_idx = i * self.args.batch_size
                    end_idx = min((i + 1) * self.args.batch_size, 20000)
                    batch = tuple(t.to(self.device) for t in batch) #转到gpu上
                    user_ids, input_ids, tag_ids, school_ids, area_ids, subject_ids,  target_pos, target_neg, answers, test_negative = batch

                    recommend_output = self.model(input_ids, tag_ids, school_ids, area_ids, subject_ids) # 使用模型对输入ID进行前向传播，得到推荐输出

                    rating_pred = self.predict_full(recommend_output) # 每个用户对所有项目的评分预测矩阵
                    all_predictions[start_idx:end_idx] = rating_pred[:, 1:].cpu().detach().numpy()
                    ###################todo: 拿rating_pred和负采样的点做评估(hr,ndcg,mrr),注意rating_pred矩阵需要调整索引和裁剪！！！！
                    # 这块在cpu上做
                    rating = rating_pred[:, 1:].cpu().clone() # 除去第一个填充元素
                    test_h5, test_h10, test_h20, test_n5, test_n10, test_n20, test_mrr = env(rating, test_negative.cpu()) #test_negative存储的item_id是从1开始的
                    valid_metric.append([test_mrr, test_h5, test_h10, test_h20, test_n5, test_n10, test_n20])
                    ############################################################################################################
                    rating_pred[:,0] = -np.inf # 避免推荐第一个填充元素

                    _,batch_pred_list = torch.topk(rating_pred,50,dim=-1) # 从评分预测中选择前100个最高的评分，返回其索引。
                    batch_pred_list = batch_pred_list.cpu().data.numpy()

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()

                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

                return self.get_full_sort_score(epoch,answer_list,pred_list,valid_metric,all_predictions=all_predictions)


class IntRecTrainer(Trainer):
    # def __init__(self, model,
    #              train_dataloader,
    #              eval_dataloader, test_dataloader, args) -> None:
    #     super(SASRecTrainer,self).__init__(model, train_dataloader,
    #                                        eval_dataloader,
    #                                        test_dataloader, args)
    def __init__(self, model,
                 train_dataloader, test_dataloader, args) -> None:
        super(IntRecTrainer, self).__init__(model, train_dataloader, test_dataloader, args)
        self.tau = self.args.tau  # 温度参数

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if len(z1.size()) == 1:
            z1 = z1.unsqueeze(0)  # Add batch dimension
        if len(z2.size()) == 1:
            z2 = z2.unsqueeze(0)  # Add batch dimension
        if z1.size()[0] == z2.size()[0]:
            if z1.size() == z2.size():
                return F.cosine_similarity(z1, z2)
            return F.cosine_similarity(z1.unsqueeze(1), z2, dim=2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def info_nce_loss(self, anchor, neg, top_n_idx, all_emb):
        """
        anchor: [hidden_size] sub or gen的用户表示
        neg: [hidden_size] gen or sub的用户表示
        top_n_idx: [n] 取前n个的索引
        all_emb: [batch_size, hidden_size]
        """
        """
        anchor: [batch_size, hidden_size] sub or gen的用户表示
        neg: [batch_size, hidden_size] gen or sub的用户表示
        top_n_idx: [batch_size] 取前1个的索引
        all_emb: [batch_size, hidden_size]
        """
        def f(x): return torch.exp(x / self.tau)
        batch_size = all_emb.size(0)
        # 选出该用户对应的一批正样本
        pos = all_emb[top_n_idx]  # 只选择不是pad_idx的索引

        pos_sim = f(self.sim(anchor, pos)) # [batch_size] # 计算正样本的相似度
        pos_pairs = pos_sim

        # sub和gen互为负样本
        between_sim = f(self.sim(anchor, neg)) # [batch_size]

        batch_idx = torch.arange(batch_size, device=top_n_idx.device)
        mask = top_n_idx == batch_idx # 正样本是自身的用户需要特殊处理一下

        random_values = torch.arange(batch_size, device=top_n_idx.device) + 1
        random_values[-1] = 0
        top_n_idx = top_n_idx.clone()
        top_n_idx[mask] = random_values[mask]

        top_n_idx = torch.cat([top_n_idx.unsqueeze(1), batch_idx.unsqueeze(1)], dim=1) # [batch_size, 2]  # 将top_n_idx和当前batch的索引拼接在一起，自身不作为负样本
        # 一个batch中除去正样本的其余用户作为负样本
        all_indices = batch_idx.unsqueeze(0).expand(batch_size, batch_size).clone()  # [batch_size, batch_size]
        # 计算剩余的索引
        remaining_idx = []
        for i in range(batch_size):
            mask = ~torch.isin(all_indices[i], top_n_idx[i])  # 创建一个布尔掩码
            idx = all_indices[i][mask]  # 使用掩码选择不在 top_n_idx 中的索引
            remaining_idx.append(idx)
        remaining_idx = torch.stack(remaining_idx, dim=0)
        remaining_neg = all_emb[remaining_idx] # [batch_size, batch_size-2, hidden_size]

        remaining_sim = f(self.sim(anchor, remaining_neg)) # [batch_size, batch_size-2]
        remaining_sim = torch.sum(remaining_sim, 1) # [batch_size]

        neg_pairs = between_sim + remaining_sim

        loss = torch.sum(-torch.log(pos_pairs / neg_pairs))
        return loss

    def iteration(self, epoch, dataloader, full_sort=False, train=False):
        str_code = "train" if train else "test"

        # train和test过程的进度条
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_cl_loss = 0.0

            alpha = self.args.alpha + (1 - self.args.alpha) * (1 - np.exp(-self.args.k * epoch / self.args.epochs)) # 指数衰减
            # alpha = min(1, self.args.alpha * (1 + self.args.decay_rate * epoch / self.args.epochs))  # 逐渐增大alpha，实则衰减对比损失权重

            for i, batch in rec_data_iter:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}  # 数据放到 GPU
                    # user_id = batch['userid'].squeeze(0)
                    item_sub = batch['item_sub'].squeeze(0)
                    item_gen = batch['item_gen'].squeeze(0)
                    tag_ids = batch['tag_ids'].squeeze(0)
                    subject_ids = batch['subject_ids'].squeeze(0)
                    answers = batch['answers'].squeeze(0)
                    sub_top_n_idx = batch['sub_top_n'].squeeze(0)
                    gen_top_n_idx = batch['gen_top_n'].squeeze(0)
                elif isinstance(batch, list):
                    batch = tuple(t.to(self.device) for t in batch)  # 数据放在GPU
                    _, item_sub, item_gen, tag_ids, subject_ids, answers = batch  #
                else:
                    raise ValueError("Unsupported batch format")

                sequence_out = self.model(item_sub, item_gen, tag_ids, subject_ids) # user_emb

                # main loss
                main_loss = self.cross_entropy(sequence_out, answers)

                if self.args.contrast:
                    sub = self.model.sub_emb
                    gen = self.model.gen_emb
                    cl_gen = self.info_nce_loss(gen, sub, gen_top_n_idx, gen)
                    cl_sub = self.info_nce_loss(sub, gen, sub_top_n_idx, sub)
                    cl_loss = (cl_gen + cl_sub) / self.args.batch_size / 2.0

                    # 计算总损失
                    loss = alpha * main_loss + (1 - alpha) * cl_loss
                    print("main_loss:", main_loss.item() * alpha)
                    print("cl_loss:", cl_loss.item()*(1 - alpha))
                else:
                    # 如果没有对比损失，则仅使用主损失
                    loss = main_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_cl_loss += cl_loss.item() if self.args.contrast else 0.0

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_cl_loss": '{:.4f}'.format(rec_avg_cl_loss / len(rec_data_iter)) if self.args.contrast else 'N/A',
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()  # 将模型设置为评估模式（torch的内置函数），这意味着模型中的 dropout 和 batch normalization 层会使用固定的参数，而不是在训练过程中更新
            pred_list = None
            # 预先分配一个numpy数组来存储所有预测结果
            all_predictions = np.zeros((20000, 53435), dtype=np.float32) # 测试数据2w条
            if full_sort:  # test和valid时
                answer_list = None
                valid_metric = []
                for i, batch in rec_data_iter:
                    # 计算当前批次的起始和结束索引
                    start_idx = i * self.args.batch_size
                    end_idx = min((i + 1) * self.args.batch_size, 20000)
                    batch = tuple(t.to(self.device) for t in batch)  # 转到gpu上
                    user_ids, item_sub, item_gen, tag_ids, subject_ids, answers, test_negative = batch

                    recommend_output = self.model(item_sub, item_gen, tag_ids, subject_ids)  # 使用模型对输入ID进行前向传播，得到推荐输出
                    rating_pred = self.predict_full(recommend_output)  # 每个用户对所有项目的评分预测矩阵
                    all_predictions[start_idx:end_idx] = rating_pred[:, 1:].cpu().detach().numpy()
                    ###################todo: 拿rating_pred和负采样的点做评估(hr,ndcg,mrr),注意rating_pred矩阵需要调整索引和裁剪！！！！
                    # 这块在cpu上做
                    rating = rating_pred[:, 1:].cpu().clone() # 除去第一个填充元素
                    test_h5, test_h10, test_h20, test_n5, test_n10, test_n20, test_mrr = env(rating, test_negative.cpu()) #test_negative存储的item_id是从1开始的
                    valid_metric.append([test_mrr, test_h5, test_h10, test_h20, test_n5, test_n10, test_n20])
                    ############################################################################################################
                    rating_pred[:, 0] = -np.inf  # 避免推荐第一个填充元素

                    _, batch_pred_list = torch.topk(rating_pred, 50, dim=-1)  # 从评分预测中选择前100个最高的评分，返回其索引。 返回推荐的前50个物品id
                    batch_pred_list = batch_pred_list.cpu().data.numpy()

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()

                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

                return self.get_full_sort_score(epoch, answer_list, pred_list, valid_metric, all_predictions) # answer和pred都是从1开始
