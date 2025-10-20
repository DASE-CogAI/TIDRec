import os
import tqdm
import torch
import argparse
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import Userdataset, BatchIndexCollate
from TIDR_model import Intrec
from utils import set_seed, EarlyStopping, get_sp_adj_matrix, get_user_seqs_split_atr, get_user_seqs_split_atr_aug
from trainer import IntRecTrainer
import scipy.sparse as sp

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='./model/', type=str)
    parser.add_argument('--data_name', default='Intrec', type=str)
    parser.add_argument('--do_eval', action='store_true')

    # model args
    parser.add_argument("--model_name", default='intrec', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=3, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.4, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.4, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default = 0.02)
    parser.add_argument('--max_seq_length_item', default=25, type=int)
    parser.add_argument('--max_seq_length_tag', default=50, type=int)
    parser.add_argument('--sparse', default=-0.1, type=float)
    parser.add_argument("--gnn_layer", type=int, default=2, help="gnn layer")

    # embedding
    parser.add_argument("--item_dict", type=str, default='./data/id_to_name.json',help="path of itemid to item name dict")
    parser.add_argument("--bert", type=str, default='./bert-base-uncased', help="bert model name or path")
    # ablation
    parser.add_argument("--ablation_tag", type=bool, default=False, help="ablation tag")
    parser.add_argument("--ablation_sub", type=bool, default=False, help="ablation attentive sequence fusion")
    # attentive fusion
    parser.add_argument("--attention_size", type=int, default=128, help="attention size")

    # contrastive learning
    parser.add_argument("--contrast", type=bool, default=True, help="whether to use contrastive learning")
    parser.add_argument("--top_n", type=int, default=1, help="top n for evaluation")
    parser.add_argument("--pad_idx", type=int, default=-1, help="padding index for sequence in collate_fn")
    parser.add_argument("--tau", type=float, default=0.2, help="temperature for contrastive learning")
    parser.add_argument("--alpha", type=float, default=0.6, help="weight for contrastive loss")
    parser.add_argument("--k", type=float, default=4, help="decay rate for contrastive learning")
    # train args
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # user data
    args = parser.parse_args()
    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    set_seed(args.seed)

    # 以下数值均为实际数量+1，为构建embedding方便
    num_item_dic = {'EducationComp': 53436}
    num_tag_dic = {'EducationComp': 3008}
    num_subject_dic = {'EducationComp': 51}

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # user_seqs_path = args.data_dir + 'EducationComp.txt'
    user_profile_path = args.data_dir + "school_area_subject.csv"
    # tag_seqs_path = args.data_dir + 'tag.txt'

    ############################################################################################################################################################################
    train_user_seqs_path = args.data_dir + 'train_set.txt'
    test_user_seqs_path = args.data_dir + 'test_set.txt'
    train_tag_seqs_path = args.data_dir + 'train_tag.txt'
    test_tag_seqs_path = args.data_dir + 'test_tag.txt'
    item_gen_path = args.data_dir + 'gen_tag_item.npz'
    # train_negative_path = args.data_dir + 'train_negative.pt'
    test_negative_path = args.data_dir + 'test_negative.pt'
    # train_negative = torch.load(train_negative_path, map_location='cpu')
    test_negative = torch.load(test_negative_path, map_location='cpu')

    ############################################################################################################################################################################
    args.item_size = num_item_dic['EducationComp']
    args.tag_size = num_tag_dic['EducationComp']
    args.train_num = 80000 # 数据增强后需要改动
    args.test_num = 20000
    # args.school_size =  num_school_dic['EducationComp'] # 暂时不用
    # args.area_size =  num_area_dic['EducationComp'] # 暂时不用
    args.subject_size =  num_subject_dic['EducationComp']
    ############################################################################################################################################################################
    train_seq_sub, train_seq_gen, train_tag_seq, train_sub, train_target = get_user_seqs_split_atr_aug(train_user_seqs_path, train_tag_seqs_path, user_profile_path, item_gen_path)
    if args.contrast:
        # 训练集 对比只应用在训练集
        train_dataset = Userdataset(args, user_seq_sub = train_seq_sub, user_seq_gen = train_seq_gen, tag_seq = train_tag_seq, user_sub = train_sub, target = train_target, data_type='train')
        train_sampler = RandomSampler(train_dataset) # 训练模型时随机采样
        collate_fn = BatchIndexCollate(top_n=args.top_n, pad_idx=args.pad_idx)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        train_dataset = Userdataset(args, user_seq_sub=train_seq_sub, user_seq_gen=train_seq_gen, tag_seq=train_tag_seq,
                                    user_sub=train_sub, target=train_target, data_type='train')
        train_sampler = RandomSampler(train_dataset)  # 训练模型时随机采样
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    # 验证集
    test_seq_sub, test_seq_gen, test_tag_seq, test_sub, test_target = get_user_seqs_split_atr(test_user_seqs_path, test_tag_seqs_path, user_profile_path, item_gen_path)
    test_dataset = Userdataset(args, user_seq_sub = test_seq_sub, user_seq_gen = test_seq_gen, tag_seq = test_tag_seq, test_neg_items = test_negative, user_sub = test_sub, target = test_target, data_type='test')
    test_sampler = SequentialSampler(test_dataset) # 评估模型时顺序采样
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # mat_path = args.data_dir + 'mat_EducationComp.npz'
    mat_path = args.data_dir + 'mat_train_int.npz' # 使用新的图
    trans_mat = sp.load_npz(mat_path)
    trans_mat_norm = get_sp_adj_matrix(trans_mat)
    model = Intrec(args=args, mat=trans_mat_norm)

    # 训练模型保存： 创建以当前时间命名的文件夹
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)

    # train_dataloader=None
    trainer = IntRecTrainer(model, train_dataloader, test_dataloader, args)
    early_stopping = EarlyStopping(output_dir=output_dir, patience=10, verbose=True)

    # 训练结果日志: 和模型保存至相同的文件夹
    logs_path = os.path.join(output_dir, 'logs_'+args.data_name.lower()+'.txt')
    f = open(logs_path,'w')

    for epoch in range(args.epochs):
        # score, msg, metric = trainer.test(epoch, full_sort=True) # 测试验证过程用
        trainer.train(epoch)

        score, msg, metric, _, mrr_neg, _ =trainer.test(epoch,full_sort=True)
        print(msg)
        print(metric)
        f.write(msg + '\n')
        f.write(metric + '\n')
        f.flush() # 直接写入文件，不用等训练结束

        # 基于MRR, 保存最佳模型
        early_stopping(score[0], mrr_neg, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    f.close()

    # 训练结束，加载最佳模型,确认模型结果
    # output_dir = './model/20250616_0026'
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model_mrr0.7.pt')))
    final_score, final_msg, final_metric,all_prediction, _, pred_50 = trainer.test(epoch=0, full_sort=True)
    np.savetxt(os.path.join(output_dir, 'TIDRec_prediction.csv'), pred_50, delimiter=',', fmt='%d', comments='')
    np.save(os.path.join(output_dir, 'all_prediction.npy'), all_prediction)
    f = open(logs_path, 'a')
    f.write('Final Test Result: ' + final_msg + '\n')
    f.write('Final Test Result: ' + final_metric + '\n')
    f.close()

if __name__ == '__main__':
    main()
