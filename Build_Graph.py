import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import pandas as pd

def get_matrix_org(train_dict='./data/EducationComp.txt', item_atr_file = './data/gen_tag_item.npz', save_mat=False, self_loop=True):
    item_num = 53435
    mat = sp.dok_matrix((item_num, item_num), dtype=np.float32)
    gen_item = set(np.load(item_atr_file)["gen_item"])
    with open(train_dict) as f:
        for lines in tqdm(f.readlines()):
            item_list = lines.strip().split(' ')
            item_list = item_list[1:]
            for i in range(len(item_list) - 2): # 最后一个是答案
                cur_i = item_list[i]
                next_i = item_list[i + 1]
                mat[int(cur_i) - 1, int(next_i) - 1] += 1
                mat[int(next_i) - 1, int(cur_i) - 1] += 1
    if self_loop:
        print('add self-loop')
        for i in range(item_num):
            mat[i, i] += 1

    # if (mat.todense()== mat.todense().T).all():
    #     print('yes')
    if save_mat:
        sp.save_npz('mat_train.npz', mat.tocsr())
    # print(mat)

def get_matrix(train_dict='./data/EducationComp.txt', item_atr_file = './data/gen_tag_item.npz', save_mat=False, self_loop=True):
    item_num = 53435
    mat = sp.dok_matrix((item_num, item_num), dtype=np.float32)
    gen_item = set(np.load(item_atr_file)["gen_item"])
    with open(train_dict) as f:
        for line in tqdm(f.readlines()):
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items_sub = []
            items_gen = []
            for idx, item in enumerate(items):
                if idx == len(items) - 1:
                    continue
                if int(item) in gen_item:
                    items_gen.append(int(item))
                else:
                    items_sub.append(int(item))
            for i in range(len(items_sub) - 1):
                cur_i = items_sub[i]
                next_i = items_sub[i + 1]
                mat[int(cur_i) - 1, int(next_i) - 1] += 1
                mat[int(next_i) - 1, int(cur_i) - 1] += 1
            for i in range(len(items_gen) - 1):
                cur_i = items_gen[i]
                next_i = items_gen[i + 1]
                mat[int(cur_i) - 1, int(next_i) - 1] += 1
                mat[int(next_i) - 1, int(cur_i) - 1] += 1
    if self_loop:
        print('add self-loop')
        for i in range(item_num):
            mat[i, i] += 1

    # if (mat.todense()== mat.todense().T).all():
    #     print('yes')
    if save_mat:
        sp.save_npz('mat_train_int.npz', mat.tocsr())
    # print(mat)

# get_matrix(train_dict='./data/EducationComp.txt', save_mat=True)

def get_test_matrix(train_dict, test_dict, save_path, self_loop=True):
    print("Building Graph......")
    item_num = 53435
    mat = sp.dok_matrix((item_num, item_num), dtype=np.float32)

    # training 数据
    with open(train_dict) as f:
        for lines in tqdm(f.readlines()):
            item_list = lines.strip().split(' ')
            item_list = item_list[1:] # 注意全部使用
            for i in range(len(item_list) - 1):
                cur_i = item_list[i]
                next_i = item_list[i + 1]
                mat[int(cur_i) - 1, int(next_i) - 1] += 1
                mat[int(next_i) - 1, int(cur_i) - 1] += 1

    # test 数据
    with open(test_dict) as f:
        for lines in tqdm(f.readlines()):
            item_list = lines.strip().split(' ')
            item_list = item_list[1:] # 注意全部使用
            for i in range(len(item_list) - 1):
                cur_i = item_list[i]
                next_i = item_list[i + 1]
                mat[int(cur_i) - 1, int(next_i) - 1] += 1
                mat[int(next_i) - 1, int(cur_i) - 1] += 1

    if self_loop:
        print('add self-loop')
        for i in range(item_num):
            mat[i, i] += 1

    sp.save_npz(save_path, mat.tocsr())
    print("Graph Saved......")


if __name__ == "__main__":
    get_matrix(train_dict='./data/train_set.txt', save_mat=True)
