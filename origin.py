import time
import random

import numpy as np
from scipy.spatial import KDTree

def load_data():
    """
    加载文件，从data/train.txt中加载原始训练数据，统计原始训练数据中的相关信息，进行打印输出
    形成字典sparse_matrix，sparse_matrix的键为用户编号，sparse_matrix的值为另一个字典rate_of_curruser
    字典rate_of_curruser的键为物品的编号，值为当前用户对该物品的评分。
    :return:sparse_matrix
    """
    start_time=time.time()
    train_path='data/train.txt'
    users=[]
    items=[]
    rates=[]
    with open(train_path,'r') as file:
        top_line=file.readline()
        sparse_matrix=dict()
        while top_line:
            user,nums=top_line.split('|')
            user=int(user)
            nums=int(nums)
            users.append(user)
            rate_of_curruser=dict()
            for i in range(nums):
                rate_line=file.readline()
                item,rate=rate_line.split()
                item=int(item)
                rate=int(rate)
                items.append(item)
                rates.append(rate)
                rate_of_curruser[item]=rate
            sparse_matrix[user]=rate_of_curruser
            top_line=file.readline()

        # 数据统计输出
        set_users = sorted(list(set(users)))
        set_items = sorted(list(set(items)))
        print('关于用户:')
        print('实际的用户数量:{}'.format(len(set_users)))
        print('用户的编号范围: {} 至 {}'.format(set_users[0],set_users[-1]))
        print('关于物品:')
        print('实际的物品数量:{}'.format(len(set_items)))
        print('物品的编号范围: {} 至 {}'.format(set_items[0], set_items[-1]))
        print('矩阵中的空闲率:{}'.format(1-len(items)/(len(set_items)*len(set_users))))
        end_time=time.time()
        print('加载原始数据，用时{}秒'.format(end_time-start_time))
        return set_users,set_items,sparse_matrix

def train_test_spilt(matrix,sample_rate=0.2):
    """

    :param matrix:原始的训练数据字典
    :param sample_rate:测试集划分比率，默认为20%
    :return:训练集字典和测试集字典
    """
    start_time=time.time()
    train_data=dict()
    test_data=dict()
    for user,rate_dict in matrix.items():
        sample_num=int(len(rate_dict)*sample_rate)

        test_keys=random.sample(list(rate_dict),sample_num)
        tmp_test_data={key:rate_dict[key] for key in test_keys}
        tmp_train_data={key:rate_dict[key] for key in rate_dict if key not in test_keys}
        train_data[user]=tmp_train_data
        test_data[user]=tmp_test_data
    end_time=time.time()
    print('训练集数据划分，用时{}秒'.format(end_time-start_time))
    return train_data,test_data

def load_attribute(bi):
    start=time.time()
    file_path='data/itemAttribute.txt'
    attr_dict=dict()
    with open(file_path,'r') as f:
        line=f.readline()
        debug=10000
        while line:
            item,att1,att2=line.split('|')
            if int(item)>debug:
                debug+=10000
                print(item)
            if 'None' in att1:
                att1=-1
            if 'None' in att2:
                att2=-1
            if int(item) in bi:
                attr_dict[int(item)]=[int(att1),int(att2)]
            line=f.readline()
    index2no=dict()
    no2index=dict()
    attr_array=np.zeros((len(attr_dict),2))
    index=0
    for item,attr in attr_dict.items():
        index2no[index]=item
        no2index[item]=index
        attr_array[index] = attr
        index+=1

    end=time.time()
    print('加载属性，用时{} s'.format(end-start))
    return index2no,no2index,attr_array

def k_neighbour(item_no,no2index,index2no,attr_array,kdtree,k):
    if item_no not in no2index:
        return []
    else:
        index=no2index[item_no]
        dist,ind=kdtree.query(attr_array[index],k)
        item_list=[index2no[i] for i in ind ]
        return item_list

def neighbour_item(bi):
    res=dict()
    index2no,no2index,attr_array=load_attribute(bi)
    kdtree=KDTree(attr_array)
    for item in bi.keys():
        # 对于当前的每一个物品
        if item not in no2index:
            res[item]=[]
        else:
            index = no2index[item]
            dist, ind = kdtree.query(attr_array[index], 5)
            item_list = [index2no[i] for i in ind]
            res[item]=item_list
    return res



if __name__ =='__main__':
    set_users,set_items,sparse_matrix=load_data()
    train,test=train_test_spilt(sparse_matrix)
    index2no,no2index,attr=load_attribute(set_items)
    pass
