import time
import random
import torch

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


if __name__ =='__main__':
    set_users,set_items,sparse_matrix=load_data()
    train,test=train_test_spilt(sparse_matrix)
    count=[]
    for user_no,items in train.items():
        count.append(len(items))
    sorted_list=sorted(count,reverse=True)

    print(sorted_list)
