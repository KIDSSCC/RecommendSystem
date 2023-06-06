from origin import load_data,train_test_spilt
import numpy as np
import time
import numba

class fit_model:

    def __init__(self,train,set_users,set_items,k):
        """

        :param train: 训练集
        :param set_users: 用户集合
        :param set_items: 物品集合
        :param k: 超参数
        """
        self.train=train
        self.k=k
        self.mean = get_mean_of_train(train)
        self.bias_u = dict()
        self.bias_i = dict()
        self.pu = dict()
        self.qi = dict()
        self.item_hidden=np.zeros((set_items[-1]+1,self.k))
        for user_no in set_users:
            self.bias_u[user_no] = 0
            self.pu[user_no] = np.random.rand(k)
        for item_no in set_items:
            self.bias_i[item_no] = 0
            self.qi[item_no] = np.random.rand(k)

        self.curr_mean_item=np.zeros((1,k))

    def predict_score(self,user_no,item_no):
        """
        预测当前用户对某一物品的评分
        :param user_no: 用户编号
        :param item_no: 物品编号
        :return: 预测评分
        """
        basic=self.mean+self.bias_u[user_no]+self.bias_i[item_no]
        # 计算隐式反馈向量平均值res
        items=self.train[user_no]
        num_of_items=len(items)
        avg=np.zeros((1,self.k))
        if num_of_items==0:
            res=avg+0.1
        else:
            # item_list=[item for item,_ in items.items()]
            item_list=list(items.keys())
            avg=np.mean(self.item_hidden[item_list],axis=0)
            res=avg/np.sqrt(num_of_items)
        self.curr_mean_item=res
        basic+=np.sum(self.qi[item_no]*(self.pu[user_no]+res))
        return basic

    def gradient_desc(self,user_no,item_no,error,lr,lamb):
        self.bias_u[user_no] += lr * (error - lamb * self.bias_u[user_no])
        self.bias_i[item_no] += lr * (error - lamb * self.bias_i[item_no])
        old_pu = self.pu[user_no]
        old_qi = self.qi[item_no]
        self.pu[user_no] += lr * (error * old_qi - lamb * old_pu)
        self.qi[item_no] += lr * (error * (old_pu+self.curr_mean_item[0]) - lamb * old_qi)
        # 更新隐式向量列表
        items=self.train[user_no]
        sqrt_len=np.sqrt(len(items))
        # item_list=[item for item,_ in items.items()]
        item_list=list(items.keys())
        tmp_array=np.array([self.qi[no] for no in item_list])
        self.item_hidden[item_list]+=lr*(error*tmp_array/sqrt_len-lamb*self.item_hidden[item_list])


def get_mean_of_train(train):
    """
    获取当前集合的全局平均值
    :param train:训练集
    :return:在训练集中，用户对所用物品评分的平均值
    """
    sum_rate=0
    count=0
    for user,items in train.items():
        for item_no in items.keys():
            sum_rate+=items[item_no]
            count+=1
    return sum_rate/count

@numba.jit(nopython=False)
def svdpp_train(train,test,set_users,set_items,n_epoch,lr,k,lamb):
    """

    :param train: 训练集
    :param test: 测试集
    :param set_users: 用户集合，保存着所有的用户
    :param set_items: 物品集合，保存着所有的物品
    :param n_epoch: 训练批次
    :param lr: 学习率
    :param k: 超参数
    :param lamb: 梯度下降参数
    :return:训练得到的模型
    """
    start_time = time.time()
    model=fit_model(train,set_users,set_items,k)
    for epoch in range(n_epoch):
        for user_no,items in train.items():
            for item_no,real_rate in items.items():
                predict_rate=model.predict_score(user_no,item_no)
                error=real_rate-predict_rate
                # 梯度下降
                model.gradient_desc(user_no,item_no,error,lr,lamb)
            if user_no %5==0:
                print('user progress:[{}/{}]'.format(user_no, len(train)))
            if user_no==300:
                end_time = time.time()
                print('前300个用户，用时{}秒'.format(end_time - start_time))
                return
        # 完成一轮迭代
        rmse_in_train = svdpp_eval(train, model)
        rmse_in_test = svdpp_eval(test, model)
        print('epoch:[{}/{}],RMSE in train is :{} , and RMSE in test is {}'.format(epoch, n_epoch, rmse_in_train,
                                                                                   rmse_in_test))


def svdpp_eval(test,fit_model):
    sum_error=0
    count=0
    for user_no,items in test.items():
        for item_no,real_rate in items.items():
            predict_rate=fit_model.predict_score(user_no,item_no)
            sum_error+=(real_rate-predict_rate)**2
            count+=1

    return np.sqrt(sum_error/count)

if __name__ =='__main__':
    set_users,set_items,sparse_matrix=load_data()
    train,test=train_test_spilt(sparse_matrix)
    svdpp_train(train,test,set_users,set_items,20,5e-4,3,0.02)