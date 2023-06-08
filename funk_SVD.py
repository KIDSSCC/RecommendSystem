from origin import load_data,train_test_spilt
import numpy as np
import pickle
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)
class fit_model:

    def __init__(self,mean,bias_u,bias_i,pu,qi):
        self.mean=mean
        self.bias_u=bias_u
        self.bias_i=bias_i
        self.pu=pu
        self.qi=qi

    def predict_score(self,user_no,item_no):
        basic=self.pu[user_no]@self.qi[item_no]
        return basic+self.mean+self.bias_u[user_no]+self.bias_i[item_no]

    def gradient_desc(self,user_no,item_no,error,lr,lamb):
        self.bias_u[user_no] += lr * (error - lamb * self.bias_u[user_no])
        self.bias_i[item_no] += lr * (error - lamb * self.bias_i[item_no])
        old_pu=self.pu[user_no]
        old_qi=self.qi[item_no]
        self.pu[user_no] += lr * (error * old_qi - lamb * old_pu)
        self.qi[item_no] += lr * (error * old_pu - lamb * old_qi)




def get_mean_of_train(train):
    sum_rate=0
    count=0
    for user,items in train.items():
        for item_no in items.keys():
            sum_rate+=items[item_no]
            count+=1
    return sum_rate/count

def funk_svd_train(train,test,set_users,set_items,n_epoch,lr,k,lamb):
    mean = get_mean_of_train(train)
    # init=np.sqrt((mean-1)/k)
    bias_u = dict()
    bias_i = dict()
    pu = dict()
    qi = dict()
    for user_no in set_users:
        bias_u[user_no]=0
        pu[user_no]=np.random.normal(0, .1, k)
    for item_no in set_items:
        bias_i[item_no]=0
        qi[item_no]=np.random.rand(k)

    model=fit_model(mean,bias_u,bias_i,pu,qi)
    for epoch in range(n_epoch):
        for user_no,items in train.items():
            for item_no,real_rate in items.items():
                predict_rate=model.predict_score(user_no,item_no)
                error=real_rate-predict_rate
                # 梯度下降
                model.gradient_desc(user_no,item_no,error,lr,lamb)
            if user_no % 2000 == 0:
                print('user progress:[{}/{}]'.format(user_no, len(train)))
        # 完成一轮迭代
        rmse_in_train=funk_svd_eval(train,model)
        rmse_in_test=funk_svd_eval(test,model)
        print('epoch:[{}/{}],RMSE in train is :{} , and RMSE in test is {}'.format(epoch,n_epoch,rmse_in_train,rmse_in_test))
        pickle_path='models/fit_model'+str(epoch)+'.pkl'
        with open(pickle_path,'wb') as f:
            pickle.dump(fit_model,f)

def funk_svd_eval(test,fit_model):
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
    funk_svd_train(train,test,set_users,set_items,50,5e-4,100,1.0)

