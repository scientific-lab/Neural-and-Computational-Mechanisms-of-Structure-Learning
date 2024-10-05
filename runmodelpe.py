import os
import numpy as np
import random
import sys
starting_ind=np.int32(sys.argv[1])

## ***Hippocampus model***
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)


    def recall(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern)
        for idx, pattern in enumerate(input_pattern):
            for _ in range(max_iterations):
                activation = np.dot(self.weights, pattern)+0.001*np.random.randn(800)
                pattern = np.sign(activation)
                input_pattern[idx] = pattern
        return input_pattern

## *** Predict Color with Hippocampus***
def exepsi(x,model1,model2,network):
    return model2.predict(network.recall(model1.predict(np.hstack((x,np.zeros((x.shape[0],360)))))))[:,1024:1384]
import numpy as np

## *** OFC model ***
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_scaled(x):
    return 2 * sigmoid(x) - 1

def sigmoid_derivative_scaled(x):
    return 2 * sigmoid(x) * (1 - sigmoid(x))

def forw(x,w_train,w_train1, b_train, b_train1):
    w_train+=np.random.uniform(-0.015*np.mean(np.abs(w_train)),0.015*np.mean(np.abs(w_train)),(w_train.shape[0],w_train.shape[1]))

    yy = sigmoid_scaled(x.dot(w_train)+b_train)+np.random.uniform(-0.375*np.mean(np.abs(w_train)),0.375*np.mean(np.abs(w_train)),(w_train.shape[1]))
    y_pred = sigmoid_scaled(yy.dot(w_train1)+b_train1)
    return y_pred


def updatew(x, y, w_train, w_train1, b_train, b_train1, lr, x_test, cv):
    yy = forw(x,w_train,w_train1, b_train, b_train1)
    pe=np.average(np.abs(y-yy),1)
    
    hidd=sigmoid_scaled(x.dot(w_train))
    grad1 = -2 * (pe.reshape(-1, 1)* hidd.reshape(-1,88)).T .dot( (y - yy) * sigmoid_derivative_scaled(yy))
    gradb1 = np.sum(-2 * pe.reshape(-1, 1)*((y - yy) * sigmoid_derivative_scaled(yy)),0)
    grad=-2 * (pe.reshape(-1, 1) * x.reshape(-1,1024)).T.dot(((y - yy) * sigmoid_derivative_scaled(yy)).dot(w_train1.T)* sigmoid_derivative_scaled(hidd))
    gradb=-2 * np.sum(pe.reshape(-1, 1) *(((y - yy) * sigmoid_derivative_scaled(yy)).dot(w_train1.T)* sigmoid_derivative_scaled(hidd)),0)

    w_post = w_train - lr * grad
    w_post1 = w_train1 - lr * grad1
    b_post = b_train - lr * gradb
    b_post1 = b_train1 - lr * gradb1

    y_post= forw(x_test,w_post,w_post1, b_post, b_post1)
    loss_post = np.sum((y_post - cv)**2)
    print(loss_post)
    
    w_train = w_post
    w_train1 = w_post1
    b_train = b_post
    b_train1 = b_post1
    return w_train,w_train1, b_train, b_train1
    
## ***Generate Color Ring***
c=np.vstack((np.ones((180,1)),-1*np.ones((180,1))))
cv=np.zeros((72,360))
for i in range (0,72):
    cv[i,:]=np.vstack((c[5*i:],c[:5*i])).reshape(360,)
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
name=os.listdir('D:\\实验项目\\hopfield\\72item')
name1=os.listdir('D:\\实验项目\\hopfield\\72items_pe_match')
l=[]
for n in name:
    if 'epsilon' in n and '.npy' in n:
        l.append(np.int32(n[7:-4]))

## *** 5 Individual Run ***
for iiii in range (starting_ind,starting_ind+25):
    if 'epsilon'+str(l[iiii])+'.npy' not in name1:
        try:
            ##***ensure to use the same stimuli with independent noise***
            iii=l[iiii]
            error=np.zeros((72,6))
            k=np.zeros((72,6))
            NN=1024
            category=np.repeat(np.arange(1,9),9)
            color=np.tile(np.arange(9)+1,8)
            c=np.vstack((np.ones((180,1)),-1*np.ones((180,1))))
            #c=np.vstack((c[50:],c[:50])).reshape(360,1)
            cv=np.zeros((72,360))
            cv_noise=np.zeros((72,360))
            n_p=np.random.normal(0,5,(72))
            for i in range (0,72):
                ii=np.int32(np.round(n_p[i]))
                print(ii)
                indnoise=5*i+ii
                print(indnoise)
                if indnoise<0:
                    indnoise=indnoise+360
                if indnoise>=360:
                    indnoise=indnoise-360
                cv[i,:]=np.vstack((c[5*i:],c[:5*i])).reshape(360,)
                cv_noise[i,:]=np.vstack((c[indnoise:],c[:indnoise])).reshape(360,)
            m_sort=np.load('m_sort_e_'+str(iii)+'.npy')
            ind=random.sample(np.arange(72*1024).tolist(),1024)
            m_noise=m_sort.copy().reshape(72*1024)
            m_noise[ind]=-m_noise[ind]
            m_noise=m_noise.reshape(72,1024)
            category_sort=category[color.argsort()]

            ## ***Train Hippocampus Structure and Link to Colors and Scenes*** 
            
            epsilon=np.random.choice([-1,1],(72,800))
            network = HopfieldNetwork(800)


            network.train(epsilon)
            
            mm=np.hstack((m_noise.copy(),np.arange(72).reshape(72,1)))
            z=np.hstack((mm[:,0:1024],cv_noise))

            model1 = MultiOutputRegressor(LogisticRegression(max_iter=1000))
            model1.fit(z, epsilon)

            model2 = MultiOutputRegressor(LogisticRegression(max_iter=1000))
            model2.fit(epsilon,z)

            ###***Predict Hippocampus Structure***
            predictions = model2.predict(network.recall(model1.predict(np.hstack((m_sort[:,0:1024],np.zeros((72,360)))))))
            angle=np.argmax(np.dot(predictions[:,1024:1384],cv.T),1)*5
            aa=angle-np.arange(72)*5
            aa[aa>180]=360-aa[aa>180]
            aa[aa<=-180]=360+aa[aa<=-180]
            error[:,1]=aa
            k[:,1]=angle

            ## ***Train OFC strucuture*** 

            w_train=np.random.randn(1024,88)
            w_train1=np.random.randn(88,360)
            b_train=np.random.randn(88)
            b_train1=np.random.randn(360)

            import numpy as np
            from sklearn.model_selection import KFold

            
            lr = 0.005 #0.05
            for epoch in range(20):

                X_train = mm[0:72, 0:1024]
                y_train = cv
                kfold = KFold(n_splits=6, shuffle=True, random_state=np.random.randint(1500))
                for train_index, test_index in kfold.split(X_train,y_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
                    y_val_hippo = exepsi(X_val_fold[:,0:1024], model1, model2,network)
                        # 使用训练集更新权重
                    w_train,w_train1, b_train, b_train1 = updatew(X_train_fold, y_train_fold, w_train,w_train1, b_train, b_train1, lr, X_val_fold, y_val_hippo)
  
            ###***Predict OFC Structure***
            predictions=forw(m_sort,w_train,w_train1, b_train, b_train1)
            yp_1=predictions
            angle=np.argmax(np.dot(predictions,cv.T),1)*5
            aa=angle-np.arange(72)*5
            aa[aa>180]=360-aa[aa>180]
            aa[aa<=-180]=360+aa[aa<=-180]
            error[:,4]=aa
            k[:,4]=angle
            np.save('w_train_e_'+str(iii)+'.npy',w_train)
            np.save('w_train1_e_'+str(iii)+'.npy',w_train1)
            np.save('b_train_e_'+str(iii)+'.npy',b_train)
            np.save('b_train1_e_'+str(iii)+'.npy',b_train1)
            #k[i.save(model.state_dict(), 'model_train_e_'+str(iii)+'.pth')
            np.save('m_sort_e_'+str(iii)+'.npy',m_sort)
            np.save('m_noise_e_'+str(iii)+'.npy',mm)
            
            ## ***Train Hippocampus Random and Link to Colors and Scenes*** 
            mm = np.load('m_noise_c_'+str(iii)+'.npy')
            z=np.hstack((mm[:,0:1024],cv_noise))
            model3 = MultiOutputRegressor(LogisticRegression(max_iter=1000))
            model3.fit(z, epsilon)

            model4 = MultiOutputRegressor(LogisticRegression(max_iter=1000))
            model4.fit(epsilon,z)
            
            ###***Predict Hippocampus Random ***
            predictions = model4.predict(network.recall(model3.predict(np.hstack((m_sort[np.int32(mm[:,1024])][:,0:1024],np.zeros((72,360)))))))
            angle=np.argmax(np.dot(predictions[:,1024:1384],cv.T),1)*5
            aa=angle-np.arange(72)*5
            aa[aa>180]=360-aa[aa>180]
            aa[aa<=-180]=360+aa[aa<=-180]
            error[:,0]=aa
            k[:,0]=angle

            ###***Train OFC Random ***
            w_train=np.random.randn(1024,88)
            w_train1=np.random.randn(88,360)
            b_train=np.random.randn(88)
            b_train1=np.random.randn(360)

            import numpy as np
            from sklearn.model_selection import KFold

            kfold = KFold(n_splits=6, shuffle=True, random_state=42)

            lr = 0.005
            for epoch in range(20):

                X_train = mm[0:360, 0:1024]
                y_train = cv

                # 进行交叉验证
                for train_index, test_index in kfold.split(X_train,y_train):
                    # 根据交叉验证的索引划分训练集和验证集
                    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
                    y_val=np.zeros((60,360))
            #         oo=0
            #         for x in X_val_fold:
            #             y_val[oo]=exepsi(x, w1, w2)
            #             oo+=1

                    #for x,y in zip(X_train_fold,y_train_fold):
                    y_val_hippo = exepsi(X_val_fold[:,0:1024], model1, model2,network)
                        # 使用训练集更新权重
                    w_train,w_train1, b_train, b_train1 = updatew(X_train_fold, y_train_fold, w_train,w_train1, b_train, b_train1, lr, X_val_fold, y_val_hippo)
            np.save('w_train_c_'+str(iii)+'.npy',w_train)
            np.save('w_train1_c_'+str(iii)+'.npy',w_train1)
            np.save('b_train_c_'+str(iii)+'.npy',b_train)
            np.save('b_train1_c_'+str(iii)+'.npy',b_train1)
            np.save('mm_c_'+str(iii)+'.npy',m_sort[np.int32(mm[:,1024])])
            np.save('m_noise_c_'+str(iii)+'.npy',mm)
            
            ###***Predict OFC Random ***
            predictions=forw(m_sort[np.int32(mm[:,1024])],w_train,w_train1, b_train, b_train1)#np.sign(np.dot(m_sort[np.int32(mm[:,1024])],w_train))
            yp_0=predictions
            #predictions = model4.predict(network.recall(model3.predict(np.hstack((m_sort[np.int32(mm[:,1024])][:,0:1024],np.zeros((72,360)))))))
            angle=np.argmax(np.dot(predictions,cv.T),1)*5
            aa=angle-np.arange(72)*5
            aa[aa>180]=360-aa[aa>180]
            aa[aa<=-180]=360+aa[aa<=-180]
            error[:,2]=aa
            k[:,2]=angle

            k[:,3]=k[:,0].copy()
            if np.where(np.abs(k[:,0]-k[:,2])>=20)[0].size>0:
                predictions = model4.predict(network.recall(model3.predict(np.hstack((m_sort[np.int32(mm[:,1024])][np.where(np.abs(k[:,0]-k[:,2])>=20)[0],0:1024],yp_0[np.where(np.abs(k[:,0]-k[:,2])>=20)[0],:]))))) 
                k[np.where(np.abs(k[:,0]-k[:,2])>=20)[0],3]=np.argmax(np.dot(predictions[:,1024:1384],cv.T),1)
            angle=k[:,3]*5
            aa=angle-np.arange(72)*5
            aa[aa>180]=360-aa[aa>180]
            aa[aa<=-180]=360+aa[aa<=-180]
            error[:,3]=aa

            k[:,5]=k[:,1].copy()
            if np.where(np.abs(k[:,1]-k[:,4])>=20)[0].size>0:
                predictions = model2.predict(network.recall(model1.predict(np.hstack((m_sort[np.where(np.abs(k[:,1]-k[:,4])>=20)[0],0:1024],yp_1[np.where(np.abs(k[:,1]-k[:,4])>=20)[0],:])))))
                k[np.where(np.abs(k[:,1]-k[:,4])>=20)[0],3]=np.argmax(np.dot(predictions[:,1024:1384],cv.T),1)
            angle=k[:,5]*5
            aa=angle-np.arange(72)*5
            aa[aa>180]=360-aa[aa>180]
            aa[aa<=-180]=360+aa[aa<=-180]
            error[:,5]=aa

            
            np.save('epsilon'+str(iii)+'.npy',epsilon)
            np.save('hweight'+str(iii)+'.npy',network.weights)
            np.save('error'+str(iii)+'.npy',error)
            np.save('k'+str(iii)+'.npy',k)
            import pickle
            with open('model1_'+str(iii)+'.pkl','wb') as file:
                pickle.dump(model1,file)
            with open('model2_'+str(iii)+'.pkl','wb') as file:
                pickle.dump(model2,file)
            with open('model3_'+str(iii)+'.pkl','wb') as file:
                pickle.dump(model3,file)
            with open('model4_'+str(iii)+'.pkl','wb') as file:
                pickle.dump(model4,file)
        except ValueError:
            pass