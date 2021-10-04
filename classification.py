import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('classification_train_data.csv')
train=train.to_numpy()
test=pd.read_csv('classification_test_data.csv')
test=test.to_numpy()

X_train,Y_train=np.column_stack((np.ones((train[:,0:-1].shape[0],1)),train[:,0:-1])),np.array([1 if x==1 else -1 for x in train[:,[-1]]])
X_test,Y_test=np.column_stack((np.ones((test[:,0:-1].shape[0],1)),test[:,0:-1])),np.array([1 if x==1 else -1 for x in test[:,[-1]]])

class classification:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def perceptron(self):
        w=np.zeros((3,1))
        w[2]=1
        for i in range(len(self.Y)):
            z=np.dot(self.X[i],w)[0]/np.linalg.norm(w)
            if z*self.Y[i]<1:
                w=w+self.Y[i]*(np.array([self.X[i]]).T)
        return w/np.linalg.norm(w)
    def cov_and_mu_estimate(self):
        mu1= np.mean(np.array([self.X[i,1:3] for i in range(len(self.X)) if self.Y[i]==1]),0).reshape((2,1))
        mu0= np.mean(np.array([self.X[i,1:3] for i in range(len(self.X)) if self.Y[i]==-1]),0).reshape((2,1))
        x=0
        for i in range(200):
            if self.Y[i]==1:
                z=self.X[i,1:3].reshape((2,1))-mu1
                x+=np.dot(z.reshape(1,2),z)[0][0]
            else:
                z=self.X[i,1:3].reshape((2,1))-mu0
                x+=np.dot(z.reshape(1,2),z)[0][0]
        C1=np.cov(np.array([X_train[i,1:3] for i in range(len(X_train)) if Y_train[i]==1]).T)
        C2=np.cov(np.array([X_train[i,1:3] for i in range(len(X_train)) if Y_train[i]==-1]).T)
        return mu1,mu0,x/(200-2)*np.identity(2),((110-1)*C1+(90-1)*C2)/(200-2),C1,C2
    def discrminant_same_variance(self,mu1,mu0,sigma1,x):
        sigma_1=np.linalg.inv(sigma1)
        return np.log(110/90)-np.dot((mu1+mu0).reshape((1,2)),np.dot(sigma_1,mu1-mu0))+np.dot(x,np.dot(sigma_1,mu1-mu0))[0]
    def discrminant_diff_variance(self,mu1,mu0,C1,C2,x):
        C_1=np.linalg.inv(C1)
        C_2=np.linalg.inv(C2)  
        z1=np.dot(x-mu1.reshape((1,2)),np.dot(C_1,x.reshape((2,1))-mu1))
        z0=np.dot(x-mu0.reshape((1,2)),np.dot(C_2,x.reshape((2,1))-mu0))
        return (np.log(110/90)-z1-abs(np.linalg.det(C1))+z0+abs(np.linalg.det(C2)))[0][0]
    def bound_disc_same_variance(self,mu1,mu0,sigma1,x):
        sigma_1=np.linalg.inv(sigma1)
        z=np.dot(sigma_1,mu1-mu0)
        return ((-np.log(110/90)+np.dot((mu1+mu0).reshape((1,2)),np.dot(sigma_1,mu1-mu0))-z[1]*x)/z[0])[0][0]
    def bound_disc_diff_variance(self,mu1,mu0,C1,C2,x):
        z=[abs(self.discrminant_diff_variance(mu1,mu0,C1,C2,np.array([x,y]))) for y in np.arange(-2,12,0.02)]
        return -2+0.02*z.index(min(z))
    def confusion_perceptron(self,w,X,Y):
        conf=np.zeros((2,2))
        Y_pred = [1 if (np.dot(x,w)[0]>0) else -1 for x in X]
        for i in range(len(Y)):
            if Y[i]==Y_pred[i]:
                if Y[i]==1:
                    conf[0][0]+=1
                else:
                    conf[1][1]+=1
            elif Y[i]==1:
                conf[0][1]+=1
            else:
                conf[1][0]+=1
        return conf
    def confusion_disc_same_var(self,mu1,mu0,sigma,X,Y):
        conf=np.zeros((2,2))
        Y_pred = [1 if self.discrminant_same_variance(mu1,mu0,sigma,x[1:3])>0 else -1 for x in X]
        for i in range(len(Y)):
            if Y[i]==Y_pred[i]:
                if Y[i]==1:
                    conf[0][0]+=1
                else:
                    conf[1][1]+=1
            elif Y[i]==1:
                conf[0][1]+=1
            else:
                conf[1][0]+=1
        return conf
    def confusion_disc_diff_var(self,mu1,mu0,C1,C2,X,Y):
        conf=np.zeros((2,2))
        Y_pred = [1 if self.discrminant_diff_variance(mu1,mu0,C1,C2,x[1:3])>0 else -1 for x in X]
        for i in range(len(Y)):
            if Y[i]==Y_pred[i]:
                if Y[i]==1:
                    conf[0][0]+=1
                else:
                    conf[1][1]+=1
            elif Y[i]==1:
                conf[0][1]+=1
            else:
                conf[1][0]+=1
        return conf



###################################################################################################


            
    
classifier=classification(X_train,Y_train)

#perceptron classifier
w=classifier.perceptron()
print("coefficients of classifier = \n", w)

boundary0=((-w[0])/w[2]-w[1]/w[2]*X_train[:,[1]])
#result on train set
colors=['red' if x==1 else 'blue' for x in Y_train]
plt.scatter(X_train[:,[1]],X_train[:,[2]], color=colors)
plt.plot(X_train[:,[1]],boundary0,'-k')
plt.title("perceptron classifier on training data")
plt.legend(["boundary","training data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("perceptron_train.png")
plt.show()

print("confusion matrix of perceptron on train: \n",classifier.confusion_perceptron(w,X_train,Y_train))


#predictions on test set
colors=['red' if x==1 else 'blue' for x in Y_test]
plt.scatter(X_test[:,[1]],X_test[:,[2]], color=colors)
plt.plot(X_train[:,[1]],boundary0,'-k')
plt.title("perceptron classifier on test data")
plt.legend(["boundary","test data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("perceptron_test.png")
plt.show()

print("confusion matrix of perceptron on test: \n",classifier.confusion_perceptron(w,X_test,Y_test))

df = pd.DataFrame({"x" : [x[1] for x in X_test],"y" : [x[2] for x in X_test], "label" : [1 if (np.dot(x,w)[0]>0) else 0 for x in X_test]})
df.to_csv("perceptron_preds.csv", index=True)






####################################################################################################





mu1,mu0,sigma,C,C1,C2=classifier.cov_and_mu_estimate()

#CoV=(sigma^2)I
boundary=[classifier.bound_disc_same_variance(mu1,mu0,sigma,x) for x in np.arange(0,12,0.1)]
colors=['red' if x==1 else 'blue' for x in Y_train]
plt.scatter(X_train[:,[1]],X_train[:,[2]], color=colors)
plt.plot(boundary,np.arange(0,12,0.1),'-k')
plt.title("discrminant(Cov=sigma^2 I) on train")
plt.legend(["boundary","training data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("discrminant_same_variance_train.png")
plt.show()

print("confusion matrix of discrminant(Cov=sigma^2 I) on train: \n",classifier.confusion_disc_same_var(mu1,mu0,sigma,X_train,Y_train))


colors=['red' if x==1 else 'blue' for x in Y_test]
plt.scatter(X_test[:,[1]],X_test[:,[2]], color=colors)
plt.plot(boundary,np.arange(0,12,0.1),'-k')
#plt.plot(X_train[:,[1]],boundary_r,'-g')
plt.title("discrminant(Cov=sigma^2 I) on test")
plt.legend(["boundary","test data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("discrminant_same_variance_test.png")
plt.show()

print("confusion matrix of discrminant(Cov=sigma^2 I) on test: \n",classifier.confusion_disc_same_var(mu1,mu0,sigma,X_test,Y_test))

df = pd.DataFrame({"x" : [x[1] for x in X_test],"y" : [x[2] for x in X_test], "label" : [1 if classifier.discrminant_same_variance(mu1,mu0,sigma,x[1:3])>0 else 0 for x in X_test]})
df.to_csv("discrminant_same_variance_preds.csv", index=True)



####################################################################################################



#C1=C2=C
boundary=[classifier.bound_disc_same_variance(mu1,mu0,C,x) for x in np.arange(0,12,0.1)]
colors=['red' if x==1 else 'blue' for x in Y_train]
plt.scatter(X_train[:,[1]],X_train[:,[2]], color=colors)
plt.plot(boundary,np.arange(0,12,0.1),'-k')
#plt.plot(X_train[:,[1]],boundary_r,'-g')
plt.title("discrminant(C1=C2) on train")
plt.legend(["boundary","training data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("discrminant_same_covariance_train.png")
plt.show()

print("confusion matrix of discrminant(C1=C2) on train: \n",classifier.confusion_disc_same_var(mu1,mu0,C,X_train,Y_train))


colors=['red' if x==1 else 'blue' for x in Y_test]
plt.scatter(X_test[:,[1]],X_test[:,[2]], color=colors)
plt.plot(boundary,np.arange(0,12,0.1),'-k')
#plt.plot(X_train[:,[1]],boundary_r,'-g')
plt.title("discrminant(C1=C2) on test")
plt.legend(["boundary","test data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("discrminant_same_covariance_test.png")
plt.show()

print("confusion matrix of discrminant(C1=C2) on test: \n",classifier.confusion_disc_same_var(mu1,mu0,C,X_test,Y_test))

df = pd.DataFrame({"x" : [x[1] for x in X_test],"y" : [x[2] for x in X_test], "label" : [1 if classifier.discrminant_same_variance(mu1,mu0,C,x[1:3])>0 else 0 for x in X_test]})
df.to_csv("discrminant_same_covariance_preds.csv", index=True)




#############################################################################################

#C1!=C2
boundary=[classifier.bound_disc_diff_variance(mu1,mu0,C1,C2,x) for x in np.arange(0,12,0.1)]
colors=['red' if x==1 else 'blue' for x in Y_train]
plt.scatter(X_train[:,[1]],X_train[:,[2]], color=colors)
plt.plot(boundary,np.arange(0,12,0.1),'-k')
plt.title("discrminant(C1!=C2) on train")
plt.legend(["boundary","training data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("discrminant_diff_covariance_train.png")
plt.show()

print("confusion matrix of discrminant(C1!=C2) on train: \n",classifier.confusion_disc_diff_var(mu1,mu0,C1,C2,X_train,Y_train))


colors=['red' if x==1 else 'blue' for x in Y_test]
plt.scatter(X_test[:,[1]],X_test[:,[2]], color=colors)
plt.plot(boundary,np.arange(0,12,0.1),'-k')
plt.title("discrminant(C1!=C2) on test")
plt.legend(["boundary","test data"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("discrminant_diff_covariance_test.png")
plt.show()

print("confusion matrix of discrminant(C1!=C2) on test: \n",classifier.confusion_disc_diff_var(mu1,mu0,C1,C2,X_test,Y_test))

df = pd.DataFrame({"x" : [x[1] for x in X_test],"y" : [x[2] for x in X_test], "label" : [1 if classifier.discrminant_diff_variance(mu1,mu0,C1,C2,x[1:3])>0 else 0 for x in X_test]})
df.to_csv("discrminant_diff_covariance_preds.csv", index=True)