import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('linear_reg_train_data.csv')
test=pd.read_csv('linear_reg_test_data.csv')
train=train.to_numpy()
test=test.to_numpy()

X_train,Y_train=train[:,0:-1],train[:,[-1]]
X_test,Y_test=test[:,0:-1],test[:,[-1]]


class linear_reg:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        self.b=None
    def RSS(self,X,Y,B):
        Z=Y-np.dot(X,B)
        return np.dot(Z.T,Z)[0][0]
    def basis_expand(self,X,n):
        Z=np.zeros((X.shape[0],n+1))
        for i in range(X.shape[0]):
            for j in range(n+1):
                Z[i][j]=X[i]**j
        return Z
    def standard_lin_reg(self,n):
        Z=self.basis_expand(self.X,n)
        B=np.dot(np.dot(np.linalg.inv(np.dot(Z.T,Z)),Z.T),self.Y)
        return B
    def ridge_lin_reg(self,n,l):
        Z=self.basis_expand(self.X,n)
        B=np.dot(np.dot(np.linalg.inv(np.dot(Z.T,Z)+l*np.identity(n+1)),Z.T),self.Y)
        return B
    def prediction(self,Z,B):
        return np.dot(Z,B)
    

##################################################################################################    


#standard linear regression
linear=linear_reg(X_train,Y_train)

n=4
B=linear.standard_lin_reg(n)
Z=linear.basis_expand(X_train,n)
Y_train_pred=linear.prediction(Z,B)
plt.plot(X_train, Y_train,'.r')
plt.plot(X_train, Y_train_pred,'.b')
plt.title("Standard Linear plot on training data for n={}".format(n))
plt.legend(["training data","Prediction"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("standard_train_fit.png")
plt.show()



print("coefficients of best fit for standard linear = \n", B)

print("RSS on training data: ",linear.RSS(Z,Y_train,B))


Z=linear.basis_expand(X_test,n)
Y_test_pred=linear.prediction(Z,B)
plt.plot(X_test, Y_test,'*r')
plt.plot(X_test, Y_test_pred,'.b')
plt.title("Standard Linear plot on test data for n={}".format(n))
plt.legend(["test data","Prediction"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("standard_test_plot.png")
plt.show()

print("RSS on test data: ",linear.RSS(Z,Y_test,B))

X=np.arange(0,10,0.1)#.reshape((100,1))
Z=linear.basis_expand(X,n)
Y_pred=linear.prediction(Z,B)
plt.plot(X_train, Y_train,'.r')
plt.plot(X, Y_pred,'-k')
plt.title("Standard Linear best fit curve for n={}".format(n))
plt.legend(["training data","curve"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("standard_best_fit_curve.png")
plt.show()

df = pd.DataFrame({"x" : [x[0] for x in X_test], "y" : [x[0] for x in Y_test_pred]})
df.to_csv("standard_linear_preds.csv", index=True)





########################################################################################




#overfit case
n=5
B=linear.standard_lin_reg(n)
Z=linear.basis_expand(X_train,n)
Y_train_pred=linear.prediction(Z,B)
plt.plot(X_train, Y_train,'.r')
plt.plot(X_train, Y_train_pred,'.b')
plt.title("Standard Linear plot on training data for n={}".format(n))
plt.legend(["training data","Prediction"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("standard_train_fit_n=5.png")
plt.show()



print("coefficients of best fit for standard linear degree 5 = \n", B)

print("RSS on training data degree 5: ",linear.RSS(Z,Y_train,B))


Z=linear.basis_expand(X_test,n)
Y_test_pred=linear.prediction(Z,B)
plt.plot(X_test, Y_test,'*r')
plt.plot(X_test, Y_test_pred,'.b')
plt.title("Standard Linear plot on test data for n={}".format(n))
plt.legend(["test data","Prediction"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("standard_test_plot_n=5.png")
plt.show()

print("RSS on test data degree 5: ",linear.RSS(Z,Y_test,B))

X=np.arange(0,10,0.1)#.reshape((100,1))
Z=linear.basis_expand(X,n)
Y_pred=linear.prediction(Z,B)
plt.plot(X_train, Y_train,'.r')
plt.plot(X, Y_pred,'-k')
plt.title("Standard Linear best fit curve for n={}".format(n))
plt.legend(["training data","curve"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("standard_best_fit_curve_n=5.png")
plt.show()

df = pd.DataFrame({"x" : [x[0] for x in X_test], "y" : [x[0] for x in Y_test_pred]})
df.to_csv("standard_linear_preds_overfit.csv", index=True)



train_error=[]
test_error=[]
degree=[]
for i in range(3,10):
    degree.append(i)
    B=linear.standard_lin_reg(i)
    Z=linear.basis_expand(X_train,i)
    train_error.append(linear.RSS(Z,Y_train,B))
    Z=linear.basis_expand(X_test,i)
    test_error.append(linear.RSS(Z,Y_test,B))    
    
plt.plot(degree,train_error,'-b')
plt.plot(degree,test_error,'-r')
plt.title("RSS vs n")
plt.legend(["train_error","test_error"])
plt.xlabel("n")
plt.ylabel("RSS")
#plt.savefig("RSS_vs_n.png")
plt.show()




######################################################################



#ridge regression
n=5
Lambda=16
B=linear.ridge_lin_reg(n,Lambda)
Z=linear.basis_expand(X_train,n)
Y_train_pred=linear.prediction(Z,B)
plt.plot(X_train, Y_train,'*r')
plt.plot(X_train, Y_train_pred,'.b')
plt.title("Ridge plot on train for n={} and lambda={}".format(n,Lambda))
plt.legend(["training data","Prediction"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("ridge_train_fit.png")
plt.show()

print("coefficients of best fit for ridge = \n", B)

print("RSS on training data for ridge: ",linear.RSS(Z,Y_train,B))

Z=linear.basis_expand(X_test,n)
Y_test_pred=linear.prediction(Z,B)
plt.plot(X_test, Y_test,'*r')
plt.plot(X_test, Y_test_pred,'.b')
plt.title("Ridge plot on test for n={} and lambda={}".format(n,Lambda))
plt.legend(["test data","Prediction"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("ridge_test_plot.png")
plt.show()

print("RSS on test data for ridge: ",linear.RSS(Z,Y_test,B))

X=np.arange(0,10,0.1)#.reshape((100,1))
Z=linear.basis_expand(X,n)
Y_pred=linear.prediction(Z,B)
plt.plot(X_train, Y_train,'.r')
plt.plot(X, Y_pred,'-k')
plt.title("ridge best fit curve for n={} and lambda={}".format(n,Lambda))
plt.legend(["training data","curve"])
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig("ridge_best_fit_curve.png")
plt.show()

df = pd.DataFrame({"x" : [x[0] for x in X_test], "y" : [x[0] for x in Y_test_pred]})
df.to_csv("ridge_preds.csv", index=True)


train_error=[]
test_error=[]
Lambda=[]
n=5
for i in np.arange(0,20,0.1):
    Lambda.append(i)
    B=linear.ridge_lin_reg(n,i)
    Z=linear.basis_expand(X_train,n)
    train_error.append(linear.RSS(Z,Y_train,B))
    Z=linear.basis_expand(X_test,n)
    test_error.append(linear.RSS(Z,Y_test,B))
plt.plot(Lambda,train_error,'-b')
plt.plot(Lambda,test_error,'-r')
plt.title("RSS_vs_lambda for n=5")
plt.legend(["train_error","test_error"])
plt.xlabel("lambda")
plt.ylabel("RSS")
#plt.savefig("RSS_vs_lambda.png")
plt.show()