# ## Téléchargement de la base d'entraînement
import os
import numpy as np
import matplotlib.pyplot as plt


# ## Chargement de la base en mémoire


import dataset_loader
train_set, valid_set, test_set = dataset_loader.load_mnist()


# Vous pouvez visualiser les différents caractères en changeant l'identifiant de l'image



img_id = 900
X=train_set[0][img_id]
plt.imshow(X.reshape(28,28),cmap='Greys')
print("label: " + str(train_set[1][img_id]))


# Question 1: Donner les caractéristiques de la base d'apprentissage train_set


def getDimDataset(train_set):
    n_training = len(train_set[0])
    n_feature = len(train_set[0][0])
    n_label = len(set(train_set[1]))
    return n_training, n_feature, n_label

(n_training,n_feature,n_label)=getDimDataset(train_set)
print (n_training,n_feature,n_label)


# ## Création du modèle


def init(n_feature,n_label):
    sigma = 1.
    W = np.random.normal(loc=0.0, scale=sigma/np.sqrt(n_feature), size=(n_label,n_feature))
    b = np.zeros((W.shape[0],1))
    return W,b

W,b=init(n_feature,n_label)


# Question 2: Donner les dimensions de W et b ainsi que le nombre total de paramètres du modèle


def printInfo(W,b):
    print("W dimensions: " + str(W.shape))
    print("b dimensions: " + str(b.shape))
    print("Number of parameters: " + str(W.shape[0]*W.shape[1]+b.shape[0]))
    
printInfo(W,b)


# Question 3: Implémenter la fonction forward

def forward(W,b,X):
    """
        Perform the forward propagation
        :param W: the weights
        :param b: the bias
        :param X: the input (minibatch_size x n_input)
        :type W: ndarray
        :type B: ndarray
        :type X: ndarray
        :return: the transformed values
        :rtype: ndarray
    """
    
    z = np.zeros((b.shape[0],1))
    for j in range(z.shape[0]):
        z[j]=W[j].dot(X)+b[j]
    
    return z

z=forward(W,b,X)
print(z)


# Question 4: Implémenter la fonction softmax 
def softmax(z):
    """
        Perform the softmax transformation to the pre-activation values
        :param z: the pre-activation values
        :type z: ndarray
        :return: the activation values
        :rtype: ndarray
    """
    out= np.zeros((z.shape[0],1))
    somme=0
    for i in range(len(z)):
        out[i]=np.exp(z[i])
        somme+=out[i]
    out=out/somme
    
    target= out.argmax()
    print 'target='+str(target)
    return out
out=softmax(z)
print (out)



# Question 5: Implémenter le calcul du gradient de l'erreur 
def gradient_out(out, one_hot_batch):
    """
    compute the gradient w.r.t. the pre-activation values of the softmax z_i
    :param out: the softmax values
    :type out: ndarray
    :param one_hot_batch: the one-hot representation of the labels
    :type one_hot_batch: ndarray
    :return: the gradient w.r.t. z
    :rtype: ndarray
    """
    derror=np.zeros((out.shape[0],1))
    for j in range(out.shape[0]):   
        derror[j]=out[j]-(one_hot_batch[j]==1)    
    return derror
derror=gradient_out(out,np.array([0,0,0,0,0,0,0,0,1,0]))
print derror


# Question 6: Implémenter la fonction du calcul de gradient par rapport aux paramètres

# In[98]:

def gradient(derror, X):
    """
        Compute the gradient w.r.t. the parameters
        :param derror: the gradient w.r.t. z
        :param X: the input (minibatch_size x n_input)
        :param minibatch_size: the minibatch size
        :type derror: ndarray
        :type minibatch: ndarray
        :type minibatch_size: unsigned
        :return: the gradient w.r.t. the parameters
        :rtype: ndarray, ndarray
    """     
    grad_w = np.zeros((derror.shape[0],X.shape[0]))
    grad_b = np.zeros((derror.shape[0]))
    print grad_w.shape, grad_b.shape
    for j in range(derror.shape[0]):
        grad_b[j]=derror[j]
        for i in range(X.shape[0]):
            grad_w[j][i]=derror[j]*X[i]
    return grad_w,grad_b
grad_w, grad_b=gradient(derror,X)
print grad_w, grad_b


# Question 7: Implémenter la fonction de mise à jour des paramètres

def update(eta, W, b, grad_w, grad_b):
    """
        Update the parameters with an update rule
        :param eta: the step-size
        :param W: the weights
        :param b: the bias
        :param grad_w: the gradient w.r.t. the weights
        :param grad_b: the gradient w.r.t. the bias
        :type eta: float
        :type W: ndarray
        :type b: ndarray
        :type grad_w: ndarray
        :type grad_b: ndarray
        :return: the updated parameters
        :rtype: ndarray, ndarray
    """   
    b=b-eta*derror
    for i in range(len(b)):
        for j in range(W.shape[1]):
            W[i][j]=W[i][j]-eta*grad_w[i][j] 
    return W, b
eta=0.1
W,b=update(eta, W, b, grad_w, grad_b)
print b