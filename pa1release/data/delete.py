import random,numpy as np
from sklearn.neighbors import KNeighborsClassifier

def KNN1(X_tr,X_te,y_tr):
    n_pred = np.shape(X_te)[0]
    pred = np.zeros(n_pred)
    for i in range(len(X_te)):
        cur_x = X_te[i]
        min_dist = float('inf')
        for j in range(len(X_tr)):
            cur_neighbor = X_tr[j]
            temp_dist = 0
            for dim in range(len(cur_x)):
                temp_dist += (cur_x[dim] - cur_neighbor[dim])**2
            temp_dist = np.sqrt(temp_dist)
            if temp_dist < min_dist:
                min_dist = temp_dist
                pred[i] = y_tr[j]
    return pred

def predict_sparse(X_tr,x_sample,y_tr):
    min_dist = float('inf')
    for sample_idx,idx_val in enumerate(X_tr):
        idx = idx_val[0] #List of indices for a training example
        val = idx_val[1] #List of corresponding values
        x_copy = x_sample
        for value_idx,dim in enumerate(idx):
            x_copy[dim] -= val[value_idx]
        temp_dist = x_copy**2
        temp_dist = np.sum(temp_dist)
        temp_dist = np.sqrt(temp_dist)

        if temp_dist < min_dist:
            min_dist = temp_dist
            pred = y_tr[sample_idx]
    return pred


def pca_prediction(V,X_tr,X_te,y_tr):
    n_pred = np.shape(X_te)[0]
    pred = np.zeros(n_pred)

    pca_X_tr = np.matmul(V,X_tr.T) #dim (Dxn)
    pca_X_te = np.matmul(V,X_te.T) #dim (Dxm)

    for sample_idx, pca_x_te in enumerate(pca_X_te): # m times
        min_dist = float('inf')
        for pca_x_tr in pca_X_tr: #n times
            temp_dist = 0
            for dim in range(len(pca_x_tr)): #D times
                temp_dist += (pca_x_te[dim] - pca_x_tr[dim]) ** 2
            temp_dist = np.sqrt(temp_dist)
            if temp_dist < min_dist:
                min_dist = temp_dist
                pred[i] = y_tr[sample_idx]
    return pred

if __name__ == "__main__":
    X_tr = np.array([[1, 4], [2, 2], [3, 2]])
    y_tr = np.array([0,1,2])
    print(X_tr)

    X_te = np.array([2,4])
    V = np.array([5,1])

    dist = []
    for i in range(len(X_tr)):
        print(i,np.linalg.norm(X_tr[i]-X_te))
        dist.append(np.linalg.norm(X_tr[i]-X_te))
    print("pred: ",np.argmin(dist))

    pca_X_tr = np.matmul(V,X_tr.T) #dim (Dxn)
    print("Pca_tr: ",pca_X_tr)
    pca_X_te = np.matmul(V,X_te.T) #dim (Dxm)
    print("pca_X_te: ", pca_X_te)
    dist1 = []
    for i in range(len(X_tr)):
        print(i,np.linalg.norm(pca_X_tr[i]-pca_X_te))
        dist1.append(np.linalg.norm(pca_X_tr[i]-pca_X_te))
    print("pca_pred: ",np.argmin(dist1))
    print(np.argmin(dist1)==np.argmin(dist))

    #### CODE FOR PART 2(A) #####
    # np.random.seed(4)
    #
    # n,d = 20,6
    # train_split = .8
    # k = 1
    # X = np.random.randint(0, 5, size=(n, d))
    # y = np.random.randint(0, 3, size=(n))
    #
    # X_tr = X[:int(np.floor(n*train_split)),:]
    # y_tr = y[:int(np.floor(n*train_split))]
    # X_te =  X[int(np.ceil(n*train_split)):,:]
    # pred = KNN1(X_tr,X_te,y_tr,k)
    # print(pred)
    #
    # knn3 = KNeighborsClassifier(n_neighbors=1)
    # knn3.fit(X_tr, y_tr)
    # print(knn3.predict(X_te))

    ##### CODE FOR PART 2(c)
    # np.random.seed(3)
    # d = 500
    # n , l =5, 10 #l is to non_zero elements in the sparse data samples
    # X_data = []
    # x_sample = np.random.randint(0, 5, size=(d))
    # for i in range(n):
    #     idx = np.random.randint(0, d, size=(l))
    #     idx = list(set(idx))
    #     idx = sorted(idx)
    #     values = np.random.randint(0, 10, size=(len(idx)))
    #     X_data.append([idx, values])
    # y = np.random.randint(0, 3, size=(n))
    #
    #
    # print(predict_sparse(X_data, x_sample, y))
