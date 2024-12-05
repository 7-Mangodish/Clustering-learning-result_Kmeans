from __future__ import print_function 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.spatial.distance import cdist

def kmeans_display(X,centers,label):
    # print(X)

    df = pd.DataFrame(X, columns=['Điểm trung bình hệ 10','Điểm rèn luyện'])
    df['Cluster'] = label
    # print(df)
    fig = px.scatter(
        df,
        x = 'Điểm trung bình hệ 10',
        y = 'Điểm rèn luyện',
        color = 'Cluster',
        hover_data = ['Điểm trung bình hệ 10', 'Điểm rèn luyện'],
        symbol = 'Cluster',
        title = 'Kết quả phân cụm')
    
    centers_df = pd.DataFrame(centers, columns=['Điểm trung bình hệ 10','Điểm rèn luyện'])
    fig.add_scatter(
        x=centers_df['Điểm trung bình hệ 10'],
        y=centers_df['Điểm rèn luyện'],
        mode='markers',
        marker=dict(size=10, color='red', symbol='star'),
        name='Centroids'  # Tên hiển thị cho điểm trung tâm        
    )

    # Di chuyển color bar sang phải để tránh chèn vào tên cluster
    fig.update_layout(
        legend=dict(
            x=1.15, y=.5,  # Di chuyển legend sang phải
            font=dict(size=10, color="black")
        ),
        # coloraxis_colorbar_x=1.2  # Di chuyển color bar xa hơn nếu cần
    )  
    fig.show()

def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    # D là một mảng 2 chiều D[i, j]: là khoảng cách từ điểm i trong tập dl X đến điểm j trong centers
    D = cdist(X, centers) 
    # print(D.shape)
    # for i in D:
    #     print(i)

    # return index of the closest center
    # tmp = np.argmin(D, axis = 1)
    # print(tmp)
    return np.argmin(D, axis = 1) 

def kmeans_update_centers(X, labels, K):
    # Tao mot mang luu cac centers moi, row = K, column bang so cot cua data
    centers = np.zeros((K, X.shape[1]))
    # print(centers)
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # print("Voi diem trung tam k=", k)
        # print("Toa do cac diem ung voi cluster k:",k)
        # print(Xk)
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
        # print('Toa do cac centers moi:',centers[k])
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    # print('Init center:', centers)
    labels = []
    it = 0 
    while True:
        # print("vong lap:", it)
        # print('Diem trung tam cua vong lap thu', it)
        # print(centers[-1])
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)

        # print("new_centers", new_centers)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1

    return (centers, labels, it)


# means = [[2, 2], [8, 3], [3, 6]] # Danh sách chứa các tọa độ trung tâm

# """
#     Ma trận hiệp phương sai
#     Các điểm dữ liệu có độ phân tán đều nhau, không có sự tương quan giữa các chiều
# """
# cov = [[1, 0], [0, 1]]

# """
#     Tạo dữ liệu, với 500 điểm mỗi cụm
# """
# N = 5
# X0 = np.random.multivariate_normal(means[0], cov, N)
# X1 = np.random.multivariate_normal(means[1], cov, N)
# X2 = np.random.multivariate_normal(means[2], cov, N)

# # Kết hợp dữ liệu theo chiều trục 0
# X = np.concatenate((X0, X1, X2), axis = 0)

# #Định nghĩa số cụm
# K = 3 

# """
#     Tạo một danh sách chứa N số 0, 1, 2. mỗi cái tương ứng với 1 cụm
#     chuyển thành một mảng numpy 
# """ 
# original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# (centers, labels, it) = kmeans(X, K)
# print('Centers found by our algorithm:')
# print(centers[-1])
# print(type(labels), type(labels[-1]),labels[-1])
# kmeans_display(X, centers[-1] ,labels[-1] )


