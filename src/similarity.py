
from scipy.sparse import load_npz, csr_matrix, save_npz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

behaviors_matrix = load_npz('data/pv_behaviors.npz')

pv_behaviors = pd.DataFrame.sparse.from_spmatrix(behaviors_matrix)

print(pv_behaviors.shape)

############ PCA ####################


# from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# Assuming `X` is your sparse matrix
svd = TruncatedSVD(n_components=1000).fit(pv_behaviors)
cumulative_variance_ratio = np.cumsum(svd.explained_variance_ratio_)

# generate scree plot (simple line segment plot that shows the fraction of 
# total variance in the data as explained or represented by each Principal Component)
explained_variance = svd.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# # Find the number of components for a certain variance

# n_comp = 3
# n_components_95 = None

# while n_components_95 == None:

#     svd = TruncatedSVD(n_components=n_comp).fit(pv_behaviors)
#     cumulative_variance_ratio = np.cumsum(svd.explained_variance_ratio_)
#     print(n_comp)
#     print(cumulative_variance_ratio)

#     n_components_95 = np.where(cumulative_variance_ratio > 0.95)[0]
#     n_components_95 = n_components_95[0] + 1 if n_components_95.size else None
#     n_comp += 1

# n_components_99 = np.where(cumulative_variance_ratio > 0.99)[0]
# n_components_99 = n_components_99[0] + 1 if n_components_99.size else None

# X_reduced = svd.fit_transform(pv_behaviors)

# pca = PCA().fit(pv_behaviors)

# cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)


# print(f"Number of components for 95% variance: {n_components_95}")
# print(f"Number of components for 99% variance: {n_components_99}")
# # print(user_id_code_df_pca.shape)


######### DISTANCE FUNCTIONS ###########

def pairwise_distance(A, B, p=2):
    """
    A: ndarray - m x n matrix, where each row is a vector of length n 
    B: ndarray - k x n matrix, where each row is a vector of length n 
    p: int - the order of the norm 

    Precondition: if A is m x n, then b must be k x n,
    i.e. the two inputs should agree in dimmension in the second component
    """
    a = A[:, None, :]
    b = B[None, :, :]
    
    return np.linalg.norm(a-b, axis=-1, ord=p)

# ##### Compute the dot product for jaccard index ########
'''
sparse_matrix = csr_matrix(pv_behaviors)
print(sparse_matrix.shape)
user_user_matrix = sparse_matrix.dot(sparse_matrix.T)
# print(user_user_matrix.shape) = (50000, 50000)
save_npz('jaccard_similarity.npz', user_user_matrix)
'''