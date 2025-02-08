import numpy as np
from scipy.spatial import KDTree
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import openslide as slide
from extract_features import get_uni_feature
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

r = 3
def calculate_cancer_patch_density(coords, cancer_flags,patch_size):
    # 创建KDTree用于快速邻近搜索
    tree = KDTree(coords)
    
    # 查询每个点周围的坐标，定义搜索半径
    radius = np.sqrt(r) * patch_size
    densities = []
    # cancer_flags = np.array(cancer_flags)  # Convert to numpy array
    
    for i, point in enumerate(coords):
        # 查询点周围的邻居索引和距离
        indices = tree.query_ball_point(point, r=radius)
        # 计算癌症patch的数量
        
        cancer_count = np.sum(cancer_flags[indices])
        densities.append(cancer_count)
    # print(densities)
    return np.array(densities)

def visualize_density(coords, densities):
    # 将坐标和密度映射到图像
    coords = np.array(coords)
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=densities, cmap='hot', s=100, edgecolors='k', alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='Cancer patch density')
    ax.set_title('Patch Density Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    plt.show()

def propagate_patches(scores,coords, densities, features, feature_threshold,dense_threshold,patch_size):
    # 创建KDTree用于快速邻近搜索
    tree = KDTree(coords)
    
    # 选择所有密度最高的patches为传播的初始点
    max_density = np.max(densities)
    start_indices = np.where(densities == max_density)[0].tolist()
    
    avg_feature = np.mean([features[i] for i in start_indices], axis=0)
    avg_feature = np.mean([features[i] for i in start_indices], axis=0)
    avg_score = np.max(scores[start_indices])
    # start_indices = np.argsort(scores)[-50:].tolist()
    ret_start_indices = start_indices.copy()
    queue = start_indices
    propagated = set(queue)

    # 定义传播函数
    def propagate_from(index):
        nonlocal propagated
        point = coords[index]
        indices = tree.query_ball_point(point, r=np.sqrt(r) * patch_size)  # 3x3格子的范围
        
        for i in indices:
            # print(feature_similarity(features[index], features[i]))

            if i not in propagated and (densities[i] > dense_threshold or feature_similarity(avg_feature, features[i]) > feature_threshold):
                propagated.add(i)
                queue.append(i)
    # print(propagated)
    
    # 传播过程
    while queue:
        current_index = queue.pop(0)
        propagate_from(current_index)
    return propagated,avg_score,ret_start_indices

def feature_similarity(feature1, feature2):
    # 计算两个特征向量的相似度，这里使用余弦相似度
    cosine_similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    adjusted_cosine_similarity = (1 + cosine_similarity) / 2
    return cosine_similarity
def get_feature(wsi_path,patch_pred):
    wsi = slide.OpenSlide(wsi_path)
    slide_id = os.path.basename(wsi_path).split('.')[0]
    dimension = wsi.level_dimensions[0] # given as width, height
    scale = dimension[1] // patch_pred.shape[0]
    coords = []
    for i in range(patch_pred.shape[0]):
        for j in range(patch_pred.shape[1]):
            if patch_pred[i, j] > 0:
                coords.append((j * scale, i * scale))
    print('/home/mdi/WS-FSS-Code/data/dsmil_feature/'+str(slide_id)+'.npy')
    if os.path.exists('/home/mdi/WS-FSS-Code/data/dsmil_feature/'+str(slide_id)+'.npy'):
        return np.load('/home/mdi/WS-FSS-Code/data/dsmil_feature/'+str(slide_id)+'.npy')
    else:
        features = get_uni_feature(wsi,scale,coords)
        np.save('/home/mdi/WS-FSS-Code/data/dsmil_feature/'+str(slide_id)+'.npy', features)
        return features
    

    
def prop(wsi_path,patch_pred,features):
    wsi = slide.OpenSlide(wsi_path)
    dimension = wsi.level_dimensions[0] # given as width, height
    scale = dimension[1] // patch_pred.shape[0]
    coords = []
    cancer_flag = []
    for i in range(patch_pred.shape[0]):
        for j in range(patch_pred.shape[1]):
            if patch_pred[i, j] == 1:
                cancer_flag.append(True)
                coords.append((j * scale, i * scale))
            else:
                cancer_flag.append(False)

    if len(coords) !=0 and len(features) !=0:
        new_pred  = prolif(coords, cancer_flag,features,0.4)
    
        return new_pred
def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores
# def compute_structure_block(features, scores, block_size=100):
#     features = np.array(features)
#     num_samples = features.shape[0]
#     result = np.zeros(num_samples)
#     anti_result = np.zeros(num_samples)
#     # 归一化features
#     norms = np.linalg.norm(features, axis=1)
#     normalized_features = features / norms[:, np.newaxis]
    
#     for start in range(0, num_samples, block_size):
#         end = min(start + block_size, num_samples)
#         block_features = normalized_features[start:end]
        
#         # 计算当前块与所有数据的相似度
#         similarity_matrix = np.dot(block_features, normalized_features.T)
        
#         # 计算每个元素的C_i值
#         temp_score = scores[start:end, np.newaxis]
#         anti_similarity_matrix = 1 - similarity_matrix
        

#         score_sim_product = temp_score * similarity_matrix
#         anti_score_sim_product = temp_score * anti_similarity_matrix

#         denominator = temp_score + similarity_matrix - score_sim_product
#         anti_denominator = temp_score + anti_similarity_matrix - anti_score_sim_product

#         denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
#         anti_denominator = np.where(anti_denominator == 0, np.finfo(float).eps, anti_denominator)

#         block_result = np.sum(score_sim_product / denominator, axis=1)
#         anti_block_result = np.sum(anti_score_sim_product / anti_denominator, axis=1)

#         result[start:end] = block_result
#         anti_result[start:end] = anti_block_result
        

#     # result = to_percentiles(result) 
#     # result /= 100
#     # anti_result = to_percentiles(anti_result)
#     # anti_result /= 100
#     print(result)
#     print(result[1000])
#     print(anti_result)
#     return  result > anti_result

def calc_structure_thres(result, anti_result):
    mask = result > anti_result
    difference_matrix = result[mask] - anti_result[mask]
    data = difference_matrix
    X = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X)
    xx = np.linspace(0, 1, 1000).reshape(-1, 1)
    posterior = gmm.predict_proba(xx)  # shape: (1000, 2)
    p1, p2 = posterior[:, 0], posterior[:, 1]
    diff = p1 - p2

    boundary_candidates = []

    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0: 
            ratio = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
            x_boundary = xx[i] + ratio*(xx[i+1] - xx[i])
            boundary_candidates.append(x_boundary[0])

    zero_idx = np.where(np.isclose(diff, 0, atol=1e-8))[0]
    for idx in zero_idx:
        boundary_candidates.append(xx[idx][0])

    boundary_candidates = sorted(set(boundary_candidates))

    print("在 [0,1] 内找到的分割点:")
    for bc in boundary_candidates:
        return bc

def compute_structure_block(features, scores):
    features = np.array(features)
    num_samples = features.shape[0]
    result = np.zeros(num_samples)
    anti_result = np.zeros(num_samples)
    # 归一化features
    norms = np.linalg.norm(features, axis=1)
    normalized_features = features / norms[:, np.newaxis]
    
    # 预先计算所有特征向量间的相似度矩阵
    similarity_matrix = normalized_features @ normalized_features.T # 这里的范围是【-1，1】，会不会影响效果？不会的话，会不会影响写作？要不换成Tanimoto Coefficient？

    fenzi = (similarity_matrix * scores).sum(axis=1)
    fenmu = similarity_matrix.sum(axis=1) + scores.sum() - (similarity_matrix * scores).sum(axis=1)

    anti_fenzi = scores.sum() - (similarity_matrix * scores).sum(axis=1)
    anti_fenmu = (1 - similarity_matrix).sum(axis=1) + scores.sum() - ((1 - similarity_matrix) * scores).sum(axis=1)

    result = fenzi / fenmu
    anti_result = anti_fenzi / anti_fenmu

    threshold = calc_structure_thres(result, anti_result)
    return  [(a > b) and ((a - b) > threshold) for a, b in zip(result, anti_result)] # 可以通过控制这个0.01来处理消融实验XD

import random
def prolif(scores,coords, cancer_flags,features,feature_threshold,dense_thresold,patch_size,correlation=False):
    
    # if correlation:
    #     cor_scores = np.array(scores, dtype=float)
    #     cor_scores[~np.array(cancer_flags)] = 0
    #     structure_score = compute_structure_block(features, cor_scores)
    #     cancer_flags = np.array(structure_score)
    #     # return structure_score
    densities = calculate_cancer_patch_density(coords, cancer_flags,patch_size)
    propagate_indicies,average_score,start_indices = propagate_patches(scores,coords,densities,features,feature_threshold,dense_thresold,patch_size)



    bool_list = [False]*len(coords)
    for index in propagate_indicies:
        bool_list[index] = True

    if correlation:
        cor_scores = np.array(scores, dtype=float)
        cor_scores[~np.array(bool_list)] = 0
        structure_score = compute_structure_block(features, cor_scores)
        cancer_flags = np.array(structure_score)
        # return structure_score

        return cancer_flags | np.array(bool_list)
    else:
        return bool_list
    # for i in range(len(scores)):
    #     if i in propagate_indicies:            
    #         score = random.uniform(0.8, 0.9)
    #     else:
    #         score = random.uniform(0.2, 0.3)
    #     scores[i] = score
    # print(len(propagate_indicies))
    # # scores[start_indices] = 0.99
    # return scores
