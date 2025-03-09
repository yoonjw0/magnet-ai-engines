"""
시각화 유틸리티 함수 모듈

시각화에 필요한 공통 유틸리티 함수들을 제공합니다.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_cached_image(cache_file, figsize=(12, 10)):
    """캐시된 이미지 로드"""
    if os.path.exists(cache_file):
        img = plt.imread(cache_file)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.axis('off')
        return fig
    return None

def create_similarity_graph(recommender, n_users, threshold, feature_name=None):
    """
    유사도 기반 그래프 생성
    
    Args:
        recommender: 추천 시스템 인스턴스
        n_users (int): 사용자 수
        threshold (float): 유사도 임계값
        feature_name (str, optional): 특정 특성 이름. None이면 전체 유사도 사용.
        
    Returns:
        networkx.Graph: 생성된 그래프
    """
    G = nx.Graph()
    
    # 노드 추가
    for i in range(min(n_users, len(recommender.profiles))):
        profile = recommender.profiles[i]
        G.add_node(i, 
                  name=profile.get('name', f'User {i}'), 
                  job=profile.get('job_title', ''),
                  specialties=profile.get('specialties', ''),
                  skills=profile.get('skills', ''))
    
    # 엣지 추가
    edge_data = []
    
    if feature_name is not None and feature_name in recommender.feature_similarity_matrices:
        # 특정 특성의 유사도 사용
        similarity_matrix = recommender.feature_similarity_matrices[feature_name]
        for i in range(n_users):
            for j in range(i+1, n_users):
                if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                    similarity = similarity_matrix[i, j]
                    if similarity >= threshold:
                        edge_data.append((i, j, similarity))
    else:
        # 전체 유사도 사용 (total_score 유사도 행렬 사용)
        if 'total_score' in recommender.feature_similarity_matrices:
            similarity_matrix = recommender.feature_similarity_matrices['total_score']
            for i in range(n_users):
                for j in range(i+1, n_users):
                    if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                        similarity = similarity_matrix[i, j]
                        if similarity >= threshold:
                            edge_data.append((i, j, similarity))
        else:
            # total_score가 없는 경우 첫 번째 사용 가능한 특성 사용
            if recommender.feature_similarity_matrices:
                first_feature = list(recommender.feature_similarity_matrices.keys())[0]
                similarity_matrix = recommender.feature_similarity_matrices[first_feature]
                for i in range(n_users):
                    for j in range(i+1, n_users):
                        if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                            similarity = similarity_matrix[i, j]
                            if similarity >= threshold:
                                edge_data.append((i, j, similarity))
            else:
                print("경고: 유사도 행렬이 없습니다. 빈 그래프를 반환합니다.")
    
    # 유사도 기준으로 내림차순 정렬
    edge_data.sort(key=lambda x: x[2], reverse=True)
    
    return G, edge_data

def filter_graph(G, edge_data, n_users, max_edges_factor=3, min_connections=2):
    """
    그래프 필터링
    
    Args:
        G (networkx.Graph): 원본 그래프
        edge_data (dict 또는 list): 엣지 데이터 (딕셔너리 또는 리스트)
        n_users (int): 사용자 수
        max_edges_factor (int): 노드 수 대비 최대 엣지 수 비율
        min_connections (int): 최소 연결 수
        
    Returns:
        networkx.Graph: 필터링된 그래프
    """
    # 상위 엣지만 추가
    max_edges = n_users * max_edges_factor
    
    # edge_data가 딕셔너리인 경우 처리
    if isinstance(edge_data, dict):
        # 유사도 기준으로 정렬
        sorted_edges = sorted(edge_data.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 max_edges개만 사용
        for (i, j), similarity in sorted_edges[:max_edges]:
            G.add_edge(i, j, weight=similarity)
    # edge_data가 리스트인 경우 처리
    elif isinstance(edge_data, list):
        # 상위 max_edges개만 사용
        for i, j, similarity in edge_data[:max_edges]:
            G.add_edge(i, j, weight=similarity)
    
    # 연결이 min_connections 미만인 노드 제거
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < min_connections]
    G.remove_nodes_from(nodes_to_remove)
    
    return G

def create_weighted_similarity_graph(recommender, n_users, threshold, feature_weights, apply_weights=True):
    """
    가중치를 적용한 유사도 그래프 생성
    
    Args:
        recommender: 추천 시스템 인스턴스
        n_users (int): 시각화할 사용자 수
        threshold (float): 연결선을 표시할 유사도 임계값
        feature_weights (dict): 특성별 가중치 딕셔너리
        apply_weights (bool): 가중치 적용 여부
        
    Returns:
        networkx.Graph: 생성된 그래프
        dict: 엣지 데이터
    """
    # 그래프 초기화
    G = nx.Graph()
    
    # 노드 추가
    for i in range(min(n_users, len(recommender.profiles))):
        profile = recommender.profiles[i]
        G.add_node(
            i,
            name=profile.get('name', f'User {i}'),
            job=profile.get('job_title', ''),
            skills=profile.get('skills', ''),
            specialties=profile.get('specialties', '')
        )
    
    # 가중치 적용 유사도 계산
    if apply_weights:
        # 가중 평균 유사도 행렬 계산
        weighted_similarity = np.zeros((n_users, n_users))
        total_weight = sum(feature_weights.values())
        
        if total_weight > 0:  # 가중치 합이 0보다 큰 경우에만 계산
            for feature, weight in feature_weights.items():
                # 특성 이름이 feature_similarity_matrices에 있는지 확인 (영어 이름만 사용)
                if feature in recommender.feature_similarity_matrices and weight > 0:
                    sim_matrix = recommender.feature_similarity_matrices[feature]
                    # 행렬 크기 확인 및 조정
                    if sim_matrix.shape[0] >= n_users:
                        weighted_similarity += (sim_matrix[:n_users, :n_users] * (weight / total_weight))
        else:
            # 가중치 합이 0인 경우 기본 유사도 사용
            # similarity_matrix가 없는 경우 첫 번째 특성 행렬 사용
            if hasattr(recommender, 'similarity_matrix'):
                weighted_similarity = recommender.similarity_matrix[:n_users, :n_users]
            elif recommender.feature_similarity_matrices:
                # 첫 번째 특성 행렬 사용
                first_feature = list(recommender.feature_similarity_matrices.keys())[0]
                weighted_similarity = recommender.feature_similarity_matrices[first_feature][:n_users, :n_users]
            else:
                # 모든 유사도가 0.5인 기본 행렬 생성
                weighted_similarity = np.ones((n_users, n_users)) * 0.5
                np.fill_diagonal(weighted_similarity, 1.0)
    else:
        # 가중치 미적용 시 기본 유사도 사용
        if hasattr(recommender, 'similarity_matrix'):
            weighted_similarity = recommender.similarity_matrix[:n_users, :n_users]
        elif recommender.feature_similarity_matrices:
            # 첫 번째 특성 행렬 사용
            first_feature = list(recommender.feature_similarity_matrices.keys())[0]
            weighted_similarity = recommender.feature_similarity_matrices[first_feature][:n_users, :n_users]
        else:
            # 모든 유사도가 0.5인 기본 행렬 생성
            weighted_similarity = np.ones((n_users, n_users)) * 0.5
            np.fill_diagonal(weighted_similarity, 1.0)
    
    # 엣지 데이터 저장
    edge_data = {}
    
    # 엣지 추가 (임계값 이상인 유사도만)
    for i in range(n_users):
        for j in range(i+1, n_users):
            similarity = weighted_similarity[i, j]
            if similarity >= threshold:
                G.add_edge(i, j, weight=similarity)
                edge_data[(i, j)] = similarity
    
    return G, edge_data 