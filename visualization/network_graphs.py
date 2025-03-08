"""
네트워크 그래프 시각화 모듈

사용자 간 관계를 네트워크 그래프로 시각화하는 기능을 제공합니다.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import platform
from .utils import load_cached_image, create_similarity_graph, filter_graph, ensure_dir

# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

def plot_force_directed_graph(recommender, n_users=30, threshold=0.75, min_connections=2, 
                             figsize=(14, 14), save_path=None, cache_dir="visualization_cache", 
                             use_cache=True, feature_name=None):
    """
    사용자 간 포스 다이렉트 그래프 시각화 (정적)
    
    Args:
        recommender: 추천 시스템 인스턴스
        n_users (int): 시각화할 사용자 수
        threshold (float): 연결선을 표시할 유사도 임계값
        min_connections (int): 노드가 가져야 할 최소 연결 수
        figsize (tuple): 그림 크기
        save_path (str, optional): 저장 경로
        cache_dir (str): 캐시 디렉토리
        use_cache (bool): 캐시 사용 여부
        feature_name (str, optional): 특정 특성 이름. None이면 전체 유사도 사용.
        
    Returns:
        matplotlib.figure.Figure: 생성된 그림 객체
    """
    # 캐시 파일 경로
    feature_suffix = f"_{feature_name}" if feature_name else ""
    cache_file = os.path.join(cache_dir, f"force_directed_graph_n{n_users}_t{threshold}_min{min_connections}{feature_suffix}.png")
    
    # 캐시 확인
    if use_cache and os.path.exists(cache_file) and save_path is None:
        print(f"캐시된 포스 다이렉트 그래프를 로드합니다: {cache_file}")
        return load_cached_image(cache_file, figsize)
    
    # 그래프 생성
    G, edge_data = create_similarity_graph(recommender, n_users, threshold, feature_name)
    
    # 그래프 필터링
    G = filter_graph(G, edge_data, n_users, max_edges_factor=3, min_connections=min_connections)
    
    # 그래프가 비어있는 경우 처리
    if len(G.nodes()) == 0:
        print(f"경고: 임계값 {threshold}와 최소 연결 수 {min_connections}로 인해 표시할 노드가 없습니다.")
        print("임계값을 낮추거나 최소 연결 수를 줄여보세요.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "표시할 노드가 없습니다.\n임계값을 낮추거나 최소 연결 수를 줄여보세요.", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # 레이아웃 계산 (포스 다이렉트 알고리즘)
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=figsize)
    
    # 유사도 범위에 따라 다른 스타일로 엣지 그룹화
    high_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 0.9]
    medium_edges = [(u, v) for u, v, d in G.edges(data=True) if 0.8 <= d['weight'] < 0.9]
    low_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0.8]
    
    # 높은 유사도 엣지
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=high_edges,
        width=2.5,
        alpha=0.8,
        edge_color='darkblue',
        ax=ax
    )
    
    # 중간 유사도 엣지
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=medium_edges,
        width=1.5,
        alpha=0.6,
        edge_color='royalblue',
        ax=ax
    )
    
    # 낮은 유사도 엣지
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=low_edges,
        width=1.0,
        alpha=0.4,
        edge_color='lightblue',
        ax=ax
    )
    
    # 노드 그리기 (연결 정도에 따라 크기 조절)
    node_degrees = dict(G.degree())
    node_sizes = [node_degrees[n] * 80 + 200 for n in G.nodes()]
    
    # 노드 색상 - 연결 정도에 따라 그라데이션
    node_colors = [node_degrees[n] for n in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap='viridis',
        alpha=0.8,
        ax=ax
    )
    
    # 연결 개수가 많은 상위 N개 노드만 레이블 표시
    top_n = 10  # 상위 N개 노드 (필요에 따라 조정 가능)
    node_degrees = dict(G.degree())
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:top_n]]
    
    # 상위 노드만 레이블 표시, 나머지는 빈 문자열로 설정
    labels = {}
    for n in G.nodes():
        if n in top_nodes:
            labels[n] = G.nodes[n]['name']
        else:
            labels[n] = ""
    
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_family='sans-serif',
        ax=ax
    )
    
    # 범례 추가
    legend_elements = [
        mpatches.Patch(color='darkblue', alpha=0.8, label='유사도 ≥ 0.9'),
        mpatches.Patch(color='royalblue', alpha=0.6, label='0.8 ≤ 유사도 < 0.9'),
        mpatches.Patch(color='lightblue', alpha=0.4, label='유사도 < 0.8')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', title='연결 강도')
    
    title_text = f"사용자 간 유사도 네트워크 (임계값: {threshold}, 최소 연결 수: {min_connections})"
    if feature_name:
        title_text = f"{feature_name} 기반 " + title_text
    
    plt.title(title_text, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # 저장
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 캐시에 저장
    if use_cache:
        ensure_dir(cache_dir)
        plt.savefig(cache_file, dpi=300, bbox_inches='tight')
    
    return fig

def plot_interactive_force_directed_graph(recommender, n_users=30, threshold=0.75, min_connections=2, 
                                         figsize=(800, 800), save_path=None, cache_dir="visualization_cache", 
                                         use_cache=True, feature_name=None):
    """
    사용자 간 인터랙티브 포스 다이렉트 그래프 시각화 (Plotly 사용)
    
    Args:
        recommender: 추천 시스템 인스턴스
        n_users (int): 시각화할 사용자 수
        threshold (float): 연결선을 표시할 유사도 임계값
        min_connections (int): 노드가 가져야 할 최소 연결 수
        figsize (tuple): 그림 크기 (픽셀)
        save_path (str, optional): 저장 경로
        cache_dir (str): 캐시 디렉토리
        use_cache (bool): 캐시 사용 여부
        feature_name (str, optional): 특정 특성 이름. None이면 전체 유사도 사용.
        
    Returns:
        plotly.graph_objects.Figure: 생성된 그림 객체
    """
    # 캐시 파일 경로
    feature_suffix = f"_{feature_name}" if feature_name else ""
    cache_file = os.path.join(cache_dir, f"interactive_force_graph_n{n_users}_t{threshold}_min{min_connections}{feature_suffix}.html")
    
    # 그래프 생성
    G, edge_data = create_similarity_graph(recommender, n_users, threshold, feature_name)
    
    # 그래프 필터링
    G = filter_graph(G, edge_data, n_users, max_edges_factor=3, min_connections=min_connections)
    
    # 그래프가 비어있는 경우 처리
    if len(G.nodes()) == 0:
        print(f"경고: 임계값 {threshold}와 최소 연결 수 {min_connections}로 인해 표시할 노드가 없습니다.")
        print("임계값을 낮추거나 최소 연결 수를 줄여보세요.")
        # 빈 그래프 반환
        fig = go.Figure()
        fig.update_layout(
            title="표시할 노드가 없습니다. 임계값을 낮추거나 최소 연결 수를 줄여보세요.",
            width=figsize[0],
            height=figsize[1]
        )
        return fig
    
    # 레이아웃 계산 (포스 다이렉트 알고리즘)
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # 노드 데이터 준비
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []  # 연결 정도에 따른 색상 변화
    
    # 연결 정도 범위 계산
    degrees = dict(G.degree())
    min_degree = min(degrees.values()) if degrees else 0
    max_degree = max(degrees.values()) if degrees else 1
    degree_range = max(1, max_degree - min_degree)
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # 선택된 사용자와 높은 강도로 연결된 상위 N명의 노드 식별
        top_n = 10  # 상위 N개 노드 (필요에 따라 조정 가능)
        
        # 가장 연결이 많은 노드를 중심 사용자로 선택
        node_degrees = dict(G.degree())
        center_node = max(node_degrees.items(), key=lambda x: x[1])[0]
        
        # 중심 노드와 높은 강도로 연결된 노드 식별
        top_nodes = [center_node]  # 중심 노드는 항상 포함
        
        # 중심 노드와 연결된 노드들의 연결 강도(weight) 확인
        connected_nodes = []
        for neighbor in G.neighbors(center_node):
            if G.has_edge(center_node, neighbor):
                weight = G[center_node][neighbor]['weight']
                connected_nodes.append((neighbor, weight))
        
        # 연결 강도(weight)에 따라 정렬하고 상위 N개 선택
        sorted_connected = sorted(connected_nodes, key=lambda x: x[1], reverse=True)
        top_connected = [node for node, _ in sorted_connected[:top_n-1]]  # 중심 노드를 제외한 상위 N-1개
        top_nodes.extend(top_connected)
        
        # 노드 텍스트 정보
        name = G.nodes[node]['name']
        job = G.nodes[node]['job']
        specialties = G.nodes[node]['specialties']
        degree = degrees[node]
        
        # 더 자세한 호버 텍스트 (모든 노드에 대해 동일)
        feature_text = f"{feature_name} " if feature_name else ""
        hover_text = (f"<b>{name}</b><br>"
                     f"직업: {job}<br>"
                     f"전문분야: {specialties}<br>"
                     f"연결 수: {degree}<br>"
                     f"평균 {feature_text}유사도: {sum(G[node][neighbor]['weight'] for neighbor in G[node]) / degree:.4f}")
        
        # 상위 노드만 이름 표시, 나머지는 빈 문자열
        if node in top_nodes:
            if node == center_node:  # 중심 노드는 강조 표시
                node_text.append(f"<b>{name}</b>")
            else:
                node_text.append(name)  # 상위 노드는 이름만 표시
        else:
            node_text.append("")  # 나머지는 빈 문자열
            
        # 호버 텍스트는 별도로 설정 (아래에서 사용)
        
        # 노드 크기 (연결 정도에 따라)
        node_size.append(degree * 3 + 10)
        
        # 노드 색상 (연결 정도에 따라 그라데이션)
        # 연결이 많을수록 진한 색상
        color_intensity = (degree - min_degree) / degree_range
        node_color.append(color_intensity)
    
    # 엣지 데이터 준비
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # 엣지 좌표 (선 그리기)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # 선 분리를 위한 None
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)  # 선 분리를 위한 None
        
        # 엣지 호버 텍스트
        weight = G.edges[edge]['weight']
        user1 = G.nodes[edge[0]]['name']
        user2 = G.nodes[edge[1]]['name']
        feature_text = f"{feature_name} " if feature_name else ""
        edge_text.append(f"{user1} - {user2}<br>{feature_text}유사도: {weight:.4f}")
        edge_text.append(f"{user1} - {user2}<br>{feature_text}유사도: {weight:.4f}")
        edge_text.append(None)
    
    # 엣지 트레이스 (유사도에 따라 여러 그룹으로 나눔)
    edge_traces = []
    
    # 유사도 범위에 따라 다른 두께와 투명도로 엣지 그룹화
    edge_groups = {
        '유사도: 매우 높음 (≥ 0.9)': {'min': 0.9, 'width': 3.0, 'color': 'rgba(50, 50, 200, 0.8)'},
        '유사도: 높음 (≥ 0.8)': {'min': 0.8, 'width': 2.0, 'color': 'rgba(100, 100, 200, 0.6)'},
        '유사도: 중간 (≥ 0.7)': {'min': 0.7, 'width': 1.5, 'color': 'rgba(150, 150, 200, 0.4)'},
        '유사도: 낮음 (< 0.7)': {'min': 0.0, 'width': 1.0, 'color': 'rgba(200, 200, 200, 0.3)'}
    }
    
    # 각 엣지 그룹별로 데이터 분리
    for group_name, group_props in edge_groups.items():
        group_x = []
        group_y = []
        group_text = []
        
        for i, edge in enumerate(G.edges()):
            weight = G.edges[edge]['weight']
            if weight >= group_props['min'] and (i*3 < len(edge_x)):
                idx = i * 3  # 각 엣지는 3개의 좌표로 구성 (시작, 끝, None)
                group_x.extend([edge_x[idx], edge_x[idx+1], edge_x[idx+2]])
                group_y.extend([edge_y[idx], edge_y[idx+1], edge_y[idx+2]])
                if idx < len(edge_text):
                    group_text.extend([edge_text[idx], edge_text[idx+1], edge_text[idx+2]])
        
        if group_x:  # 그룹에 데이터가 있는 경우만 트레이스 추가
            edge_trace = go.Scatter(
                x=group_x, y=group_y,
                mode='lines',
                line=dict(
                    width=group_props['width'],
                    color=group_props['color']
                ),
                hoverinfo='text',
                text=group_text,
                name=group_name
            )
            edge_traces.append(edge_trace)
    
    # 노드 트레이스
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",  # 텍스트 위치를 노드 위쪽 중앙으로 설정
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='연결 정도',
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        ),
        name='사용자'
    )
    
    # 그래프 레이아웃
    title_text = f'사용자 간 유사도 네트워크 (임계값: {threshold}, 최소 연결 수: {min_connections})'
    if feature_name:
        title_text = f'{feature_name} 기반 ' + title_text
    
    layout = go.Layout(
        title=dict(
            text=title_text,
            font=dict(size=16)
        ),
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            ),
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1
        )
    )
    
    # 그래프 생성
    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    
    # 저장
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.write_html(save_path)
    
    # 캐시에 저장
    if use_cache:
        ensure_dir(cache_dir)
        fig.write_html(cache_file)
    
    return fig

def plot_user_centered_network(recommender, user_idx, n_users=30, threshold=0.7, min_connections=2,
                              figsize=(10, 8), save_path=None, cache_dir="visualization_cache",
                              use_cache=True, feature_name=None):
    """
    선택된 사용자를 중심으로 한 네트워크 그래프 시각화
    
    Args:
        recommender: 추천 시스템 인스턴스
        user_idx (int): 중심에 위치할 사용자 인덱스
        n_users (int): 시각화할 사용자 수
        threshold (float): 연결선을 표시할 유사도 임계값
        min_connections (int): 노드가 가져야 할 최소 연결 수
        figsize (tuple): 그림 크기
        save_path (str, optional): 저장 경로
        cache_dir (str): 캐시 디렉토리
        use_cache (bool): 캐시 사용 여부
        feature_name (str, optional): 특정 특성 이름. None이면 전체 유사도 사용.
        
    Returns:
        matplotlib.figure.Figure: 생성된 그림 객체
    """
    # 캐시 파일 경로
    feature_suffix = f"_{feature_name}" if feature_name else ""
    cache_file = os.path.join(cache_dir, f"user_centered_network_u{user_idx}_n{n_users}_t{threshold}_min{min_connections}{feature_suffix}.png")
    
    # 캐시 확인
    if use_cache and os.path.exists(cache_file) and save_path is None:
        print(f"캐시된 사용자 중심 네트워크 그래프를 로드합니다: {cache_file}")
        return load_cached_image(cache_file, figsize)
    
    # 그래프 생성
    if feature_name is not None and feature_name in recommender.feature_similarity_matrices:
        G, edge_data = create_similarity_graph(recommender, n_users, threshold, feature_name)
    else:
        G, edge_data = create_similarity_graph(recommender, n_users, threshold)
    
    # 그래프 필터링
    G = filter_graph(G, edge_data, n_users, max_edges_factor=3, min_connections=min_connections)
    
    # 선택된 사용자가 그래프에 없는 경우 추가
    if user_idx not in G.nodes():
        profile = recommender.profiles[user_idx]
        G.add_node(user_idx,
                  name=profile.get('name', f'User {user_idx}'),
                  job=profile.get('job_title', ''),
                  specialties=profile.get('specialties', ''),
                  skills=profile.get('skills', ''))
        
        # 선택된 사용자와 다른 노드 간의 엣지 추가
        for node in list(G.nodes()):
            if node != user_idx:
                if feature_name is not None and feature_name in recommender.feature_similarity_matrices:
                    similarity_matrix = recommender.feature_similarity_matrices[feature_name]
                    if user_idx < similarity_matrix.shape[0] and node < similarity_matrix.shape[1]:
                        similarity = similarity_matrix[user_idx, node]
                        if similarity >= threshold:
                            G.add_edge(user_idx, node, weight=similarity)
                else:
                    # total_score 유사도 행렬 사용
                    if 'total_score' in recommender.feature_similarity_matrices:
                        similarity_matrix = recommender.feature_similarity_matrices['total_score']
                        if user_idx < similarity_matrix.shape[0] and node < similarity_matrix.shape[1]:
                            similarity = similarity_matrix[user_idx, node]
                            if similarity >= threshold:
                                G.add_edge(user_idx, node, weight=similarity)
                    else:
                        # total_score가 없는 경우 첫 번째 사용 가능한 특성 사용
                        if recommender.feature_similarity_matrices:
                            first_feature = list(recommender.feature_similarity_matrices.keys())[0]
                            similarity_matrix = recommender.feature_similarity_matrices[first_feature]
                            if user_idx < similarity_matrix.shape[0] and node < similarity_matrix.shape[1]:
                                similarity = similarity_matrix[user_idx, node]
                                if similarity >= threshold:
                                    G.add_edge(user_idx, node, weight=similarity)
    
    # 그래프가 비어있는 경우 처리
    if len(G.nodes()) == 0:
        print(f"경고: 임계값 {threshold}와 최소 연결 수 {min_connections}로 인해 표시할 노드가 없습니다.")
        print("임계값을 낮추거나 최소 연결 수를 줄여보세요.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "표시할 노드가 없습니다.\n임계값을 낮추거나 최소 연결 수를 줄여보세요.",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # 선택된 사용자를 중심으로 레이아웃 계산
    # 선택된 사용자는 중앙에 위치하고, 다른 노드들은 주변에 배치
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # 선택된 사용자를 중앙에 배치
    if user_idx in G.nodes():
        pos[user_idx] = np.array([0, 0])  # 중앙에 배치
        
        # 다른 노드들의 위치 조정 (선택된 사용자를 중심으로)
        for node in G.nodes():
            if node != user_idx:
                # 현재 위치에서 중앙 방향으로 약간 이동
                current_pos = pos[node]
                # 중앙과의 거리 계산
                distance = np.linalg.norm(current_pos)
                # 거리가 너무 가까우면 조정
                if distance < 0.2:
                    # 방향은 유지하되 거리를 늘림
                    direction = current_pos / (distance + 1e-10)  # 0으로 나누기 방지
                    pos[node] = direction * 0.3
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=figsize)
    
    # 유사도 범위에 따라 다른 스타일로 엣지 그룹화
    high_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 0.9]
    medium_edges = [(u, v) for u, v, d in G.edges(data=True) if 0.8 <= d['weight'] < 0.9]
    low_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0.8]
    
    # 높은 유사도 엣지
    nx.draw_networkx_edges(
        G, pos,
        edgelist=high_edges,
        width=2.5,
        alpha=0.8,
        edge_color='darkblue',
        ax=ax
    )
    
    # 중간 유사도 엣지
    nx.draw_networkx_edges(
        G, pos,
        edgelist=medium_edges,
        width=1.5,
        alpha=0.6,
        edge_color='royalblue',
        ax=ax
    )
    
    # 낮은 유사도 엣지
    nx.draw_networkx_edges(
        G, pos,
        edgelist=low_edges,
        width=1.0,
        alpha=0.4,
        edge_color='lightblue',
        ax=ax
    )
    
    # 노드 크기 및 색상 설정
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        if node == user_idx:
            # 선택된 사용자는 크고 빨간색으로 표시
            node_sizes.append(800)
            node_colors.append('red')
        else:
            # 다른 노드들은 연결 정도에 따라 크기 조절
            degree = G.degree(node)
            node_sizes.append(degree * 80 + 200)
            # 선택된 사용자와 직접 연결된 노드는 다른 색상으로 표시
            if G.has_edge(user_idx, node):
                node_colors.append('orange')
            else:
                node_colors.append('skyblue')
    
    # 노드 그리기
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    # 레이블 그리기
    # 선택된 사용자와 연결된 노드만 레이블 표시
    labels = {}
    for node in G.nodes():
        if node == user_idx or (user_idx in G.nodes() and G.has_edge(user_idx, node)):
            labels[node] = G.nodes[node]['name']
    
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_family='sans-serif',
        ax=ax
    )
    
    # 범례 추가
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.8, label='선택된 사용자'),
        mpatches.Patch(color='orange', alpha=0.8, label='직접 연결된 사용자'),
        mpatches.Patch(color='skyblue', alpha=0.8, label='기타 사용자'),
        mpatches.Patch(color='darkblue', alpha=0.8, label='유사도 ≥ 0.9'),
        mpatches.Patch(color='royalblue', alpha=0.6, label='0.8 ≤ 유사도 < 0.9'),
        mpatches.Patch(color='lightblue', alpha=0.4, label='유사도 < 0.8')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', title='범례')
    
    title_text = f"사용자 중심 네트워크 (임계값: {threshold}, 최소 연결 수: {min_connections})"
    if feature_name:
        title_text = f"{feature_name} 기반 " + title_text
    
    plt.title(title_text, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # 저장
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 캐시에 저장
    if use_cache:
        ensure_dir(cache_dir)
        plt.savefig(cache_file, dpi=300, bbox_inches='tight')
    
    return fig

def plot_interactive_user_centered_network(recommender, user_idx, n_users=30, threshold=0.7, min_connections=2,
                                          figsize=(800, 800), save_path=None, cache_dir="visualization_cache",
                                          use_cache=True, feature_name=None):
    """
    선택된 사용자를 중심으로 한 인터랙티브 네트워크 그래프 시각화 (Plotly 사용)
    
    Args:
        recommender: 추천 시스템 인스턴스
        user_idx (int): 중심에 위치할 사용자 인덱스
        n_users (int): 시각화할 사용자 수
        threshold (float): 연결선을 표시할 유사도 임계값
        min_connections (int): 노드가 가져야 할 최소 연결 수
        figsize (tuple): 그림 크기 (픽셀)
        save_path (str, optional): 저장 경로
        cache_dir (str): 캐시 디렉토리
        use_cache (bool): 캐시 사용 여부
        feature_name (str, optional): 특정 특성 이름. None이면 전체 유사도 사용.
        
    Returns:
        plotly.graph_objects.Figure: 생성된 그림 객체
    """
    # 캐시 파일 경로
    feature_suffix = f"_{feature_name}" if feature_name else ""
    cache_file = os.path.join(cache_dir, f"interactive_user_centered_network_u{user_idx}_n{n_users}_t{threshold}_min{min_connections}{feature_suffix}.html")
    
    # 그래프 생성
    if feature_name is not None and feature_name in recommender.feature_similarity_matrices:
        G, edge_data = create_similarity_graph(recommender, n_users, threshold, feature_name)
    else:
        G, edge_data = create_similarity_graph(recommender, n_users, threshold)
    
    # 그래프 필터링
    G = filter_graph(G, edge_data, n_users, max_edges_factor=3, min_connections=min_connections)
    
    # 선택된 사용자가 그래프에 없는 경우 추가
    if user_idx not in G.nodes():
        profile = recommender.profiles[user_idx]
        G.add_node(user_idx,
                  name=profile.get('name', f'User {user_idx}'),
                  job=profile.get('job_title', ''),
                  specialties=profile.get('specialties', ''),
                  skills=profile.get('skills', ''))
        
        # 선택된 사용자와 다른 노드 간의 엣지 추가
        for node in list(G.nodes()):
            if node != user_idx:
                if feature_name is not None and feature_name in recommender.feature_similarity_matrices:
                    similarity_matrix = recommender.feature_similarity_matrices[feature_name]
                    if user_idx < similarity_matrix.shape[0] and node < similarity_matrix.shape[1]:
                        similarity = similarity_matrix[user_idx, node]
                        if similarity >= threshold:
                            G.add_edge(user_idx, node, weight=similarity)
                else:
                    # total_score 유사도 행렬 사용
                    if 'total_score' in recommender.feature_similarity_matrices:
                        similarity_matrix = recommender.feature_similarity_matrices['total_score']
                        if user_idx < similarity_matrix.shape[0] and node < similarity_matrix.shape[1]:
                            similarity = similarity_matrix[user_idx, node]
                            if similarity >= threshold:
                                G.add_edge(user_idx, node, weight=similarity)
                    else:
                        # total_score가 없는 경우 첫 번째 사용 가능한 특성 사용
                        if recommender.feature_similarity_matrices:
                            first_feature = list(recommender.feature_similarity_matrices.keys())[0]
                            similarity_matrix = recommender.feature_similarity_matrices[first_feature]
                            if user_idx < similarity_matrix.shape[0] and node < similarity_matrix.shape[1]:
                                similarity = similarity_matrix[user_idx, node]
                                if similarity >= threshold:
                                    G.add_edge(user_idx, node, weight=similarity)
    
    # 그래프가 비어있는 경우 처리
    if len(G.nodes()) == 0:
        print(f"경고: 임계값 {threshold}와 최소 연결 수 {min_connections}로 인해 표시할 노드가 없습니다.")
        print("임계값을 낮추거나 최소 연결 수를 줄여보세요.")
        # 빈 그래프 반환
        fig = go.Figure()
        fig.update_layout(
            title="표시할 노드가 없습니다. 임계값을 낮추거나 최소 연결 수를 줄여보세요.",
            width=figsize[0],
            height=figsize[1]
        )
        return fig
    
    # 선택된 사용자를 중심으로 레이아웃 계산
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # 선택된 사용자를 중앙에 배치
    if user_idx in G.nodes():
        pos[user_idx] = np.array([0, 0])  # 중앙에 배치
        
        # 다른 노드들의 위치 조정 (선택된 사용자를 중심으로)
        for node in G.nodes():
            if node != user_idx:
                # 현재 위치에서 중앙 방향으로 약간 이동
                current_pos = pos[node]
                # 중앙과의 거리 계산
                distance = np.linalg.norm(current_pos)
                # 거리가 너무 가까우면 조정
                if distance < 0.2:
                    # 방향은 유지하되 거리를 늘림
                    direction = current_pos / (distance + 1e-10)  # 0으로 나누기 방지
                    pos[node] = direction * 0.3
    
    # 노드 데이터 준비
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_group = []  # 노드 그룹 (선택된 사용자, 직접 연결된 사용자, 기타 사용자)
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # 노드 텍스트 정보
        name = G.nodes[node]['name']
        job = G.nodes[node]['job']
        specialties = G.nodes[node]['specialties']
        degree = G.degree(node)
        
        # 선택된 사용자와 높은 강도로 연결된 상위 N명의 노드 식별
        top_n = 10  # 상위 N개 노드 (필요에 따라 조정 가능)
        top_nodes = [user_idx]  # 선택된 사용자는 항상 포함
        
        # 선택된 사용자와 연결된 노드들의 연결 강도(weight) 확인
        if user_idx in G.nodes():
            connected_nodes = []
            for neighbor in G.neighbors(user_idx):
                if G.has_edge(user_idx, neighbor):
                    weight = G[user_idx][neighbor]['weight']
                    connected_nodes.append((neighbor, weight))
            
            # 연결 강도(weight)에 따라 정렬하고 상위 N개 선택
            sorted_connected = sorted(connected_nodes, key=lambda x: x[1], reverse=True)
            top_connected = [node for node, _ in sorted_connected[:top_n-1]]  # 선택된 사용자를 제외한 상위 N-1개
            top_nodes.extend(top_connected)
            
        # 더 자세한 호버 텍스트 (모든 노드에 표시)
        feature_text = f"{feature_name} " if feature_name else ""
        hover_info = (f"<b>{name}</b><br>"
                     f"직업: {job}<br>"
                     f"전문분야: {specialties}<br>"
                     f"연결 수: {degree}<br>"
                     f"평균 {feature_text}유사도: {sum(G[node][neighbor]['weight'] for neighbor in G[node]) / max(1, degree):.4f}")
        
        # 상위 노드와 중심 사용자만 이름 표시, 나머지는 빈 문자열
        if node in top_nodes:
            if node == user_idx:  # 중심 사용자는 강조 표시
                node_text.append(f"<b>{name}</b>")
            else:
                node_text.append(name)
        else:
            node_text.append("")
        
        # 노드 그룹 및 크기 설정
        if node == user_idx:
            # 선택된 사용자
            node_group.append("선택된 사용자")
            node_size.append(30)  # 더 큰 크기
            node_color.append("red")
        elif user_idx in G.nodes() and G.has_edge(user_idx, node):
            # 선택된 사용자와 직접 연결된 노드
            node_group.append("직접 연결된 사용자")
            # 유사도에 따라 크기 조절
            similarity = G[user_idx][node]['weight']
            node_size.append(20 + similarity * 10)
            node_color.append("orange")
        else:
            # 기타 노드
            node_group.append("기타 사용자")
            node_size.append(10 + degree * 2)
            node_color.append("skyblue")
    
    # 엣지 데이터 준비
    edge_x = []
    edge_y = []
    edge_text = []
    edge_color = []
    edge_width = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # 엣지 좌표 (선 그리기)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # 선 분리를 위한 None
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)  # 선 분리를 위한 None
        
        # 엣지 호버 텍스트
        weight = G.edges[edge]['weight']
        user1 = G.nodes[edge[0]]['name']
        user2 = G.nodes[edge[1]]['name']
        feature_text = f"{feature_name} " if feature_name else ""
        edge_text.append(f"{user1} - {user2}<br>{feature_text}유사도: {weight:.4f}")
        edge_text.append(f"{user1} - {user2}<br>{feature_text}유사도: {weight:.4f}")
        edge_text.append(None)
        
        # 엣지 색상 및 두께 (유사도에 따라)
        if weight >= 0.9:
            color = "rgba(50, 50, 200, 0.8)"
            width = 3
            group_name = "유사도: 매우 높음 (≥ 0.9)"
        elif weight >= 0.8:
            color = "rgba(100, 100, 200, 0.6)"
            width = 2
            group_name = "유사도: 높음 (≥ 0.8)"
        else:
            color = "rgba(150, 150, 200, 0.4)"
            width = 1
            group_name = "유사도: 중간/낮음 (< 0.8)"
        
        # 선택된 사용자와 연결된 엣지는 강조
        if edge[0] == user_idx or edge[1] == user_idx:
            color = "rgba(255, 100, 100, 0.8)"
            width += 1
            group_name = "선택된 사용자와의 연결"
        
        edge_color.extend([color, color, None])
        edge_width.extend([width, width, None])
    
    # 엣지 트레이스 (색상별로 분리)
    edge_traces = []
    
    # 색상별로 엣지 그룹화 - 유사도 범위에 따라 구분
    edge_groups = {}
    edge_group_names = {
        "rgba(50, 50, 200, 0.8)": "유사도: 매우 높음 (≥ 0.9)",
        "rgba(100, 100, 200, 0.6)": "유사도: 높음 (≥ 0.8)",
        "rgba(150, 150, 200, 0.4)": "유사도: 중간/낮음 (< 0.8)",
        "rgba(255, 100, 100, 0.8)": "선택된 사용자와의 연결"
    }
    
    for i in range(0, len(edge_x), 3):
        if i+2 < len(edge_color) and edge_color[i] is not None:
            color = edge_color[i]
            width = edge_width[i]
            
            if color not in edge_groups:
                edge_groups[color] = {
                    'x': [], 'y': [], 'text': [], 'width': width,
                    'name': edge_group_names.get(color, f'연결')
                }
            
            # 각 엣지는 3개의 점으로 구성 (시작, 끝, None)
            edge_groups[color]['x'].extend([edge_x[i], edge_x[i+1], edge_x[i+2]])
            edge_groups[color]['y'].extend([edge_y[i], edge_y[i+1], edge_y[i+2]])
            
            if i < len(edge_text):
                edge_groups[color]['text'].extend([edge_text[i], edge_text[i+1], edge_text[i+2]])
    
    # 각 색상 그룹별로 트레이스 생성
    for color, data in edge_groups.items():
        edge_trace = go.Scatter(
            x=data['x'], y=data['y'],
            mode='lines',
            line=dict(color=color, width=data['width']),
            hoverinfo='text',
            text=data['text'],
            name=data['name']
        )
        edge_traces.append(edge_trace)
    
    # 노드 트레이스 (그룹별로 분리)
    node_traces = []
    
    # 그룹별로 노드 분리
    groups = ["선택된 사용자", "직접 연결된 사용자", "기타 사용자"]
    colors = ["red", "orange", "skyblue"]
    
    for group, color in zip(groups, colors):
        indices = [i for i, g in enumerate(node_group) if g == group]
        
        if indices:
            node_trace = go.Scatter(
                x=[node_x[i] for i in indices],
                y=[node_y[i] for i in indices],
                mode='markers+text',
                marker=dict(
                    size=[node_size[i] for i in indices],
                    color=color,
                    line=dict(width=2, color='white')
                ),
                hoverinfo='text',
                text=[node_text[i] for i in indices],
                textposition="top center",  # 텍스트 위치를 노드 위쪽 중앙으로 설정
                name=group
            )
            node_traces.append(node_trace)
    
    # 그래프 레이아웃
    title_text = f'사용자 중심 네트워크 (임계값: {threshold}, 최소 연결 수: {min_connections})'
    if feature_name:
        title_text = f'{feature_name} 기반 ' + title_text
    
    layout = go.Layout(
        title=dict(
            text=title_text,
            font=dict(size=16)
        ),
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            ),
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1
        )
    )
    
    # 그래프 생성
    fig = go.Figure(data=edge_traces + node_traces, layout=layout)
    
    # 저장
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.write_html(save_path)
    
    # 캐시에 저장
    if use_cache:
        ensure_dir(cache_dir)
        fig.write_html(cache_file)
    
    return fig

def plot_feature_specific_network(recommender, feature_name, n_users=30, threshold=0.7, min_connections=2,
                                  figsize=(800, 800), save_path=None, cache_dir="visualization_cache",
                                  use_cache=True, interactive=True):
    """
    특정 특성(차원)의 유사도 점수만을 기반으로 한 네트워크 시각화
    
    Args:
        recommender: 추천 시스템 인스턴스
        feature_name (str): 시각화할 특성 이름 (예: "최근 대화", "관심사", "성격" 등)
        n_users (int): 시각화할 사용자 수
        threshold (float): 연결선을 표시할 유사도 임계값
        min_connections (int): 노드가 가져야 할 최소 연결 수
        figsize (tuple): 그림 크기
        save_path (str, optional): 저장 경로
        cache_dir (str): 캐시 디렉토리
        use_cache (bool): 캐시 사용 여부
        interactive (bool): 인터랙티브 시각화 여부
        
    Returns:
        plotly.graph_objects.Figure 또는 matplotlib.figure.Figure: 생성된 그림 객체
    """
    # 특성 유사도 행렬이 비어있는 경우 생성
    if not hasattr(recommender, 'feature_similarity_matrices') or not recommender.feature_similarity_matrices:
        print("특성별 유사도 행렬이 없습니다. 계산을 시작합니다...")
        recommender.compute_feature_similarity_matrices()
    
    # 특성 이름이 유효한지 확인
    if feature_name not in recommender.feature_similarity_matrices:
        available_features = list(recommender.feature_similarity_matrices.keys())
        print(f"오류: '{feature_name}' 특성을 찾을 수 없습니다. 사용 가능한 특성: {available_features}")
        
        if not available_features:
            print("사용 가능한 특성이 없습니다. 전체 유사도 기반 네트워크를 대신 시각화합니다.")
            if interactive:
                return plot_interactive_force_directed_graph(
                    recommender=recommender,
                    n_users=n_users,
                    threshold=threshold,
                    min_connections=min_connections,
                    figsize=figsize,
                    save_path=save_path,
                    cache_dir=cache_dir,
                    use_cache=use_cache
                )
            else:
                return plot_force_directed_graph(
                    recommender=recommender,
                    n_users=n_users,
                    threshold=threshold,
                    min_connections=min_connections,
                    figsize=figsize,
                    save_path=save_path,
                    cache_dir=cache_dir,
                    use_cache=use_cache
                )
        return None
    
    # 인터랙티브 또는 정적 시각화 선택
    if interactive:
        return plot_interactive_force_directed_graph(
            recommender=recommender,
            n_users=n_users,
            threshold=threshold,
            min_connections=min_connections,
            figsize=figsize,
            save_path=save_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            feature_name=feature_name
        )
    else:
        return plot_force_directed_graph(
            recommender=recommender,
            n_users=n_users,
            threshold=threshold,
            min_connections=min_connections,
            figsize=figsize,
            save_path=save_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            feature_name=feature_name
        )