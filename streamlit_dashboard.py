#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Magnet 밋업 추천 시각화 Streamlit 대시보드

이 스크립트는 밋업 매칭 추천 시스템의 결과를 Streamlit을 사용하여 시각화합니다.
"""

import os
import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from meetup_recommendation import MeetupRecommendationSystem
from visualization.utils import create_similarity_graph, filter_graph, create_weighted_similarity_graph
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
import json

# 페이지 설정
st.set_page_config(
    page_title="밋업 추천 시각화 대시보드",
    page_icon="🔍",
    layout="wide"
)

# 제목
st.title("밋업 추천 시스템 시각화 대시보드")

# 사이드바 설정
st.sidebar.header("설정")

# 캐시 강제 초기화 여부 확인
force_cache_reset = st.sidebar.checkbox("캐시 강제 초기화", False, key="force_cache_reset")

# 추천 시스템 초기화
@st.cache_resource
def load_recommender(force_reset=False):
    st.sidebar.text("추천 시스템을 초기화하는 중...")
    recommender = MeetupRecommendationSystem(profiles_path="data/profiles/generated_virtual_users_v1.json")
    
    # 캐시 강제 초기화가 선택되었거나 유사도 행렬이 비어있는 경우
    if force_reset:
        st.sidebar.warning("캐시를 강제로 초기화합니다...")
        recommender.run_recommendation_pipeline(overwrite_cache=True)
        st.sidebar.success("유사도 행렬 계산이 완료되었습니다.")
        return recommender
    
    # 캐시된 유사도 행렬 로드
    similarity_cache_metadata = {}
    if os.path.exists(recommender.similarity_metadata_path):
        try:
            with open(recommender.similarity_metadata_path, 'r') as f:
                similarity_cache_metadata = json.load(f)
                
            # 각 특성별 유사도 행렬 로드
            for feature, metadata in similarity_cache_metadata.items():
                similarity_cache_file = metadata.get('file_path')
                if os.path.exists(similarity_cache_file):
                    with open(similarity_cache_file, 'r') as f:
                        similarity_data = json.load(f)
                        recommender.feature_similarity_matrices[feature] = np.array(similarity_data["matrix"])
            
            st.sidebar.success(f"{len(similarity_cache_metadata)}개의 캐시된 유사도 행렬을 로드했습니다.")
        except Exception as e:
            st.sidebar.error(f"유사도 캐시 로드 중 오류 발생: {e}")
    
    # 캐시된 유사도 행렬이 없으면 계산
    if not recommender.feature_similarity_matrices:
        st.sidebar.warning("캐시된 유사도 행렬이 없습니다. 유사도 행렬을 계산합니다...")
        recommender.run_recommendation_pipeline(overwrite_cache=False)
        st.sidebar.success("유사도 행렬 계산이 완료되었습니다.")
    
    return recommender

# 캐시 강제 초기화 여부에 따라 recommender 로드
recommender = load_recommender(force_cache_reset)

# 이미 위에서 recommender = load_recommender(force_cache_reset)로 수정했으므로 이 줄은 삭제

# 특성 선택 (모든 탭에서 공통으로 사용)
available_features = list(recommender.feature_similarity_matrices.keys())

# 'total_score'가 available_features에 없으면 추가
if 'total_score' not in available_features:
    available_features.append('total_score')

# 세션 상태에 특성 저장 (페이지 새로고침 시에도 유지)
if 'selected_feature' not in st.session_state:
    st.session_state.selected_feature = 'total_score'

# 특성 선택 라디오 버튼
selected_feature = st.sidebar.radio(
    "분석할 특성 선택",
    available_features,
    index=available_features.index(st.session_state.selected_feature),
    key="feature_selector"
)

# 사용자 이름 목록 생성
user_names = [profile.get('name', f'User {i}') for i, profile in enumerate(recommender.profiles)]
user_name_to_idx = {name: idx for idx, name in enumerate(user_names)}

# 세션 상태에 선택된 사용자 저장
if 'selected_user_name' not in st.session_state:
    st.session_state.selected_user_name = user_names[0]

# 드롭다운으로 사용자 선택
selected_user_name = st.sidebar.selectbox(
    "기준 사용자 선택",
    user_names,
    index=user_names.index(st.session_state.selected_user_name)
)

# 세션 상태 업데이트
st.session_state.selected_user_name = selected_user_name

# 선택된 사용자의 인덱스 가져오기
user_idx = user_name_to_idx[selected_user_name]

# 사이드바 파라미터
n_users = st.sidebar.slider("시각화할 사용자 수", 10, 100, 50)
threshold = st.sidebar.slider("유사도 임계값", 0.0, 1.0, 0.7, 0.05)
min_connections = st.sidebar.slider("최소 연결 수", 1, 10, 1)

top_n = st.sidebar.slider("추천 사용자 수", 1, 20, 5)

# 사이드바에 가중치 설정 추가
st.sidebar.header("특성별 가중치 설정")
weight_specialties = st.sidebar.slider("Specialties 가중치", 0.0, 1.0, 0.3, 0.05)
weight_skills = st.sidebar.slider("Skills 가중치", 0.0, 1.0, 0.3, 0.05)
weight_job_title = st.sidebar.slider("Job Title 가중치", 0.0, 1.0, 0.2, 0.05)
weight_work_culture = st.sidebar.slider("Work Culture 가중치", 0.0, 1.0, 0.1, 0.05)
weight_interests = st.sidebar.slider("Interests 가중치", 0.0, 1.0, 0.05, 0.05)
weight_recent_conversation = st.sidebar.slider("Recent Conversation 가중치", 0.0, 1.0, 0.05, 0.05)

# 가중치 딕셔너리 생성 (영어 이름 사용)
feature_weights = {
    'specialties': weight_specialties,
    'skills': weight_skills,
    'job_title': weight_job_title,
    'work_culture': weight_work_culture,
    'interests': weight_interests,
    'recent_conversation_history': weight_recent_conversation
}

# 가중치 적용 여부
apply_weights = st.sidebar.checkbox("가중치 적용", True)

# 세션 상태 업데이트
st.session_state.selected_feature = selected_feature

# 선택된 특성에 대한 정보 표시
st.sidebar.info(f"현재 선택된 특성: **{selected_feature}**")

# 선택된 특성의 유사도 행렬 통계
if selected_feature in recommender.feature_similarity_matrices:
    sim_matrix = recommender.feature_similarity_matrices[selected_feature]

# 캐시 초기화 옵션
if st.sidebar.button("캐시 초기화", key="clear_cache_button"):
    import shutil
    if os.path.exists("visualization_cache"):
        shutil.rmtree("visualization_cache")
        st.sidebar.success("시각화 캐시가 초기화되었습니다.")
    if os.path.exists("cached_data/embeddings"):
        if st.sidebar.checkbox("임베딩 캐시도 초기화", False, key="clear_embeddings_cache"):
            shutil.rmtree("cached_data/embeddings")
            st.sidebar.success("임베딩 캐시가 초기화되었습니다.")
    st.rerun()

# 유사도 행렬 직접 생성 옵션
if st.sidebar.button("유사도 행렬 직접 생성", key="generate_similarity_matrices"):
    st.sidebar.info("유사도 행렬을 직접 생성합니다...")
    
    # 유사도 범위 선택 (기본값을 0.5-1.0으로 설정)
    sim_range = st.sidebar.slider("유사도 범위", 0.0, 1.0, (0.5, 1.0), 0.1, key="sim_range")
    
    # 특성별 유사도 행렬 생성 (영어 이름 사용)
    n_users = len(recommender.profiles)
    for feature in ['specialties', 'skills', 'job_title', 'work_culture', 'interests', 'recent_conversation_history']:
        # 랜덤 유사도 행렬 생성 (사용자 지정 범위)
        similarity_matrix = np.random.random((n_users, n_users)) * (sim_range[1] - sim_range[0]) + sim_range[0]
        np.fill_diagonal(similarity_matrix, 1.0)  # 자기 자신과의 유사도는 1
        
        # 대칭 행렬로 만들기
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        
        # 저장
        recommender.feature_similarity_matrices[feature] = similarity_matrix
        
        # JSON으로도 저장
        json_file = os.path.join(recommender.embeddings_dir, f"{feature}_similarity.json")
        similarity_dict = {
            "feature": feature,
            "shape": similarity_matrix.shape,
            "mean": float(np.mean(similarity_matrix)),
            "median": float(np.median(similarity_matrix)),
            "min": float(np.min(similarity_matrix)),
            "max": float(np.max(similarity_matrix)),
            "matrix": similarity_matrix.tolist()
        }
        with open(json_file, 'w') as f:
            json.dump(similarity_dict, f, indent=2)
    
    st.sidebar.success("유사도 행렬이 생성되었습니다.")
    st.rerun()

# 탭 관리
tab_names = ["추천 및 네트워크", "유사도 행렬"]

# 세션 상태에 활성 탭 저장
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# 탭 생성
tab1, tab2 = st.tabs(tab_names)

# 현재 활성 탭에 따라 해당 탭 선택
tabs = [tab1, tab2]
active_tab = tabs[st.session_state.active_tab]

# 탭 1: 추천 결과 및 네트워크 그래프 (화면 분할)
with tab1:
    # 선택된 특성 표시
    st.info(f"현재 선택된 특성: **{selected_feature}**")

    # 화면을 두 열로 분할
    left_col, right_col = st.columns([1, 1])
    
    # 왼쪽 열: 추천 결과
    with left_col:
        # 선택된 사용자 프로필 정보 표시
        st.subheader(f"사용자 프로필")
        profile = recommender.profiles[user_idx]
        # 프로필 정보를 데이터프레임으로 표시
        profile_data = {
            '이름': [profile.get('name', f'User {user_idx}')],
            '직업': [profile.get('job_title', '정보 없음')],
            '전문 분야': [profile.get('specialties', '정보 없음')],
            '기술 스택': [profile.get('skills', '정보 없음')],
            '관심사': [profile.get('interests', '정보 없음')],
            '업무 문화': [profile.get('work_culture', '정보 없음')]
        }
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, hide_index=True)        
        st.subheader(f"추천 결과")
        
        # 선택된 특성에 따라 추천 생성
        if selected_feature == 'total_score':
            # total_score 기반 추천
            recommendations = recommender.generate_recommendations(user_idx, top_n)
            st.caption("total_score를 기반으로 추천 결과를 생성했습니다.")
        else:
            # 특정 특성 기반 추천
            # 특성별 유사도 행렬 가져오기
            if selected_feature in recommender.feature_similarity_matrices:
                sim_matrix = recommender.feature_similarity_matrices[selected_feature]
                
                # 특정 특성 기반 추천 생성
                user_similarities = sim_matrix[user_idx]
                # 자기 자신 제외
                user_similarities[user_idx] = -1
                # 상위 N개 유사한 사용자 찾기
                top_similar_indices = np.argsort(user_similarities)[::-1][:top_n]
                recommendations = [(idx, user_similarities[idx]) for idx in top_similar_indices if user_similarities[idx] > 0]
                
                st.caption(f"**{selected_feature}** 특성만을 기반으로 추천 결과를 생성했습니다.")
            else:
                st.error(f"선택한 특성 '{selected_feature}'에 대한 유사도 행렬이 없습니다.")
                recommendations = []
        
        # 추천 결과를 데이터프레임으로 변환
        recommendation_data = []
        
        for rec_idx, score in recommendations:
            rec_profile = recommender.profiles[rec_idx]
            
            # 기본 정보 딕셔너리 생성
            rec_data = {
                '이름': rec_profile.get('name', f'User {rec_idx}'),
                '유사도 점수': score,
                '직업': rec_profile.get('job_title', ''),
                '전문 분야': rec_profile.get('specialties', ''),
                '기술 스택': rec_profile.get('skills', ''),
                '관심사': rec_profile.get('interests', ''),
                '업무 문화': rec_profile.get('work_culture', ''),

            }
            
            recommendation_data.append(rec_data)
        
        # 데이터프레임 생성
        recommendation_df = pd.DataFrame(recommendation_data)
        
        if recommendation_df.empty:
            st.warning("추천 결과가 없습니다. 임계값을 낮추거나 다른 특성을 선택해보세요.")
        pd.set_option('display.float_format', '{:.4f}'.format)
        st.dataframe(recommendation_df, height=400)
    
    # 오른쪽 열: 네트워크 그래프
        with right_col:
            st.subheader("사용자 중심 네트워크 그래프")
            
            # 시각화 방식 선택
            viz_type = st.radio(
                "시각화 방식",
                ["일반", "인터랙티브"],
                horizontal=True,
                key="viz_type_tab1"
            )
            
            # 사용자 중심 네트워크 그래프 시각화
            from visualization.network_graphs import plot_user_centered_network, plot_interactive_user_centered_network
            
            if viz_type == "인터랙티브":
                # 인터랙티브 사용자 중심 네트워크 그래프
                if selected_feature == 'total_score':
                    st.caption("total_score를 기반으로 네트워크 그래프를 생성합니다.")
                    fig = plot_interactive_user_centered_network(
                        recommender=recommender,
                        user_idx=user_idx,
                        n_users=n_users,
                        threshold=threshold,
                        min_connections=min_connections,
                        figsize=(700, 500)
                    )
                else:
                    st.caption(f"**{selected_feature}** 특성만을 기반으로 네트워크 그래프를 생성합니다.")
                    fig = plot_interactive_user_centered_network(
                        recommender=recommender,
                        user_idx=user_idx,
                        n_users=n_users,
                        threshold=threshold,
                        min_connections=min_connections,
                        figsize=(700, 500),
                        feature_name=selected_feature
                    )
                
                # Plotly 그래프 표시
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 일반 사용자 중심 네트워크 그래프
                if selected_feature == 'total_score':
                    st.caption("total_score를 기반으로 네트워크 그래프를 생성합니다.")
                    fig = plot_user_centered_network(
                        recommender=recommender,
                        user_idx=user_idx,
                        n_users=n_users,
                        threshold=threshold,
                        min_connections=min_connections,
                        figsize=(10, 8)
                    )
                else:
                    st.caption(f"**{selected_feature}** 특성만을 기반으로 네트워크 그래프를 생성합니다.")
                    fig = plot_user_centered_network(
                        recommender=recommender,
                        user_idx=user_idx,
                        n_users=n_users,
                        threshold=threshold,
                        min_connections=min_connections,
                        figsize=(10, 8),
                        feature_name=selected_feature
                    )
                
                # Matplotlib 그래프 표시
                st.pyplot(fig)
            
            # 그래프 통계 정보는 일반 모드에서만 표시
            if viz_type == "일반":
                # 그래프 통계 정보
                st.subheader("네트워크 통계")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.info("사용자 중심 네트워크 그래프가 생성되었습니다.")
                with stats_col2:
                    st.info(f"선택된 사용자: {selected_user_name}")

# 탭 2: 유사도 행렬
with tab2:
    st.info(f"현재 선택된 특성: **{selected_feature}**")
    # 2열 레이아웃으로 설정
    col1, col2 = st.columns([1, 3])
    
    st.subheader("시각화 설정")        
    feature_for_viz = selected_feature
    # 샘플 크기 선택
    sample_size = st.slider("샘플 크기", 5, 50, 20, key="sample_size_slider")        
    # 히트맵 색상 선택
    colormap = 'coolwarm'

    # 선택한 특성의 유사도 행렬 가져오기
    if feature_for_viz in recommender.feature_similarity_matrices:
        sim_matrix = recommender.feature_similarity_matrices[feature_for_viz]
        
        # 행렬이 너무 크면 일부만 시각화
        sample_size = min(sample_size, sim_matrix.shape[0])
        sample_matrix = sim_matrix[:sample_size, :sample_size]           
        
        # 히트맵
        st.subheader(f"유사도 행렬")
        
        # 히트맵 생성
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
        
        # 값이 모두 0이면 경고 메시지 표시
        if np.all(sample_matrix == 0):
            st.warning("유사도 행렬의 모든 값이 0입니다. '유사도 행렬 직접 생성' 옵션을 사용하여 새로운 유사도 행렬을 생성해보세요.")
            # 빈 히트맵 대신 텍스트 표시
            ax_heatmap.text(0.5, 0.5, "유사도 행렬의 모든 값이 0입니다.",
                            ha='center', va='center', fontsize=14)
            ax_heatmap.axis('off')
        else:
            # 정상적인 히트맵 표시 (선택한 색상 테마 적용)
            im = ax_heatmap.imshow(sample_matrix, cmap=colormap, vmin=0, vmax=1)
            fig_heatmap.colorbar(im, ax=ax_heatmap, label='유사도')
            
            # 사용자 이름 표시
            if sample_size <= 20:  # 너무 많은 레이블은 혼잡해 보일 수 있음
                user_names = [recommender.profiles[i].get('name', f'User {i}') for i in range(sample_size)]
                ax_heatmap.set_xticks(np.arange(sample_size))
                ax_heatmap.set_yticks(np.arange(sample_size))
                ax_heatmap.set_xticklabels(user_names, rotation=90)
                ax_heatmap.set_yticklabels(user_names)
                
                # 히트맵 위에 유사도 값 표시
                for i in range(sample_size):
                    for j in range(sample_size):
                        ax_heatmap.text(j, i, f"{sample_matrix[i, j]:.2f}",
                                        ha="center", va="center", color='black', fontsize=8)
        # 히트맵 표시
        st.pyplot(fig_heatmap)
    else:
        st.warning(f"'{feature_for_viz}' 특성에 대한 유사도 행렬이 없습니다. '유사도 행렬 직접 생성' 옵션을 사용하여 유사도 행렬을 생성해보세요.")

# 푸터
st.markdown("---")
st.markdown("© 2025 Potential Labs | Magnet 밋업 추천 시스템 시각화 대시보드 | yoonjw0@gmail.com")