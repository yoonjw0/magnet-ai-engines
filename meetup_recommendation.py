#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
밋업 매칭 추천 시스템

이 모듈은 가상 유저 프로필을 기반으로 밋업 매칭을 추천하는 시스템을 구현합니다.
하이브리드 접근법을 사용하여 콘텐츠 기반 유사도와 피드백 기반 예측을 결합합니다.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai._common import GoogleGenerativeAIError

import random
from tqdm import tqdm
import time
from typing import Dict

from dotenv import load_dotenv
load_dotenv()


class MeetupRecommendationSystem:
    """밋업 매칭 추천 시스템 클래스"""
    
    def __init__(self, profiles_path="data/profiles/generated_virtual_users.json", cache_dir="cached_data", overwrite_cache=False):
        """
        초기화 함수
        
        Args:
            profiles_path (str): 가상 유저 프로필 파일 경로
            cache_dir (str): 캐시 데이터를 저장할 디렉토리 경로
            overwrite_cache (bool): 기존 캐시를 덮어쓸지 여부 (기본값: False)
        """
        self.profiles_path = profiles_path
        self.cache_dir = cache_dir
        
        # 캐시 디렉토리 구조 설정
        self.embedding_cache_dir = os.path.join(cache_dir, "embedding_cache")
        self.embedding_metadata_path = os.path.join(self.embedding_cache_dir, "metadata.json")
        
        self.similarity_cache_dir = os.path.join(cache_dir, "similarity_cache")
        self.similarity_metadata_path = os.path.join(self.similarity_cache_dir, "metadata.json")
        
        self.overwrite_cache = overwrite_cache
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        os.makedirs(self.similarity_cache_dir, exist_ok=True)
        
        self.profiles = self._load_profiles()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=1.0,  # 최대 온도 사용 (Gemini API는 0.0-1.0 범위만 허용)
            api_key=os.getenv('GOOGLE_API_KEY'),
        )
        
        # 특성별 가중치 초기화 (6가지 특성)
        self.feature_weights = {
            'specialties': 0.5,
            'skills': 0.5,
            'job_title': 0.5,
            'work_culture': 0.5,
            'interests': 0.5,
            'recent_conversation_history': 0.5
        }
        
        # 임베딩 결과를 저장할 디렉토리 설정 (기존 호환성 유지)
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # 피드백 데이터 초기화 (실제로는 DB에서 로드)
        self.feedback_matrix = self._initialize_feedback_matrix()
        
        # 특성별 임베딩 및 유사도 행렬
        self.feature_embeddings = {}
        self.feature_similarity_matrices = {}
    
    def _load_profiles(self):
        """프로필 데이터 로드"""
        try:
            with open(self.profiles_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            print(f"{len(profiles)}개의 프로필을 로드했습니다.")
            return profiles
        except Exception as e:
            print(f"프로필 로드 중 오류 발생: {e}")
            return []
    
    def _initialize_feedback_matrix(self):
        """피드백 행렬 초기화 (실제로는 DB에서 로드)"""
        n_users = len(self.profiles)
        # 희소 행렬로 초기화 (모든 값은 NaN)
        feedback_matrix = np.full((n_users, n_users), np.nan)
        
        # 초기에는 피드백 없이 시작
        # 실제 시스템에서는 사용자 피드백이 수집되면 이 행렬을 업데이트
        print("빈 피드백 행렬로 초기화했습니다. 콘텐츠 기반 추천만 사용됩니다.")
        
        return feedback_matrix
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if not text or not isinstance(text, str):
            return ""
        
        import re
        
        # 1. 소문자 변환
        text = text.lower()
        
        # 2. HTML 태그 제거
        text = re.sub(r'<.*?>', ' ', text)
        
        # 3. 특수 문자를 공백으로 대체 (단, 한글, 영문, 숫자, 일부 구두점은 유지)
        text = re.sub(r'[^\w\s\.\,\?\!\-\'\"\:\;가-힣]', ' ', text)
        
        # 4. 여러 공백을 하나의 공백으로 대체
        text = re.sub(r'\s+', ' ', text)
        
        # 5. 앞뒤 공백 제거
        text = text.strip()
        
        # 6. 중복 문장 제거 (동일한 문장이 반복되는 경우)
        sentences = text.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # 7. 중복 단어 제거 (연속으로 같은 단어가 반복되는 경우)
        words = ' '.join(unique_sentences).split()
        unique_words = []
        prev_word = None
        for word in words:
            if word != prev_word:
                unique_words.append(word)
                prev_word = word
        
        # 8. 최종 텍스트 생성
        final_text = ' '.join(unique_words)
        
        return final_text
    
    def _get_embedding_with_retry(self, text, max_retries=5, initial_wait=2):
        """임베딩을 얻는 함수 (재시도 로직 포함)"""
        wait_time = initial_wait
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_query(text)
            except GoogleGenerativeAIError as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 재시도
                        print(f"Rate limit exceeded, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        wait_time *= 2  # 대기 시간을 지수적으로 증가
                        continue
                raise  # 다른 종류의 에러이거나 최대 재시도 횟수를 초과한 경우
    
    def compute_embeddings(self):
        """각 특성별 임베딩 계산"""
        print("특성별 임베딩 계산 중...")
        
        # 임베딩 캐시 메타데이터 초기화 또는 로드
        embedding_cache_metadata = {}
        if os.path.exists(self.embedding_metadata_path) and not self.overwrite_cache:
            try:
                with open(self.embedding_metadata_path, 'r') as f:
                    embedding_cache_metadata = json.load(f)
            except Exception as e:
                print(f"임베딩 캐시 메타데이터 로드 중 오류 발생: {e}")
        
        # 각 특성별로 임베딩 계산
        feature_pbar = tqdm(self.feature_weights.keys(), desc="특성 처리 중")
        for feature in feature_pbar:
            feature_cache_file = os.path.join(self.embedding_cache_dir, f"{feature}_embeddings.json")
            feature_pbar.set_description(f"특성 처리 중: {feature}")
            
            # 이미 계산된 결과가 있는지 확인
            if not self.overwrite_cache and feature in embedding_cache_metadata and os.path.exists(feature_cache_file):
                feature_pbar.set_postfix_str("저장된 임베딩 로드 중...")
                with open(feature_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.feature_embeddings[feature] = cached_data["embeddings"]
                continue
            
            feature_pbar.set_postfix_str("임베딩 계산 중...")
            embeddings_list = []
            user_embeddings = []  # 사용자별 임베딩 저장
            batch_size = 5  # 배치 크기 설정
            
            # 프로필을 배치로 처리
            batch_pbar = tqdm(range(0, len(self.profiles), batch_size),
                               desc=f"{feature} 배치 처리",
                               leave=False)
            
            for i in batch_pbar:
                batch_profiles = self.profiles[i:i + batch_size]
                batch_embeddings = []
                current_batch = min(batch_size, len(self.profiles) - i)
                batch_pbar.set_postfix_str(f"현재 배치: {i//batch_size + 1}, 배치 크기: {current_batch}")
                
                # 각 프로필의 특성 텍스트에 대한 임베딩 계산
                for j, profile in enumerate(batch_profiles):
                    user_idx = i + j
                    # 특성 텍스트 가져오기 및 전처리
                    feature_text = profile.get(feature, "")
                    
                    # 텍스트 전처리 개선 (더 자세한 전처리)
                    if isinstance(feature_text, str):
                        # 특수 문자 제거, 소문자 변환, 여러 공백 제거
                        feature_text = self.preprocess_text(feature_text)
                        
                        # 텍스트가 너무 짧으면 임베딩 품질이 낮을 수 있음
                        if len(feature_text) < 5:  # 5글자 미만은 너무 짧다고 판단
                            print(f"경고: 프로필 {user_idx}의 '{feature}' 텍스트가 너무 짧습니다: '{feature_text}'")
                    
                    if feature_text:
                        try:
                            # 임베딩 계산
                            embedding = self._get_embedding_with_retry(feature_text)
                            
                            # 임베딩 품질 확인 (모든 값이 같은 경우 등)
                            if len(set(embedding[:10])) <= 2:  # 처음 10개 값 중 유니크한 값이 2개 이하면 의심
                                print(f"경고: 프로필 {user_idx}의 '{feature}' 임베딩이 의심스럽습니다.")
                                print(f"  텍스트: '{feature_text[:50]}...'")
                                print(f"  임베딩 샘플: {embedding[:5]}")
                            
                            batch_embeddings.append(embedding)
                            
                            # 사용자별 임베딩 저장
                            user_embeddings.append({
                                "user_id": profile.get("id", user_idx),
                                "user_name": profile.get("name", f"User {user_idx}"),
                                "embedding": embedding,
                                "text": feature_text[:100]  # 텍스트 샘플도 저장 (디버깅용)
                            })
                        except Exception as e:
                            print(f"임베딩 계산 중 치명적인 오류 발생: {e}")
                            raise  # 오류를 상위로 전파하여 프로세스 중단
                    else:
                        zero_embedding = [0] * 768  # 빈 텍스트는 0 벡터로 처리
                        batch_embeddings.append(zero_embedding)
                        
                        # 빈 텍스트에 대한 사용자별 임베딩 저장
                        user_embeddings.append({
                            "user_id": profile.get("id", user_idx),
                            "user_name": profile.get("name", f"User {user_idx}"),
                            "embedding": zero_embedding,
                            "text": ""
                        })
                
                embeddings_list.extend(batch_embeddings)
                time.sleep(1)  # API 요청 간 1초 대기
            
            feature_pbar.set_postfix_str("결과 저장 중...")
            
            # 계산된 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(embeddings_list)
            self.feature_embeddings[feature] = embeddings_array.tolist()
            
            # 임베딩 캐시에 저장
            cache_data = {
                "feature": feature,
                "embeddings": embeddings_array.tolist(),
                "user_embeddings": user_embeddings,
                "embedding_shape": embeddings_array.shape,
                "timestamp": time.time()
            }
            
            with open(feature_cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # 메타데이터 업데이트
            embedding_cache_metadata[feature] = {
                "feature": feature,
                "user_count": len(self.profiles),
                "embedding_dim": embeddings_array.shape[1] if embeddings_array.shape[0] > 0 else 0,
                "file_path": feature_cache_file,
                "timestamp": time.time()
            }
            
            # 기존 호환성을 위한 파일 저장
            feature_file = os.path.join(self.embeddings_dir, f"{feature}_embeddings.json")
            feature_metadata_file = os.path.join(self.embeddings_dir, f"{feature}_metadata.json")
            
            # 메타데이터 저장 (사용자 ID와 임베딩 인덱스 매핑)
            metadata = {
                "feature": feature,
                "user_ids": [profile.get("id", i) for i, profile in enumerate(self.profiles)],
                "user_names": [profile.get("name", f"User {i}") for i, profile in enumerate(self.profiles)],
                "embedding_shape": embeddings_array.shape,
                "timestamp": time.time()
            }
            
            with open(feature_metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            with open(feature_file, 'w') as f:
                json.dump({"embeddings": embeddings_array.tolist()}, f)
                
            print(f"{feature} 특성의 임베딩을 저장했습니다. 형태: {embeddings_array.shape}")
        
        # 전체 임베딩 캐시 메타데이터 저장
        with open(self.embedding_metadata_path, 'w') as f:
            json.dump(embedding_cache_metadata, f, indent=2)
        
        print(f"임베딩 캐시 메타데이터를 저장했습니다: {self.embedding_metadata_path}")
    
    def compute_feature_similarity_matrices(self):
        """각 특성(차원)별 유사도 행렬 계산"""
        self.feature_similarity_matrices = {}
        
        # 유사도 캐시 메타데이터 초기화 또는 로드
        similarity_cache_metadata = {}
        if os.path.exists(self.similarity_metadata_path) and not self.overwrite_cache:
            try:
                with open(self.similarity_metadata_path, 'r') as f:
                    similarity_cache_metadata = json.load(f)
            except Exception as e:
                print(f"유사도 캐시 메타데이터 로드 중 오류 발생: {e}")
        
        # 임베딩 캐시 디렉토리에서 사용 가능한 특성 찾기
        available_features = []
        
        # 먼저 임베딩 캐시 디렉토리 확인
        for file in os.listdir(self.embedding_cache_dir):
            if file.endswith("_embeddings.json"):
                feature = file.replace("_embeddings.json", "")
                available_features.append(feature)
        
        # 기존 호환성을 위해 임베딩 디렉토리도 확인
        if not available_features:
            for file in os.listdir(self.embeddings_dir):
                if file.endswith("_embeddings.json"):
                    feature = file.replace("_embeddings.json", "")
                    available_features.append(feature)
        
        if not available_features:
            print("사용 가능한 임베딩 파일이 없습니다. 기본 특성으로 유사도 행렬을 생성합니다.")
            # 기본 특성 목록 (6가지 특성)
            features = list(self.feature_weights.keys())
            
            # 각 특성별로 유사도 행렬 계산
            for feature in features:
                # 랜덤 유사도 행렬 생성 (원래 범위 유지)
                n_users = len(self.profiles)
                similarity_matrix = np.random.random((n_users, n_users))
                np.fill_diagonal(similarity_matrix, 1.0)  # 자기 자신과의 유사도는 1
                
                # 대칭 행렬로 만들기
                similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
                
                # 저장
                self.feature_similarity_matrices[feature] = similarity_matrix
                
                # 유사도 캐시에 저장
                similarity_cache_file = os.path.join(self.similarity_cache_dir, f"{feature}_similarity.json")
                
                # 유사도 행렬 메타데이터
                similarity_metadata = {
                    "feature": feature,
                    "user_count": n_users,
                    "shape": similarity_matrix.shape,
                    "mean": float(np.mean(similarity_matrix)),
                    "median": float(np.median(similarity_matrix)),
                    "min": float(np.min(similarity_matrix)),
                    "max": float(np.max(similarity_matrix)),
                    "timestamp": time.time()
                }
                
                # 유사도 행렬 데이터
                similarity_data = {
                    "feature": feature,
                    "matrix": similarity_matrix.tolist(),
                    "metadata": similarity_metadata
                }
                
                # 유사도 캐시에 저장
                with open(similarity_cache_file, 'w') as f:
                    json.dump(similarity_data, f, indent=2)
                
                # 메타데이터 업데이트
                similarity_cache_metadata[feature] = {
                    "feature": feature,
                    "user_count": n_users,
                    "file_path": similarity_cache_file,
                    "timestamp": time.time()
                }
                
                # 기존 호환성을 위한 파일 저장
                json_file = os.path.join(self.embeddings_dir, f"{feature}_similarity.json")
                similarity_dict = {
                    "feature": feature,
                    "shape": similarity_matrix.shape,
                    "mean": float(np.mean(similarity_matrix)),
                    "median": float(np.median(similarity_matrix)),
                    "min": float(np.min(similarity_matrix)),
                    "max": float(np.max(similarity_matrix)),
                    "matrix": similarity_matrix.tolist()  # 전체 행렬 저장
                }
                with open(json_file, 'w') as f:
                    json.dump(similarity_dict, f, indent=2)
        else:
            print(f"임베딩 파일을 찾았습니다: {available_features}")
            # 임베딩 파일이 있는 특성에 대해 코사인 유사도 계산
            for feature in available_features:
                # 유사도 행렬 캐시 파일 경로
                similarity_cache_file = os.path.join(self.similarity_cache_dir, f"{feature}_similarity.json")
                
                # 이미 계산된 유사도 행렬이 있는지 확인
                if not self.overwrite_cache and feature in similarity_cache_metadata and os.path.exists(similarity_cache_file):
                    print(f"저장된 유사도 행렬을 로드합니다: {feature}")
                    with open(similarity_cache_file, 'r') as f:
                        similarity_data = json.load(f)
                        self.feature_similarity_matrices[feature] = np.array(similarity_data["matrix"])
                    continue
                
                # 임베딩 캐시에서 임베딩 로드
                embedding_cache_file = os.path.join(self.embedding_cache_dir, f"{feature}_embeddings.json")
                
                if os.path.exists(embedding_cache_file):
                    print(f"임베딩 캐시에서 임베딩을 로드하여 유사도 행렬을 계산합니다: {feature}")
                    with open(embedding_cache_file, 'r') as f:
                        cached_data = json.load(f)
                        embeddings = np.array(cached_data["embeddings"])
                else:
                    # 기존 호환성을 위해 임베딩 디렉토리에서도 확인
                    embeddings_file = os.path.join(self.embeddings_dir, f"{feature}_embeddings.json")
                    if os.path.exists(embeddings_file):
                        print(f"임베딩 디렉토리에서 임베딩을 로드하여 유사도 행렬을 계산합니다: {feature}")
                        with open(embeddings_file, 'r') as f:
                            embeddings_dict = json.load(f)
                            embeddings = np.array(embeddings_dict["embeddings"])
                    else:
                        print(f"경고: {feature} 임베딩 파일을 찾을 수 없습니다.")
                        continue
                
                # 코사인 유사도 계산
                from sklearn.metrics.pairwise import cosine_similarity
                from scipy.spatial.distance import pdist, squareform
                
                # 임베딩 벡터가 모두 0인지 확인
                zero_vectors = np.all(embeddings == 0, axis=1)
                zero_count = np.sum(zero_vectors)
                print(f"특성 '{feature}'에서 0 벡터 수: {zero_count}/{len(embeddings)}")
                
                # 임베딩 벡터 정규화 (길이가 1이 되도록)
                normalized_embeddings = np.copy(embeddings)
                for i in range(len(embeddings)):
                    if not zero_vectors[i]:
                        norm = np.linalg.norm(embeddings[i])
                        if norm > 0:
                            normalized_embeddings[i] = embeddings[i] / norm
                
                # 텍스트 유사도 계산 (텍스트가 동일한지 확인)
                text_similarity_matrix = np.zeros((len(self.profiles), len(self.profiles)))
                for i in range(len(self.profiles)):
                    for j in range(len(self.profiles)):
                        if i == j:
                            text_similarity_matrix[i, j] = 1.0  # 자기 자신과의 유사도는 1
                        else:
                            text_i = self.profiles[i].get(feature, '')
                            text_j = self.profiles[j].get(feature, '')
                            # 텍스트가 동일한 경우에만 1.0, 그렇지 않으면 0.0
                            text_similarity_matrix[i, j] = 1.0 if text_i == text_j and text_i != '' else 0.0
                
                # 코사인 유사도 계산
                cosine_similarity_matrix = cosine_similarity(normalized_embeddings)
                
                # 유클리드 거리 기반 유사도 계산 (거리가 가까울수록 유사도가 높음)
                euclidean_distances = squareform(pdist(normalized_embeddings, 'euclidean'))
                # 거리를 유사도로 변환 (최대 거리를 1로 정규화하고 1에서 빼서 유사도로 변환)
                max_distance = np.max(euclidean_distances) if np.max(euclidean_distances) > 0 else 1.0
                euclidean_similarity_matrix = 1.0 - (euclidean_distances / max_distance)
                
                # 코사인 유사도와 유클리드 유사도를 결합 (가중 평균)
                similarity_matrix = 0.7 * cosine_similarity_matrix + 0.3 * euclidean_similarity_matrix
                
                # 텍스트가 다른데 유사도가 1.0인 경우 보정
                for i in range(similarity_matrix.shape[0]):
                    for j in range(similarity_matrix.shape[1]):
                        if i != j and similarity_matrix[i, j] >= 0.999 and text_similarity_matrix[i, j] == 0.0:
                            # 텍스트가 다른데 유사도가 1.0에 가까운 경우 0.9로 제한
                            similarity_matrix[i, j] = 0.9
                
                # 유사도 통계 출력 (보정 전)
                print(f"특성 '{feature}' 유사도 통계 (보정 전):")
                print(f"  평균: {np.mean(similarity_matrix):.4f}")
                print(f"  중앙값: {np.median(similarity_matrix):.4f}")
                print(f"  최소값: {np.min(similarity_matrix):.4f}")
                print(f"  최대값: {np.max(similarity_matrix):.4f}")
                print(f"  1.0 값 개수: {np.sum(similarity_matrix == 1.0)}")
                
                # 대각선을 제외한 값들 중 1인 값들 찾기
                mask = np.eye(similarity_matrix.shape[0], dtype=bool)
                off_diagonal_ones = np.where((similarity_matrix == 1.0) & ~mask)
                if len(off_diagonal_ones[0]) > 0:
                    print(f"  대각선 제외 1.0 값 개수: {len(off_diagonal_ones[0])}")
                    for i, j in zip(off_diagonal_ones[0][:10], off_diagonal_ones[1][:10]):  # 처음 10개만 출력
                        print(f"    유사도 1.0: 프로필 {i}와 프로필 {j}")
                        print(f"      프로필 {i} 텍스트: {self.profiles[i].get(feature, '')[:50]}...")
                        print(f"      프로필 {j} 텍스트: {self.profiles[j].get(feature, '')[:50]}...")
                        # 텍스트가 동일한지 확인
                        if self.profiles[i].get(feature, '') == self.profiles[j].get(feature, ''):
                            print(f"      텍스트 동일: 예")
                        else:
                            print(f"      텍스트 동일: 아니오")
                
                # 0 벡터에 대한 유사도 보정 (자기 자신과의 유사도만 1로 설정, 다른 벡터와의 유사도는 0으로 설정)
                for i in range(len(zero_vectors)):
                    if zero_vectors[i]:
                        similarity_matrix[i, :] = 0  # 행을 0으로 설정
                        similarity_matrix[:, i] = 0  # 열을 0으로 설정
                        similarity_matrix[i, i] = 1  # 자기 자신과의 유사도는 1
                
                # 유사도 값이 너무 높은 경우 보정 (대각선 제외)
                threshold = 0.9  # 유사도 임계값 (더 낮게 설정)
                for i in range(similarity_matrix.shape[0]):
                    for j in range(similarity_matrix.shape[1]):
                        if i != j and similarity_matrix[i, j] > threshold:
                            # 텍스트가 동일한 경우에는 높은 유사도 유지
                            if text_similarity_matrix[i, j] == 1.0:
                                continue
                            
                            # 텍스트가 다른 경우 유사도 값을 스케일링하여 낮춤 (0.9 이상의 값을 0.7-0.9 범위로 스케일링)
                            similarity_matrix[i, j] = 0.7 + (similarity_matrix[i, j] - threshold) * (0.2 / (1 - threshold))
                
                # 유사도 통계 출력 (보정 후)
                print(f"특성 '{feature}' 유사도 통계 (보정 후):")
                print(f"  평균: {np.mean(similarity_matrix):.4f}")
                print(f"  중앙값: {np.median(similarity_matrix):.4f}")
                print(f"  최소값: {np.min(similarity_matrix):.4f}")
                print(f"  최대값: {np.max(similarity_matrix):.4f}")
                print(f"  1.0 값 개수: {np.sum(similarity_matrix == 1.0)}")
                
                # 대각선을 제외한 값들 중 1인 값들 찾기 (보정 후)
                off_diagonal_ones = np.where((similarity_matrix == 1.0) & ~mask)
                if len(off_diagonal_ones[0]) > 0:
                    print(f"  대각선 제외 1.0 값 개수 (보정 후): {len(off_diagonal_ones[0])}")
                    for i, j in zip(off_diagonal_ones[0][:10], off_diagonal_ones[1][:10]):  # 처음 10개만 출력
                        print(f"    유사도 1.0: 프로필 {i}와 프로필 {j}")
                        print(f"      프로필 {i} 텍스트: {self.profiles[i].get(feature, '')[:50]}...")
                        print(f"      프로필 {j} 텍스트: {self.profiles[j].get(feature, '')[:50]}...")
                        # 텍스트가 동일한지 확인
                        if self.profiles[i].get(feature, '') == self.profiles[j].get(feature, ''):
                            print(f"      텍스트 동일: 예")
                        else:
                            print(f"      텍스트 동일: 아니오")
                
                # 메모리에 저장
                self.feature_similarity_matrices[feature] = similarity_matrix
                
                # 유사도 행렬 메타데이터
                n_users = similarity_matrix.shape[0]
                similarity_metadata = {
                    "feature": feature,
                    "user_count": n_users,
                    "shape": similarity_matrix.shape,
                    "mean": float(np.mean(similarity_matrix)),
                    "median": float(np.median(similarity_matrix)),
                    "min": float(np.min(similarity_matrix)),
                    "max": float(np.max(similarity_matrix)),
                    "timestamp": time.time()
                }
                
                # 유사도 행렬 데이터
                similarity_data = {
                    "feature": feature,
                    "matrix": similarity_matrix.tolist(),
                    "metadata": similarity_metadata
                }
                
                # 유사도 캐시에 저장
                with open(similarity_cache_file, 'w') as f:
                    json.dump(similarity_data, f, indent=2)
                
                # 메타데이터 업데이트
                similarity_cache_metadata[feature] = {
                    "feature": feature,
                    "user_count": n_users,
                    "file_path": similarity_cache_file,
                    "timestamp": time.time()
                }
                
                print(f"{feature} 특성의 유사도 행렬을 저장했습니다. 형태: {similarity_matrix.shape}")
        
        # 종합 유사도 계산
        print("특성별 가중치를 사용하여 종합 유사도 행렬을 계산합니다...")
        total_score_matrix = self.compute_total_score_matrix(self.feature_weights)
        self.feature_similarity_matrices['total_score'] = total_score_matrix
        
        # 종합 유사도 행렬을 파일에 저장
        total_score_cache_file = os.path.join(self.similarity_cache_dir, "total_score_similarity.json")
        
        # 유사도 행렬 메타데이터
        total_score_metadata = {
            "feature": "total_score",
            "user_count": total_score_matrix.shape[0],
            "shape": total_score_matrix.shape,
            "mean": float(np.mean(total_score_matrix)),
            "median": float(np.median(total_score_matrix)),
            "min": float(np.min(total_score_matrix)),
            "max": float(np.max(total_score_matrix)),
            "timestamp": time.time()
        }
        
        # 유사도 행렬 데이터
        total_score_data = {
            "feature": "total_score",
            "matrix": total_score_matrix.tolist(),
            "metadata": total_score_metadata
        }
        
        # 유사도 캐시에 저장
        with open(total_score_cache_file, 'w') as f:
            json.dump(total_score_data, f, indent=2)
        
        # 메타데이터 업데이트
        similarity_cache_metadata['total_score'] = {
            "feature": "total_score",
            "user_count": total_score_matrix.shape[0],
            "file_path": total_score_cache_file,
            "timestamp": time.time()
        }
        
        # 기존 호환성을 위한 파일 저장
        total_score_file = os.path.join(self.embeddings_dir, "total_score_similarity.json")
        total_score_dict = {
            "feature": "total_score",
            "shape": total_score_matrix.shape,
            "mean": float(np.mean(total_score_matrix)),
            "median": float(np.median(total_score_matrix)),
            "min": float(np.min(total_score_matrix)),
            "max": float(np.max(total_score_matrix)),
            "matrix": total_score_matrix.tolist()  # 전체 행렬 저장
        }
        with open(total_score_file, 'w') as f:
            json.dump(total_score_dict, f, indent=2)
            
        print(f"종합 유사도 행렬을 저장했습니다. 형태: {total_score_matrix.shape}")

        # 전체 유사도 캐시 메타데이터 저장
        with open(self.similarity_metadata_path, 'w') as f:
            json.dump(similarity_cache_metadata, f, indent=2)
        
        print(f"유사도 캐시 메타데이터를 저장했습니다: {self.similarity_metadata_path}")
        print(f"특성별 유사도 행렬 계산 완료: {list(self.feature_similarity_matrices.keys())}")        
    
    def compute_total_score_matrix(self, feature_weights: Dict):
        """특성별 가중치를 적용하여 종합 유사도 행렬 계산"""
        n_users = len(self.profiles)
        total_similarity_matrix = np.zeros((n_users, n_users))

        # 특성별 가중치와 유사도 행렬을 NumPy 배열로 변환
        feature_names = list(feature_weights.keys())
        weights = np.array([feature_weights[feature] for feature in feature_names if feature in self.feature_similarity_matrices])
        similarity_matrices = np.array([self.feature_similarity_matrices[feature] for feature in feature_names if feature in self.feature_similarity_matrices])

        # 가중 평균 계산: 각 특성별 유사도 행렬에 가중치를 곱하고 합산
        if weights.size > 0:  # 유사도 행렬이 하나라도 존재할 경우에만 계산
            weighted_sum = np.sum(similarity_matrices * weights[:, np.newaxis, np.newaxis], axis=0)
            total_similarity_matrix = weighted_sum / np.sum(weights)  # 가중치 합으로 나눔
        else:
            print("경고: 사용 가능한 특성 유사도 행렬이 없습니다. 0으로 채워진 행렬을 반환합니다.")

        return total_similarity_matrix

    
    def _load_cached_similarities(self):
        """캐시된 유사도 데이터프레임 로드"""
        try:
            if os.path.exists(self.similarity_metadata_path):
                with open(self.similarity_metadata_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"유사도 캐시 로드 중 오류 발생: {e}")
            return {}
    
    def _save_cached_similarities(self, cache_dict):
        """유사도 데이터프레임을 캐시에 저장"""
        try:
            with open(self.similarity_metadata_path, 'w') as f:
                json.dump(cache_dict, f)
        except Exception as e:
            print(f"유사도 캐시 저장 중 오류 발생: {e}")

    
    def run_recommendation_pipeline(self, user_idx=None, top_n=5, overwrite_cache=None):
        """
        추천 파이프라인 실행 (모든 프로필과 특성 사용)
        
        Args:
            user_idx (int, optional): 추천을 생성할 사용자 인덱스. 기본값은 None으로, 랜덤 선택됨.
            top_n (int, optional): 생성할 추천 수. 기본값은 5.
            overwrite_cache (bool, optional): 이 실행에서 캐시를 덮어쓸지 여부.
                                             None이면 클래스 초기화 시 설정한 값 사용.
        
        Returns:
            list: 추천 결과 목록 (사용자 인덱스, 점수)
        """
        # 이 실행에서만 캐시 덮어쓰기 설정 변경 (임시)
        original_overwrite = self.overwrite_cache
        if overwrite_cache is not None:
            self.overwrite_cache = overwrite_cache
        
        # 임베딩 및 유사도 행렬 계산
        self.compute_embeddings()
        self.compute_feature_similarity_matrices()

    
    def generate_recommendations(self, user_idx, top_n=5):
        """
        특정 사용자에 대한 추천 생성
        
        Args:
            user_idx (int): 추천을 생성할 사용자 인덱스
            top_n (int): 생성할 추천 수
            
        Returns:
            list: 추천 결과 목록 (사용자 인덱스, 점수)
        """
        # total_score 유사도 행렬이 있는지 확인
        if 'total_score' not in self.feature_similarity_matrices:
            print("경고: total_score 유사도 행렬이 없습니다. 기본 유사도 행렬을 생성합니다.")
            # 종합 유사도 계산
            total_score_matrix = self.compute_total_score_matrix(self.feature_weights)
            self.feature_similarity_matrices['total_score'] = total_score_matrix
        
        # 유사도 행렬 가져오기
        sim_matrix = self.feature_similarity_matrices['total_score']
        
        # 특정 사용자에 대한 유사도 점수 가져오기
        user_similarities = sim_matrix[user_idx]
        
        # 자기 자신 제외
        user_similarities[user_idx] = -1
        
        # 상위 N개 유사한 사용자 찾기
        top_similar_indices = np.argsort(user_similarities)[::-1][:top_n]
        
        # 추천 결과 생성 (인덱스, 점수)
        recommendations = [(idx, user_similarities[idx]) for idx in top_similar_indices if user_similarities[idx] > 0]
        
        return recommendations

    def load_embeddings(self, feature):
        """저장된 임베딩 로드"""
        # 임베딩 캐시 파일 경로 (JSON 사용)
        embeddings_file = os.path.join(self.embeddings_dir, f"{feature}_embeddings.json")
        
        if os.path.exists(embeddings_file):
            print(f"저장된 임베딩을 로드합니다: {feature}")
            with open(embeddings_file, 'r') as f:
                embeddings_dict = json.load(f)
                return np.array(embeddings_dict["embeddings"])
        else:
            print(f"경고: {feature} 임베딩 파일을 찾을 수 없습니다.")
            return None

# 메인 실행 코드
if __name__ == "__main__":
    recommender = MeetupRecommendationSystem(profiles_path="data/profiles/generated_virtual_users_v0.1.json", overwrite_cache=True)
    recommender.run_recommendation_pipeline()
