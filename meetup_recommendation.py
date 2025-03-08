#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
밋업 매칭 추천 시스템

이 모듈은 가상 유저 프로필을 기반으로 밋업 매칭을 추천하는 시스템을 구현합니다.
하이브리드 접근법을 사용하여 콘텐츠 기반 유사도와 피드백 기반 예측을 결합합니다.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import random
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

class MeetupRecommendationSystem:
    """밋업 매칭 추천 시스템 클래스"""
    
    def __init__(self, profiles_path="data/profiles/generated_virtual_users.json", cache_dir="cached_data"):
        """
        초기화 함수
        
        Args:
            profiles_path (str): 가상 유저 프로필 파일 경로
            cache_dir (str): 캐시 데이터를 저장할 디렉토리 경로
        """
        self.profiles_path = profiles_path
        self.cache_dir = cache_dir
        self.similarity_cache_path = os.path.join(cache_dir, "similarity_cache.pkl")
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        
        self.profiles = self._load_profiles()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=1.0,  # 최대 온도 사용 (Gemini API는 0.0-1.0 범위만 허용)
            convert_system_message_to_human=True,
            api_key=os.getenv('GOOGLE_API_KEY'),
        )
        
        # 특성별 가중치 초기화
        self.feature_weights = {
            'specialties': 1.0,
            'skills': 1.0,
            'job_title':1.0,
            'work_culture': 1.0,
            'interests': 1.0,
            'report_summary': 1.0,
            'recent_conversation_history': 1.0
        }
        
        # 임베딩 결과를 저장할 파일 경로 설정
        self.embeddings_path = os.path.join(cache_dir, "embeddings.json")
        
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
        # 간단한 전처리 (소문자 변환, 특수문자 제거 등)
        return text.lower().strip()
    
    def _get_embedding_with_retry(self, text, max_retries=5, initial_wait=2):
        """임베딩을 얻는 함수 (재시도 로직 포함)"""
        import time
        from langchain_google_genai._common import GoogleGenerativeAIError
        
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
        
        import time
        from langchain_google_genai._common import GoogleGenerativeAIError
        
        # 임베딩 결과를 저장할 디렉토리
        embeddings_dir = os.path.join(self.cache_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # 각 특성별로 임베딩 계산
        feature_pbar = tqdm(self.feature_weights.keys(), desc="특성 처리 중")
        for feature in feature_pbar:
            feature_file = os.path.join(embeddings_dir, f"{feature}_embeddings.json")
            feature_pbar.set_description(f"특성 처리 중: {feature}")
            
            # 이미 계산된 결과가 있는지 확인
            if os.path.exists(feature_file):
                feature_pbar.set_postfix_str("저장된 임베딩 로드 중...")
                with open(feature_file, 'r') as f:
                    self.feature_embeddings[feature] = json.load(f)
                continue
            
            feature_pbar.set_postfix_str("임베딩 계산 중...")
            self.feature_embeddings[feature] = []
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
                for profile in batch_profiles:
                    feature_text = self.preprocess_text(profile.get(feature, ""))
                    
                    if feature_text:
                        try:
                            embedding = self._get_embedding_with_retry(feature_text)
                            batch_embeddings.append(embedding)
                        except Exception as e:
                            print(f"임베딩 계산 중 치명적인 오류 발생: {e}")
                            raise  # 오류를 상위로 전파하여 프로세스 중단
                    else:
                        batch_embeddings.append([0] * 768)  # 빈 텍스트는 0 벡터로 처리
                
                self.feature_embeddings[feature].extend(batch_embeddings)
                time.sleep(1)  # API 요청 간 1초 대기
            
            feature_pbar.set_postfix_str("결과 저장 중...")
            
            # 계산된 임베딩을 JSON 파일로 저장
            print(f"{feature} 특성의 임베딩을 저장합니다...")
            with open(feature_file, 'w') as f:
                json.dump(self.feature_embeddings[feature], f)
    
    def compute_similarity_matrices(self):
        """각 특성별 유사도 행렬 계산"""
        print("특성별 유사도 행렬 계산 중...")
        
        sim_pbar = tqdm(self.feature_weights.keys(), desc="유사도 행렬 계산")
        for feature in sim_pbar:
            sim_pbar.set_description(f"유사도 행렬 계산 중: {feature}")
            if feature in self.feature_embeddings:
                embeddings = np.array(self.feature_embeddings[feature])
                similarity_matrix = cosine_similarity(embeddings)
                self.feature_similarity_matrices[feature] = similarity_matrix
    
    def get_content_based_similarity(self, user_idx1, user_idx2):
        """콘텐츠 기반 유사도 계산"""
        total_similarity = 0
        total_weight = 0
        
        for feature, weight in self.feature_weights.items():
            if feature in self.feature_similarity_matrices:
                similarity = self.feature_similarity_matrices[feature][user_idx1, user_idx2]
                total_similarity += similarity * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_similarity / total_weight
        return 0
    
    def get_feedback_based_score(self, user_idx1, user_idx2):
        """피드백 기반 점수 계산"""
        # 직접 피드백이 있는 경우
        if not np.isnan(self.feedback_matrix[user_idx1, user_idx2]):
            return self.feedback_matrix[user_idx1, user_idx2]
        
        # 피드백 데이터가 전혀 없는지 확인
        if np.isnan(self.feedback_matrix).all():
            return 0.5  # 피드백 데이터가 없으면 중립값 반환
        
        # 피드백이 없는 경우 협업 필터링 기반 예측
        similar_users = []
        for i in range(len(self.profiles)):
            if i != user_idx1 and not np.isnan(self.feedback_matrix[i, user_idx2]):
                content_similarity = self.get_content_based_similarity(user_idx1, i)
                similar_users.append((i, content_similarity))
        
        # 상위 3명의 유사 사용자 선택
        similar_users.sort(key=lambda x: x[1], reverse=True)
        top_similar_users = similar_users[:3]
        
        if not top_similar_users:
            return 0.5  # 기본값
        
        # 유사 사용자들의 가중 평균 피드백
        weighted_sum = sum(self.feedback_matrix[i, user_idx2] * sim for i, sim in top_similar_users)
        total_weight = sum(sim for _, sim in top_similar_users)
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.5  # 기본값
    
    def compute_hybrid_score(self, user_idx1, user_idx2, content_weight=0.7, feedback_weight=0.3):
        """하이브리드 매칭 점수 계산"""
        content_score = self.get_content_based_similarity(user_idx1, user_idx2)
        
        # 피드백 데이터가 있는지 확인
        has_feedback = not np.isnan(self.feedback_matrix).all()
        
        if has_feedback:
            # 피드백 데이터가 있는 경우 하이브리드 점수 계산
            feedback_score = self.get_feedback_based_score(user_idx1, user_idx2)
            return (content_score * content_weight) + (feedback_score * feedback_weight)
        else:
            # 피드백 데이터가 없는 경우 콘텐츠 기반 점수만 사용
            return content_score
    
    def generate_recommendations(self, user_idx, top_n=5):
        """특정 사용자를 위한 추천 생성"""
        recommendations = []
        
        for i in range(len(self.profiles)):
            if i != user_idx:  # 자기 자신 제외
                score = self.compute_hybrid_score(user_idx, i)
                recommendations.append((i, score))
        
        # 점수 기준 내림차순 정렬
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 N개 추천 반환
        return recommendations[:top_n]
    
    def generate_recommendation_explanation(self, user_idx1, user_idx2):
        """추천 설명 생성"""
        user1 = self.profiles[user_idx1]
        user2 = self.profiles[user_idx2]
        
        # 유사도 데이터프레임 가져오기
        similarity_df = self.get_feature_similarity_dataframe(user_idx1, user_idx2)
        
        # 가장 유사한 특성 3개 추출
        feature_similarities = similarity_df[~similarity_df['Feature'].isin(['Content Score', 'Hybrid Score'])]
        top_features = feature_similarities.nlargest(3, 'Similarity')
        
        # 사용자 프로필 정보 정리
        user1_info = {
            "name": user1["name"],
            "job_title": user1["job_title"],
            "specialties": user1.get("specialties", ""),
            "skills": user1.get("skills", ""),
            "work_culture": user1.get("work_culture", ""),
            "interests": user1.get("interests", ""),
            "report_summary": user1.get("report_summary", "")
        }
        
        user2_info = {
            "name": user2["name"],
            "job_title": user2["job_title"],
            "specialties": user2.get("specialties", ""),
            "skills": user2.get("skills", ""),
            "work_culture": user2.get("work_culture", ""),
            "interests": user2.get("interests", ""),
            "report_summary": user2.get("report_summary", "")
        }
        
        # 새로운 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an engaging matchmaker for professional networking, creating exciting and personalized match explanations.
            Your explanations should be concise, engaging, and highlight the synergies between the two people.
            Create a catchy title that captures the essence of why these two people would benefit from meeting.
            Focus on their complementary skills, shared interests, and potential collaborative opportunities.
            Use a friendly, enthusiastic tone that makes the match sound exciting and valuable.
            Structure your response in Korean with:
            1. A catchy title (e.g., "J와 P의 육합적 구직 스터디가 기대되는 조합이야!")
            2. Brief profile summaries for each person
            3. Key reasons why they should meet (based on their top matching features)
            """),
            ("human", """
            Create an engaging match explanation for these two professionals:
            
            Person 1: {user1_name} ({user1_job})
            Specialties: {user1_specialties}
            Skills: {user1_skills}
            Work Culture: {user1_work_culture}
            Interests: {user1_interests}
            Profile Summary: {user1_report}
            
            Person 2: {user2_name} ({user2_job})
            Specialties: {user2_specialties}
            Skills: {user2_skills}
            Work Culture: {user2_work_culture}
            Interests: {user2_interests}
            Profile Summary: {user2_report}
            
            Their top matching features are:
            {top_features}
            
            Create a catchy title and explain why they should meet in an engaging way.
            """)
        ])
        
        # 특성 유사도 정보 포맷팅
        top_features_text = "\n".join([f"{row['Feature']}: {row['Similarity']:.2f}" for _, row in top_features.iterrows()])
        
        chain = prompt | self.chat_model
        explanation = chain.invoke({
            "user1_name": user1_info["name"],
            "user1_job": user1_info["job_title"],
            "user1_specialties": user1_info["specialties"],
            "user1_skills": user1_info["skills"],
            "user1_work_culture": user1_info["work_culture"],
            "user1_interests": user1_info["interests"],
            "user1_report": user1_info["report_summary"],
            
            "user2_name": user2_info["name"],
            "user2_job": user2_info["job_title"],
            "user2_specialties": user2_info["specialties"],
            "user2_skills": user2_info["skills"],
            "user2_work_culture": user2_info["work_culture"],
            "user2_interests": user2_info["interests"],
            "user2_report": user2_info["report_summary"],
            
            "top_features": top_features_text
        })
        
        return explanation
    
    def _load_cached_similarities(self):
        """캐시된 유사도 데이터프레임 로드"""
        try:
            with open(self.similarity_cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    
    def _save_cached_similarities(self, cache_dict):
        """유사도 데이터프레임을 캐시에 저장"""
        with open(self.similarity_cache_path, 'wb') as f:
            pickle.dump(cache_dict, f)
    
    def get_feature_similarity_dataframe(self, user_idx1, user_idx2):
        """
        각 특성별 유사도 점수를 데이터프레임으로 반환
        캐시된 결과가 있으면 로드하고, 없으면 계산 후 캐시에 저장
        
        Args:
            user_idx1 (int): 첫 번째 사용자 인덱스
            user_idx2 (int): 두 번째 사용자 인덱스
            
        Returns:
            pd.DataFrame: 특성별 유사도 점수가 포함된 데이터프레임
        """
        # 캐시 키 생성
        cache_key = f"{min(user_idx1, user_idx2)}_{max(user_idx1, user_idx2)}"
        
        # 캐시된 결과 확인
        cached_similarities = self._load_cached_similarities()
        if cache_key in cached_similarities:
            return cached_similarities[cache_key]
        
        user1 = self.profiles[user_idx1]
        user2 = self.profiles[user_idx2]
        
        # 각 특성별 유사도 계산
        feature_similarities = {}
        for feature, weight in self.feature_weights.items():
            if feature in self.feature_similarity_matrices:
                similarity = self.feature_similarity_matrices[feature][user_idx1, user_idx2]
                feature_similarities[feature] = similarity
        
        # 전체 점수
        content_score = self.get_content_based_similarity(user_idx1, user_idx2)
        hybrid_score = self.compute_hybrid_score(user_idx1, user_idx2)
        
        # 피드백 데이터가 있는지 확인
        has_feedback = not np.isnan(self.feedback_matrix).all()
        
        # 데이터프레임 생성
        df_data = {
            'Feature': list(feature_similarities.keys()) + ['Content Score', 'Hybrid Score'],
            'Similarity': list(feature_similarities.values()) + [content_score, hybrid_score],
            'Weight': [self.feature_weights.get(feature, 0) for feature in feature_similarities.keys()] + [1.0, 1.0]
        }
        
        # 사용자 정보 추가
        df_data['User1'] = user1.get('name', f'User {user_idx1}')
        df_data['User2'] = user2.get('name', f'User {user_idx2}')
        
        # 데이터프레임 생성
        similarity_df = pd.DataFrame(df_data)
        
        # 결과를 캐시에 저장
        cached_similarities[cache_key] = similarity_df
        self._save_cached_similarities(cached_similarities)
        
        return similarity_df
    
    def run_recommendation_pipeline(self, user_idx=None, top_n=5):
        """추천 파이프라인 실행 (모든 프로필과 특성 사용)"""
        # 임베딩 및 유사도 행렬 계산
        self.compute_embeddings()
        self.compute_similarity_matrices()
        
        # 특정 사용자가 지정되지 않은 경우 랜덤 선택
        if user_idx is None:
            user_idx = random.randint(0, len(self.profiles) - 1)
        
        # 추천 생성
        recommendations = self.generate_recommendations(user_idx, top_n)
        
        # 결과 출력
        user = self.profiles[user_idx]
        print(f"\n{user['name']}님({user['job_title']})을 위한 추천:\n")
        
        for i, (rec_idx, score) in enumerate(recommendations):
            rec_user = self.profiles[rec_idx]
            print(f"\n{i+1}. {rec_user['name']}님({rec_user['job_title']}) - 매칭 점수: {score:.2f}")
            
            # 특성별 유사도 점수 출력
            similarity_df = self.get_feature_similarity_dataframe(user_idx, rec_idx)
            print("\n특성별 유사도 점수:")
            for _, row in similarity_df.iterrows():
                if row['Feature'] in self.feature_weights:
                    print(f"- {row['Feature']}: {row['Similarity']:.2f}")
            
            # 추천 이유 생성
            print("\n추천 이유:")
            explanation = self.generate_recommendation_explanation(user_idx, rec_idx)
            print(explanation)
            print("\n" + "-"*50)
        
        return recommendations

# 메인 실행 코드
if __name__ == "__main__":
    recommender = MeetupRecommendationSystem()
    recommender.run_recommendation_pipeline()
