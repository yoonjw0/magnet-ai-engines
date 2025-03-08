"""
Question Generation Engine for Magnet AI

This module is responsible for generating questions to facilitate conversations
between users based on their profiles and interests.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'system_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)

# 기본 질문 상수
DEFAULT_QUESTIONS = [
    {
        "question_id": "q_1_001",
        "text": "최근 팀에서 맡은 역할 중에서 가장 도전적이었던 건 뭐야?",
        "is_selected": False
    },
    {
        "question_id": "q_1_002",
        "text": "요즘 새로운 기술 중에서 관심 가는 게 있어?",
        "is_selected": False
    }
]

# 하드코딩된 사용자 프로필
DEFAULT_USER_PROFILE = {
    "user_id": "b7f3c8d1-92e5-4d3f-9b8e-63f27f123456",
    "name": "김민수",
    "age": 32,
    "email": "minsu.kim0124@naver.com",
    "region": "서울",
    "district": "강남구",
    "sns_url": "https://linkedin.com/in/minsu-kim",
    "job_title": "AI Engineer",
    "career_years": 7,
    "MBTI": "INFP",
    "specialties": ["AI 모델 최적화", "MLOps", "데이터 분석"],
    "skills": ["PyTorch", "Docker", "Kubernetes", "Ray"],
    "work_culture": ["데이터 기반 문제 해결", "협업과 지식 공유", "자동화 및 효율성 극대화"],
    "interests": ["Generative AI", "AI 모델 서빙 최적화", "AI 윤리 문제"]
}

class QuestionGenerationEngine:
    """
    Engine for generating personalized questions for users based on their profiles.
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """Initialize the Question Generation Engine.
        
        Args:
            user_id: Optional user ID to load a specific user profile. If not provided,
                    uses the default user profile.
        """
        # Load user profile
        self.user_profile = self._load_user_profile(user_id) if user_id else DEFAULT_USER_PROFILE
        self.conversation_history = []
        self.current_turn_id = 0
        self.conversation_id = str(uuid.uuid4())
        
        # Initialize the chat model
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=1.0
        )
        
        # Set initial questions based on user profile
        self.last_generated_questions = DEFAULT_QUESTIONS if self.user_profile == DEFAULT_USER_PROFILE else self._generate_initial_questions()
        
        # Load existing conversation history if available
        self._load_conversation_history()
    
    def _load_user_profile(self, user_id: str) -> Dict:
        """Load user profile from the generated virtual users file.
        
        Args:
            user_id: The ID of the user to load.
            
        Returns:
            Dict containing the user profile data.
            
        Raises:
            FileNotFoundError: If the profiles file doesn't exist.
            ValueError: If the user ID is not found in the profiles.
        """
        profiles_path = "/Users/jinwooyoon/CascadeProjects/magnet-ai-engines_qa_gen/data/profiles/generated_virtual_users.json"
        
        if not os.path.exists(profiles_path):
            raise FileNotFoundError(f"Profiles file not found at: {profiles_path}")
        
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            
        for profile in profiles:
            if profile.get('user_id') == user_id:
                return profile
                
        raise ValueError(f"User profile not found for ID: {user_id}")
    
    def _create_system_prompt(self) -> str:
        """Create a system prompt with user profile information and conversation history."""
        conversation_context = ""
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            selected_question = next((q for q in last_turn["ai_generated_questions"] if q["is_selected"]), None)
            if selected_question:
                conversation_context = f"""

마지막 대화 내용:
질문: {selected_question['text']}
답변: {last_turn['user_response']}"""

        return f"""당신은 사용자와 대화하는 AI 에이전트입니다. 다음 사용자 프로필과 대화 내용을 기반으로 자연스럽고 흥미로운 대화를 이어가야 합니다:

{json.dumps(self.user_profile, ensure_ascii=False, indent=2)}{conversation_context}

중요: 당신은 반드시 JSON 배열 형식으로만 응답해야 합니다. 다른 텍스트나 설명을 추가하지 마세요. 오직 ["질문1", "질문2"] 형식의 JSON 배열만 반환해야 합니다.

질문 생성 규칙:
1. 사용자의 경험, 관심사, 및 전문성을 고려해야 합니다.
2. 이전 대화 내용을 고려하여 자연스럽게 이어지는 질문을 생성해야 합니다. 특히 사용자의 마지막 답변에 대해 더 깊이 있는 대화를 이끌어내세요.
3. 질문은 한국어로 작성해야 합니다.
4. 사용자의 답변이 짧거나 모호한 경우, 구체적인 예시나 경험을 물어보며 대화를 풍부하게 만드세요."""

    def _create_conversation_messages(self) -> List[Dict]:
        """Create conversation messages from history."""
        messages = []
        
        # 최근 5개의 대화 히스토리만 사용
        for turn in self.conversation_history[-5:]:
            selected_question = next((q for q in turn["ai_generated_questions"] if q["is_selected"]), None)
            if selected_question:
                messages.append({
                    "role": "assistant",
                    "content": selected_question["text"]
                })
                messages.append({
                    "role": "user",
                    "content": turn["user_response"]
                })
        
        return messages

    def _load_conversation_history(self):
        """Load existing conversation history for the user if available."""
        history_dir = "data/conversations"
        if not os.path.exists(history_dir):
            return
        
        # Find the most recent conversation file for this user
        user_files = []
        for filename in os.listdir(history_dir):
            filepath = os.path.join(history_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('user_id') == self.user_profile['user_id']:
                        user_files.append((filepath, data))
            except:
                continue
        
        if user_files:
            # Sort by timestamp in filename (newest first)
            latest_file = sorted(user_files, key=lambda x: x[0], reverse=True)[0]
            self.conversation_history = latest_file[1]['conversation_history']
            self.conversation_id = latest_file[1]['conversation_id']
            self.current_turn_id = len(self.conversation_history)
    
    def _generate_initial_questions(self) -> List[Dict]:
        """Generate initial questions based on user profile."""
        system_prompt = self._create_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "사용자와의 첫 대화를 시작하기 위한 2개의 질문을 JSON 배열 형식으로 제안해주세요. 사용자의 경험, 관심사, 전문성을 고려하여 흐미로운 대화를 이끌어낼 수 있는 질문을 작성해주세요."}
        ]
        
        messages_obj = [
            SystemMessage(content=system_prompt) if msg["role"] == "system"
            else HumanMessage(content=msg["content"])
            for msg in messages
        ]
        
        response = self.chat_model.invoke(messages_obj)
        questions = json.loads(response.content)
        
        return [
            {"question_id": f"q_1_00{i}", "text": q, "is_selected": False}
            for i, q in enumerate(questions[:2], start=1)
        ]
    
    def generate_questions(self) -> List[Dict]:
        """Generate two different questions based on user profile and conversation history."""
        # If this is the first turn, return the initial questions
        if self.current_turn_id == 0:
            return self.last_generated_questions
        
        # Otherwise, generate new questions based on conversation history
        system_prompt = self._create_system_prompt()
        conversation_messages = self._create_conversation_messages()
        
        # 이전 대화 내용을 분석하여 더 구체적인 지시 생성
        user_instruction = "이전 대화를 기반으로 2개의 질문을 JSON 배열 형식으로 제안해주세요."
        
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            last_response = last_turn['user_response']
            
            # 답변이 짧거나 모호한 경우 더 구체적인 질문 요청
            if len(last_response.strip()) < 20 or '모르' in last_response or '추천' in last_response:
                user_instruction = "사용자의 답변이 다소 짧거나 모호합니다. 더 구체적인 경험이나 생각을 끌어낼 수 있는 2개의 질문을 JSON 배열 형식으로 제안해주세요."
            else:
                user_instruction = "사용자의 답변을 더 깊이 있게 탐구하고 관련 경험을 끌어낼 수 있는 2개의 질문을 JSON 배열 형식으로 제안해주세요."
        
        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": msg["role"], "content": msg["content"]} for msg in conversation_messages],
            {"role": "user", "content": f"{user_instruction} 반드시 [\"질문1\", \"질문2\"] 형식으로만 응답해야 합니다."}
        ]
        
        messages_obj = [
            SystemMessage(content=system_prompt) if msg["role"] == "system"
            else (HumanMessage(content=msg["content"]) if msg["role"] == "user"
                  else AIMessage(content=msg["content"]))
            for msg in messages
        ]
        logging.info("=== 요청 메시지 ===")
        for msg in messages:
            logging.info(f"Role: {msg['role']}")
            logging.info(f"Content: {msg['content']}")
        
        response = self.chat_model.invoke(messages_obj)
        logging.info("=== LLM 응답 ===")
        logging.info(f"Content: {response.content}")
        
        questions = json.loads(response.content)
        self.last_generated_questions = [
            {"question_id": f"q_{self.current_turn_id + 1}_00{i}", "text": q, "is_selected": False}
            for i, q in enumerate(questions[:2], start=1)  # 최대 2개의 질문만 사용
        ]
        return self.last_generated_questions
    
    def save_conversation(self, output_path: str = "data/conversations"):
        """Save the conversation history to a JSON file."""
        os.makedirs(output_path, exist_ok=True)
        
        conversation_data = {
            "conversation_id": self.conversation_id,
            "user_id": self.user_profile["user_id"],
            "conversation_history": self.conversation_history
        }
        
        filename = f"conversation_{self.conversation_id}.json"
        filepath = os.path.join(output_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"대화가 저장되었습니다: {filepath}")
        return filepath

    def add_turn(self, selected_question_id: str, user_response: str):
        """Add a new turn to the conversation history."""
        self.current_turn_id += 1
        # 선택된 질문 표시
        for question in self.last_generated_questions:
            question["is_selected"] = (question["question_id"] == selected_question_id)
        
        turn = {
            "turn_id": self.current_turn_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ai_generated_questions": self.last_generated_questions,
            "user_response": user_response,
            "context": []
        }
        
        # 이전 턴의 컨텍스트 추가
        if len(self.conversation_history) > 0:
            turn["context"].append({
                "reference_turn_id": self.conversation_history[-1]["turn_id"],
                "relevant_info": self.conversation_history[-1]["user_response"]
            })
        
        self.conversation_history.append(turn)

def run_conversation(user_id: Optional[str] = None):
    """Run the interactive conversation loop.
    
    Args:
        user_id: Optional user ID to load a specific user profile.
    """
    engine = QuestionGenerationEngine(user_id)
    
    # 현재 활성화된 사용자 정보 출력
    logging.info("=== 현재 활성화된 사용자 정보 ===")
    logging.info(f"이름: {engine.user_profile.get('name', 'N/A')}")
    logging.info(f"직책: {engine.user_profile.get('job_title', 'N/A')}")
    logging.info(f"전문분야: {engine.user_profile.get('specialties', 'N/A')}")
    logging.info(f"관심사: {engine.user_profile.get('interests', 'N/A')}")
    
    if engine.conversation_history:
        logging.info(f"기존 대화 내역을 불러왔습니다. 현재 {len(engine.conversation_history)}번의 대화가 있습니다.")
    print("\n대화를 시작합니다. 종료하려면 'q' 또는 'quit'를 입력하세요.\n")
    
    while True:
        # 두 개의 질문 생성
        questions = engine.generate_questions()
        
        # 질문 출력
        print("\n다음 질문 중 하나를 선택해주세요:")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q['text']}")
        print("q. 대화 종료")
        
        # 사용자 선택 입력
        choice = input("\n선택 (1/2/q): ").strip().lower()
        
        if choice in ['q', 'quit']:
            break
        
        if choice not in ['1', '2']:
            print("\n올바른 선택이 아닙니다. 1, 2, 또는 q를 입력해주세요.")
            continue
        
        selected_question = questions[int(choice) - 1]
        print(f"\n선택한 질문: {selected_question['text']}")
        
        # 사용자 응답 입력
        response = input("\n답변을 입력해주세요: ").strip()
        if response.lower() in ['q', 'quit']:
            break
        
        # 대화 턴 추가
        engine.add_turn(selected_question['question_id'], response)
    
    # 대화 저장
    if engine.conversation_history:
        saved_path = engine.save_conversation()
        print(f"\n대화가 저장되었습니다: {saved_path}")

# Example usage
if __name__ == "__main__":
    import sys
    user_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_conversation(user_id)
