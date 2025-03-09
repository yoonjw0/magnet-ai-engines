import json
# from langchain import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import time
import os
import random
import concurrent.futures
from tqdm import tqdm
import uuid
from dotenv import load_dotenv
load_dotenv()

# Update deprecated import
from pydantic import BaseModel, Field
class UserData(BaseModel):
    user_id: str = Field(description="blank")
    name: str = Field(description="blank")
    age: int = Field(description="age, between 20 and 50")
    job_title: str = Field(description="job title")
    specialties: str = Field(description="specialties, comma separated")
    skills: str = Field(description="skills, comma separated")
    work_culture: str = Field(description="work culture, comma separated")
    interests: str = Field(description="interests, comma separated")
    report_summary: str = Field(description="report summary")
    recent_conversation_history: str = Field(description="recent conversation history")

# 한국 성씨 목록 (더 다양한 성씨 추가)
surnames = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", 
    "황", "안", "송", "전", "홍", "유", "고", "문", "양", "손", "배", "백", "허", "남", "심",
    "노", "하", "곽", "성", "차", "주", "우", "구", "민", "유", "류", "나", "진", "지", "엄",
    "채", "원", "천", "방", "공", "현", "함", "변", "염", "여", "추", "도", "소", "석", "선"
]

# 남자 이름 목록 (더 다양한 이름 추가)
male_names = [
    "민준", "서준", "도윤", "예준", "시우", "하준", "지호", "준서", "준우", "현우", 
    "도현", "지훈", "건우", "우진", "민재", "현준", "선우", "서진", "연우", "정우",
    "승우", "승현", "시윤", "준혁", "은우", "지환", "승민", "지우", "유준", "진우",
    "시현", "지원", "민성", "준영", "재원", "윤우", "민규", "준호", "태윤", "재민",
    "은찬", "한결", "윤호", "민호", "시원", "건호", "은호", "재윤", "지한", "태민"
]

# 여자 이름 목록 (더 다양한 이름 추가)
female_names = [
    "서연", "서윤", "지우", "민서", "하은", "하윤", "윤서", "지민", "지유", "채원", 
    "지윤", "은서", "수아", "다은", "예은", "지아", "은우", "서현", "예린", "수빈",
    "소율", "예원", "지원", "소윤", "지안", "하린", "채은", "가은", "윤아", "민지",
    "유진", "세아", "예나", "은지", "수민", "혜원", "지은", "유나", "민주", "예진",
    "주아", "시은", "아린", "서영", "연서", "다인", "다연", "은채", "아인", "서아"
]

# Global set to track used names
used_names = set()

def generate_unique_name(used_names):
    """중복되지 않는 이름 생성"""
    gender = random.choice(["male", "female"])
    surname = random.choice(surnames)
    
    if gender == "male":
        first_name = random.choice(male_names)
    else:
        first_name = random.choice(female_names)
    
    full_name = surname + first_name
    
    # 이미 사용된 이름이면 다시 생성
    while full_name in used_names:
        surname = random.choice(surnames)
        if gender == "male":
            first_name = random.choice(male_names)
        else:
            first_name = random.choice(female_names)
        full_name = surname + first_name
    
    used_names.add(full_name)
    return full_name


# 직업 목록 (더 다양한 직업 추가)
jobs = [
    "데이터 과학자", "소프트웨어 엔지니어", "UX/UI 디자이너", "마케팅 매니저", "금융 컨설턴트",
    "역사 교사", "푸드 스타일리스트", "소믈리에", "건축가", "영화 제작 PD", "AI 엔지니어",
    "블록체인 개발자", "호텔 매니저", "벤처 투자자", "AI 윤리 컨설턴트", "도예가", "요리사",
    "영양사", "수제 맥주 브루어", "투자 은행가", "역사학 교수", "고고학 연구원", "ESG 컨설턴트",
    "창업 컨설턴트", "패션 디자이너", "자연어 처리 연구원", "스타트업 마케터", "금융 상품 개발자",
    "부동산 투자 컨설턴트", "소셜 미디어 매니저", "큐레이터", "금융 애널리스트", "UX 리서치 전문가",
    "게임 개발자", "웹 개발자", "모바일 앱 개발자", "클라우드 아키텍트", "사이버 보안 전문가",
    "데이터 엔지니어", "그래픽 디자이너", "콘텐츠 크리에이터", "디지털 마케터", "SEO 전문가",
    "제품 매니저", "프로젝트 매니저", "HR 컨설턴트", "법률 컨설턴트", "의료 정보 분석가"
]

# 관심사 목록 (더 다양한 관심사 추가)
interests = [
    "영화 감상", "여행", "요리", "와인", "독서", "음악 감상", "사진 촬영", "그림 그리기", "등산",
    "캠핑", "수영", "요가", "명상", "테니스", "골프", "축구", "농구", "야구", "배드민턴", "볼링",
    "스키", "스노보드", "서핑", "스케이트보드", "클라이밍", "낚시", "사이클링", "달리기", "마라톤",
    "트레일 러닝", "크로스핏", "필라테스", "발레", "재즈 댄스", "힙합 댄스", "살사 댄스", "탱고",
    "플라멩코", "기타 연주", "피아노 연주", "드럼 연주", "바이올린 연주", "첼로 연주", "플루트 연주",
    "색소폰 연주", "트럼펫 연주", "하모니카 연주", "우쿨렐레 연주", "노래", "합창", "오페라 감상",
    "클래식 음악 감상", "재즈 음악 감상", "록 음악 감상", "힙합 음악 감상", "팝 음악 감상", "인디 음악 감상",
    "전자 음악 감상", "국악 감상", "뮤지컬 관람", "연극 관람", "발레 관람", "오페라 관람", "콘서트 관람",
    "미술 전시회 관람", "박물관 방문", "역사 유적지 탐방", "문화재 답사", "건축물 탐방", "정원 방문",
    "식물원 방문", "동물원 방문", "수족관 방문", "천문대 방문", "과학관 방문", "테마파크 방문", "놀이공원 방문",
    "쇼핑", "패션", "뷰티", "인테리어", "가드닝", "원예", "반려동물 키우기", "반려식물 키우기", "DIY 공예",
    "뜨개질", "자수", "퀼트", "도자기", "목공", "가죽 공예", "종이 공예", "캘리그라피", "레터링",
    "스크랩북 만들기", "다이어리 꾸미기", "스티커 수집", "우표 수집", "동전 수집", "인형 수집", "피규어 수집"
]

# 1. 대표 유저 프로필 데이터 (recent_conversation_history를 Q/A 형태로 수정)
representative_profiles = [
    {
        "user_id": "",
        "name": "",
        "age": 32,
        "job_title": "AI Engineer",
        "specialties": "AI 모델 최적화, MLOps, 데이터 분석",
        "skills": "PyTorch, Docker, Kubernetes, Ray",
        "work_culture": "데이터 기반 문제 해결, 협업과 지식 공유, 자동화 및 효율성 극대화",
        "interests": "Generative AI, AI 모델 서빙 최적화, AI 윤리 문제",
        "report_summary": (
            "논리적인 사고를 바탕으로 문제를 해결하고 최적의 전략을 설계하는 능력이 강함. "
            "최근 프로젝트에서 AI 모델 성능 최적화를 맡으며 복잡한 문제 해결 경험을 쌓음. "
            "또한, 새로운 기술을 배우고 도전하는 것을 즐기며, 성장 과정에서 큰 만족감을 느낌."
        ),
        "recent_conversation_history": (
            "Question: 최근 팀에서 맡은 역할 중에서 가장 도전적이었던 건 뭐야?\n"
            "Answer: 최근 프로젝트에서 성능 최적화를 맡았는데, 예상보다 복잡한 문제들이 많아서 해결하는 데 시간이 걸렸어.\n"
            "Question: 문제 해결 과정에서 어떤 점을 배웠어?\n"
            "Answer: 병목 현상을 해결하면서 시스템 전반을 깊이 이해하는 게 중요하다는 걸 배웠어.\n"
            "Question: 이번 경험을 통해 어떤 기술에 더 관심이 생겼어?\n"
            "Answer: 분산 시스템과 최적화 기법에 더 관심이 생겼어.\n"
            "Question: 실제 프로젝트에 어떻게 적용할 계획이야?\n"
            "Answer: AI 모델 서빙 최적화에 적용해볼 생각이야.\n"
            "Question: 사용하는 기술 스택은 무엇이야?\n"
            "Answer: TensorRT와 ONNX Runtime을 활용해 최적화를 진행하고 있어."
        )
    },
    {
        "user_id": "",
        "name": "",
        "age": 29,
        "job_title": "Data Scientist",
        "specialties": "데이터 분석, 머신러닝, 통계",
        "skills": "TensorFlow, Keras, Docker",
        "work_culture": "협업, 소통, 유연한 근무환경",
        "interests": "새로운 기술, 데이터 시각화, 사용자 경험",
        "report_summary": (
            "창의적 사고와 협업을 통해 지속적으로 발전하는 데이터 사이언티스트로, "
            "새로운 아이디어를 적극적으로 탐색하며, 팀워크를 중시함."
        ),
        "recent_conversation_history": (
            "Question: 최근 데이터 전처리 과정에서 어떤 오류를 만났어?\n"
            "Answer: 예상치 못한 오류를 해결하며 새로운 분석 기법의 필요성을 느꼈어.\n"
            "Question: 모델 튜닝 과정에서 어떤 방법을 시도했어?\n"
            "Answer: 다양한 하이퍼파라미터 최적화 방법을 실험해봤어.\n"
            "Question: 팀과의 소통은 어땠어?\n"
            "Answer: 분석 결과를 팀과 공유하며 피드백을 받아 모델 성능을 개선했어.\n"
            "Question: 데이터 시각화 도구는 어떻게 활용했어?\n"
            "Answer: 결과를 보다 직관적으로 전달할 수 있었어.\n"
            "Question: 최신 머신러닝 기술 동향은 어떻게 반영하고 있어?\n"
            "Answer: 최신 기술 동향을 반영해 프로젝트에 적용하는 방안을 모색 중이야."
        )
    },
    {
        "user_id": "",
        "name": "",
        "age": 35,
        "job_title": "Portfolio Consultant",
        "specialties": "포트폴리오 컨설팅, 문제 해결, 기획",
        "skills": "Excel, PowerPoint",
        "work_culture": "자율적, 창의적, 논리적 사고",
        "interests": "포트폴리오 컨설팅, 창의적 아이디어, 문제 해결",
        "report_summary": (
            "논리적인 사고와 추진력을 바탕으로 포트폴리오 컨설팅에 강점을 가지며, "
            "고객의 성장과 성공을 지원하는 전문가."
        ),
        "recent_conversation_history": (
            "Question: 최근 클라이언트의 요구사항 분석은 어땠어?\n"
            "Answer: 클라이언트의 요구사항을 분석하며 포트폴리오 전반을 재정비했어.\n"
            "Question: 문제 해결 과정에서 어떤 전략을 세웠어?\n"
            "Answer: 여러 대안을 제시하며 전략적 기획을 진행했어.\n"
            "Question: 팀과의 협업 경험은 어땠어?\n"
            "Answer: 팀과의 협업을 통해 프레젠테이션 자료를 개선했고, 만족도를 높였어.\n"
            "Question: 새로운 디자인 트렌드를 어떻게 반영했어?\n"
            "Answer: 트렌드를 반영해 포트폴리오 업데이트 방향을 잡았어.\n"
            "Question: 구체적인 컨설팅 전략은 무엇이야?\n"
            "Answer: 실제 사례 분석을 통해 구체적인 전략을 마련 중이야."
        )
    },
    {
        "user_id": "",
        "name": "",
        "age": 30,
        "job_title": "Software Developer",
        "specialties": "백엔드 개발, 클라우드 컴퓨팅, 시스템 아키텍처",
        "skills": "Java, Spring, AWS, Docker",
        "work_culture": "애자일, 협업, 지속적 통합",
        "interests": "마이크로서비스, 컨테이너화, 서버리스 컴퓨팅",
        "report_summary": (
            "실용적인 문제 해결과 효율적인 시스템 개발을 중시하는 개발자로, "
            "안정적인 클라우드 기반 시스템 구축에 주력하며, 최신 기술 동향에 관심을 가지고 있음."
        ),
        "recent_conversation_history": (
            "Question: 최근 서버 부하 문제를 어떻게 해결했어?\n"
            "Answer: 클라우드 인프라를 재구성하며 문제를 해결했어.\n"
            "Question: 애자일 방법론은 어떻게 적용했어?\n"
            "Answer: 애자일 방법론을 적용하여 팀과의 커뮤니케이션을 강화했어.\n"
            "Question: 백엔드 API 성능 개선은 어떻게 진행했어?\n"
            "Answer: 다양한 아키텍처를 시도하며 성능을 개선했어.\n"
            "Question: 배포 자동화 시스템은 어떻게 구축했어?\n"
            "Answer: Docker와 AWS를 활용한 배포 자동화 시스템을 구축했어.\n"
            "Question: 서버리스 컴퓨팅 도입은 어떤 효과가 있었어?\n"
            "Answer: 비용 절감 효과를 확인 중이야."
        )
    },
    {
        "user_id": "",
        "name": "",
        "age": 28,
        "job_title": "Product Manager",
        "specialties": "제품 전략, UX/UI, 데이터 기반 의사결정",
        "skills": "Figma, Google Analytics, Notion",
        "work_culture": "협업, 혁신, 사용자 중심",
        "interests": "디자인 씽킹, 시장 동향, 디지털 전환",
        "report_summary": (
            "고객 중심의 사고와 창의적인 제품 전략 수립에 강점을 가진 PM으로, "
            "팀 내 협업을 통해 제품의 경쟁력을 강화하고, 사용자 경험을 최우선으로 고려함."
        ),
        "recent_conversation_history": (
            "Question: 최근 제품 기능 개선에 대해 어떻게 접근했어?\n"
            "Answer: 사용자 피드백을 분석하며 개선 방향을 모색했어.\n"
            "Question: 신규 기능 론칭 전 어떤 준비를 했어?\n"
            "Answer: 다양한 시나리오를 테스트하며 전략을 수립했어.\n"
            "Question: 디자인 씽킹 워크샵은 어땠어?\n"
            "Answer: 팀원들과 혁신적인 아이디어를 공유했어.\n"
            "Question: 시장 동향 분석은 어떻게 진행했어?\n"
            "Answer: 시장 동향을 분석해 제품 개선 방향을 재정의했어.\n"
            "Question: UX/UI 개선을 위한 구체적인 계획은 뭐야?\n"
            "Answer: 사용자 경험을 극대화하기 위한 개선안을 도출 중이야."
        )
    }
]

# 2. Example Prompt Template (대표 프로필 예시 포함)
example_prompt_template = """
아래의 대표 유저 프로필 JSON 데이터를 참고하여 동일한 포맷으로 완전히 새롭고 독창적인 가상 유저 프로필을 생성해줘.

중요: 이전에 생성된 프로필과 완전히 다른 프로필을 만들어야 합니다. 다음 사항을 반드시 지켜주세요:
1. 나이는 20-50 사이에서 다양하게 설정하세요. 같은 나이를 반복하지 마세요.
2. 직업은 다양한 분야({jobs})에서 선택하세요.
3. 관심사와 기술도 직업에 맞게 다양하게 설정하세요. 관심사는 ({interests})를 참고하세요
4. 대화 내용도 해당 직업과 관심사에 맞게 구성하세요.

각 유저 프로필은 아래 필드를 포함해야 해:
- user_id (유니크한 임의의 UUID)
- name (실제처럼 보이는 한글 이름, 다양한 성씨 사용)
- age (정수, 20-50 사이, 다양하게 설정)
- job_title (다양한 직업군에서 선택)
- specialties (콤마로 구분된 문자열)
- skills (콤마로 구분된 문자열)
- work_culture (콤마로 구분된 문자열)
- interests (콤마로 구분된 문자열, 직업과 관련된 것과 취미 모두 포함)
- report_summary (문장 형태의 요약)
- recent_conversation_history (최근 대화 내역: 여러 Q/A 쌍으로 구성)

참고 프로필:
{rep_profiles}

생성된 프로필들은 JSON 배열 형식으로 출력해줘.
"""

# 3. Initialize Langchain Google Generative AI LLM
llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=1.0,  # 최대 온도 사용 (Gemini API는 0.0-1.0 범위만 허용)
            api_key=os.getenv('GOOGLE_API_KEY'),
        ).with_structured_output(UserData)


# Helper function for exponential backoff retry
def retry_with_exponential_backoff(func, max_retries=5, initial_delay=1, max_delay=60):
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            # Add jitter to delay to prevent all threads hitting API at same time
            jitter = random.uniform(0.8, 1.2)
            actual_delay = delay * jitter
            
            if retries > 0:
                print(f"Retry attempt {retries}/{max_retries} after {actual_delay:.2f}s delay")
                time.sleep(actual_delay)
                
            return func()
            
        except Exception as e:
            if "429 Resource has been exhausted" in str(e) and retries < max_retries:
                # Only retry on rate limit errors
                retries += 1
                # Exponential backoff: double the delay with each retry
                delay = min(delay * 2, max_delay)
                continue
            else:
                # For other errors or when max retries exceeded, just raise
                print(f"Error (retry {retries}/{max_retries}): {e}")
                raise

# 4. Function to generate a single user profile using Google Generative AI
def generate_single_profile():
    def _generate_profile():
        global used_names
        # Create a new LLM instance for each process to avoid sharing
        local_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=1.0,
            api_key=os.getenv('GOOGLE_API_KEY'),
        ).with_structured_output(UserData)
        
        # 대표 프로필을 랜덤하게 선택하여 다양한 참조 예시 제공
        selected_profiles = random.sample(representative_profiles, min(2, len(representative_profiles)))
        rep_profiles_json = json.dumps(selected_profiles, ensure_ascii=False, indent=2)
        
        # 기존 프로필 정보를 프롬프트에 추가
        custom_prompt = example_prompt_template
        if diversity_prompt_addition:
            custom_prompt += diversity_prompt_addition
        # Format the prompt with all required placeholders
        prompt = custom_prompt.format(
            rep_profiles=rep_profiles_json,
            jobs=', '.join(jobs),
            interests=', '.join(interests)
        )
        
        # Call API to generate a single profile
        # Since we're using structured output, this will directly return a UserData object
        response = local_llm.invoke(prompt)
        
        # Convert the UserData object to a dictionary using the newer method
        profile = response.model_dump()
        profile["name"] = generate_unique_name(used_names)
        profile["user_id"] = str(uuid.uuid4())

        return profile
        
    try:
        # Use our retry mechanism to handle rate limiting
        return retry_with_exponential_backoff(_generate_profile)
    except Exception as e:
        print(f"Failed to generate profile after retries: {e}")
        return None

# 5. Function to handle profile generation with progress bar
def generate_with_progress(total_profiles):
    # Using fewer worker threads to avoid rate limiting
    # Google AI API has rate limits that we need to respect
    max_workers = 3  # Reduced from 8 to avoid rate limits
    results = []
    
    # Setup progress bar
    pbar = tqdm(total=total_profiles, desc="Generating user profiles")
    
    # Keep track of active tasks to throttle submission
    active_tasks = set()
    completed_tasks = 0
    # used_names is now global, no need to redefine it here
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit initial batch of tasks
        for i in range(min(max_workers, total_profiles)):
            future = executor.submit(generate_single_profile)
            active_tasks.add(future)
        
        # Process tasks as they complete and submit new ones
        while active_tasks and completed_tasks < total_profiles:
            # Wait for the next task to complete
            done, active_tasks = concurrent.futures.wait(
                active_tasks, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                profile = None
                try:
                    profile = future.result()
                    if profile is not None:
                        results.append(profile)
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
                
                # Update progress
                completed_tasks += 1
                pbar.update(1)
                
                # Submit a new task if needed
                if completed_tasks < total_profiles:
                    # Add a longer delay between API calls to avoid rate limiting
                    # Using a random delay between 1-2 seconds to avoid synchronized requests
                    delay = random.uniform(1.0, 2.0)
                    time.sleep(delay)  
                    new_future = executor.submit(generate_single_profile)
                    active_tasks.add(new_future)
    
    pbar.close()
    return results

# Generate profiles using multithreading with rate limiting
# Limiting the number of profiles to generate to avoid hitting API limits
total_profiles_to_generate = 100  # API 할당량을 고려하여 적은 수의 프로필 생성

# 이미 생성된 프로필이 있는지 확인
# 이를 통해 이전에 생성된 프로필과 중복되지 않도록 함
# 데이터 디렉토리 경로 설정
data_dir = 'data/profiles'
os.makedirs(data_dir, exist_ok=True)

generated_profiles_path = os.path.join(data_dir, 'generated_virtual_users_v1.json')
existing_profiles = []

# 프롬프트 업데이트를 위한 변수 선언
diversity_prompt_addition = ""

try:
    if os.path.exists(generated_profiles_path):
        with open(generated_profiles_path, 'r', encoding='utf-8') as f:
            existing_profiles = json.load(f)
        print(f"\n이미 {len(existing_profiles)}개의 프로필이 있습니다.")
        
        # 기존 프로필의 이름과 직업 확인
        existing_names = [profile.get('name', '') for profile in existing_profiles]
        existing_jobs = [profile.get('job_title', '') for profile in existing_profiles]
        print(f"\n기존 이름: {', '.join(existing_names[:5])}{'...' if len(existing_names) > 5 else ''}")
        print(f"\n기존 직업: {', '.join(existing_jobs[:5])}{'...' if len(existing_jobs) > 5 else ''}")
        
        # 프롬프트에 추가할 정보 준비
        diversity_prompt_addition = f"\n\n기존 이름: {', '.join(existing_names[:10])}\n기존 직업: {', '.join(existing_jobs[:10])}\n\n위에 있는 기존 이름과 직업을 절대 사용하지 마세요. 완전히 새로운 이름과 직업을 사용해야 합니다."
        
except Exception as e:
    print(f"\n기존 프로필 로드 오류: {e}")

all_generated_profiles = generate_with_progress(total_profiles_to_generate)

# 6. Save the generated profiles to a JSON file
# 기존 프로필이 있으면 새로 생성된 프로필과 합쳐서 저장
if existing_profiles:
    # 기존 프로필과 새로 생성된 프로필 합치기
    combined_profiles = existing_profiles + all_generated_profiles
    print(f"\n기존 프로필 {len(existing_profiles)}개와 새로 생성된 프로필 {len(all_generated_profiles)}개를 합쳐서 저장합니다.")
    
    with open(generated_profiles_path, "w", encoding="utf-8") as f:
        json.dump(combined_profiles, f, ensure_ascii=False, indent=2)
    
    print(f"총 저장된 가상 사용자: {len(combined_profiles)}명")
else:
    # 새로 생성된 프로필만 저장
    with open(generated_profiles_path, "w", encoding="utf-8") as f:
        json.dump(all_generated_profiles, f, ensure_ascii=False, indent=2)
    
    print(f"총 생성된 가상 사용자: {len(all_generated_profiles)}명")

print(f"Total generated virtual users: {len(all_generated_profiles)}")