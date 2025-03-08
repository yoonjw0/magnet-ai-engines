import json
# from langchain import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import time
import os
import random
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Update deprecated import
from pydantic import BaseModel, Field
class UserData(BaseModel):
    user_id: str = Field(description="unique random UUID")
    name: str = Field(description="name like real")
    age: int = Field(description="age, between 20 and 50")
    job_title: str = Field(description="job title")
    specialties: str = Field(description="specialties, comma separated")
    skills: str = Field(description="skills, comma separated")
    work_culture: str = Field(description="work culture, comma separated")
    interests: str = Field(description="interests, comma separated")
    report_summary: str = Field(description="report summary")
    recent_conversation_history: str = Field(description="recent conversation history")

# 1. 대표 유저 프로필 데이터 (recent_conversation_history를 Q/A 형태로 수정)
representative_profiles = [
    {
        "user_id": "b7f3c8d1-92e5-4d3f-9b8e-63f27f123456",
        "name": "김민수",
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
        "user_id": "user2-unique-id",
        "name": "에스님",
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
        "user_id": "user3-unique-id",
        "name": "엔멘토",
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
        "user_id": "user4-unique-id",
        "name": "정우",
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
        "user_id": "user5-unique-id",
        "name": "박지은",
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
1. 이름은 다양한 성(김, 이, 박, 최, 정, 강, 조, 윤, 장, 임 등)과 이름을 사용하세요.
2. 나이는 20-50 사이에서 다양하게 설정하세요. 같은 나이를 반복하지 마세요.
3. 직업은 다양한 분야(IT, 금융, 교육, 의료, 예술, 엔터테인먼트, 요식업, 제조업 등)에서 선택하세요.
4. 관심사와 기술도 직업에 맞게 다양하게 설정하세요.
5. 대화 내용도 해당 직업과 관심사에 맞게 구성하세요.

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
            convert_system_message_to_human=True,
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
        # Create a new LLM instance for each process to avoid sharing
        local_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=1.0,  # 최대 온도 사용 (Gemini API는 0.0-1.0 범위만 허용)
            convert_system_message_to_human=True,
            api_key=os.getenv('GOOGLE_API_KEY'),
        ).with_structured_output(UserData)
        
        # 대표 프로필을 랜덤하게 선택하여 다양한 참조 예시 제공
        selected_profiles = random.sample(representative_profiles, min(2, len(representative_profiles)))
        rep_profiles_json = json.dumps(selected_profiles, ensure_ascii=False, indent=2)
        
        # 기존 프로필 정보를 프롬프트에 추가
        custom_prompt = example_prompt_template
        if diversity_prompt_addition:
            custom_prompt += diversity_prompt_addition
        prompt = custom_prompt.format(rep_profiles=rep_profiles_json)
        
        # Call API to generate a single profile
        # Since we're using structured output, this will directly return a UserData object
        response = local_llm.invoke(prompt)
        
        # Convert the UserData object to a dictionary using the newer method
        profile = response.model_dump()
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

generated_profiles_path = os.path.join(data_dir, 'generated_virtual_users.json')
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