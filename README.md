# Magnet AI Engines

## 프로젝트 개요
이 저장소는 밋업 매칭 추천 시스템을 포함한 AI 엔진의 구현을 담고 있습니다. 사용자 프로필 기반의 유사도 측정을 통해 최적의 매칭을 추천하고, 이를 시각화하는 도구를 제공합니다.

## 주요 기능
- **QnA 대화 생성**: 사용자 맞춤형 질문 생성
- **밋업 매칭 추천**: 사용자 프로필 기반 유사도 측정 및 추천
- **시각화**: 다양한 방식의 관계 그래프 및 추천 결과 시각화
- **추천 설명 생성**: LLM을 활용한 맞춤형 추천 이유 설명
- **가상 사용자 시뮬레이션**: 테스트를 위한 가상 사용자 생성 및 시뮬레이션

## 핵심 모듈
- `simulation_virtual_users.py`: 가상 사용자 생성 및 시뮬레이션
- `meetup_recommendation.py`: 밋업 매칭 시뮬레이션
- `run_visualization.py`: 시각화 실행 스크립트
- `visualization/`: 시각화 모듈 패키지

## 프로젝트 구조

```
/
├── architecture_design/   # 아키텍처 설계 문서
├── data/                  # 데이터 파일
│   └── profiles/          # 사용자 프로필 데이터
├── cached_data/           # 캐시 데이터 저장소
├── visualization_results/ # 시각화 결과
├── visualization_cache/   # 시각화 캐시 디렉토리
├── tests/                 # 테스트 코드
├── visualization/         # 시각화 패키지
│   ├── __init__.py        # 패키지 초기화 파일
│   ├── utils.py           # 유틸리티 함수
│   ├── visualizer.py      # 메인 시각화 클래스
│   ├── network_graphs.py  # 네트워크 그래프 시각화 모듈
│   ├── feature_graphs.py  # 특성 그래프 시각화 모듈
│   ├── embedding_viz.py   # 임베딩 시각화 모듈
│   └── recommendation_viz.py # 추천 결과 시각화 모듈
├── simulation_virtual_users.py  # 가상 사용자 생성 및 시뮬레이션
├── meetup_recommendation.py     # 밋업 매칭 시뮬레이션
└── run_visualization.py         # 시각화 실행 스크립트
```

## 시작하기

### 필요 환경
- Python 3.10+
- 필요 패키지: numpy, pandas, scikit-learn, langchain-google-genai, matplotlib, networkx, plotly

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/yourusername/magnet-ai-engines.git
cd magnet-ai-engines

# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

### 사용 예시
```python
# 밋업 추천 시스템 사용
from meetup_recommendation import MeetupRecommendationSystem

recommender = MeetupRecommendationSystem()
recommender.run_recommendation_pipeline()

# 시각화 도구 사용
from visualization import MeetupVisualization

visualizer = MeetupVisualization(recommender)
visualizer.plot_force_directed_graph(n_users=50, threshold=0.75)
```

## 시각화 도구 사용하기

### 명령행에서 실행

```bash
# 모든 시각화 생성
python run_visualization.py --user 0 --n-users 50 --top-n 5

# 특정 특성 기반 네트워크 시각화
python run_visualization.py --feature "최근 대화" --threshold 0.7 --n-users 50
```

### 코드에서 사용

```python
from meetup_recommendation import MeetupRecommendationSystem
from visualization import MeetupVisualization

# 추천 시스템 인스턴스 생성
recommender = MeetupRecommendationSystem()

# 시각화 인스턴스 생성
visualizer = MeetupVisualization(recommender=recommender)

# 포스 다이렉트 그래프 생성
force_graph = visualizer.plot_force_directed_graph(
    n_users=50, 
    threshold=0.75,
    min_connections=2
)

# 인터랙티브 포스 다이렉트 그래프 생성
interactive_force_graph = visualizer.plot_interactive_force_directed_graph(
    n_users=50, 
    threshold=0.75,
    min_connections=2
)

# 특성별 유사도 분포 시각화
feature_dist = visualizer.plot_feature_similarity_distribution(
    user_idx=0,
    top_n=5
)

# 특정 특성 기반 네트워크 시각화
feature_network = visualizer.plot_feature_specific_network(
    feature_name="최근 대화",
    n_users=50,
    threshold=0.7
)

# 모든 시각화 생성 및 저장
results = visualizer.generate_all_visualizations(
    user_idx=0,
    n_users=50,
    top_n=5,
    output_dir="visualization_results"
)
```

## 시각화 유형

1. **포스 다이렉트 그래프**: 사용자 간 유사도를 기반으로 한 네트워크 그래프
2. **인터랙티브 포스 다이렉트 그래프**: 인터랙티브한 네트워크 그래프 (Plotly 사용)
3. **특성별 유사도 분포**: 특성별 유사도 분포를 보여주는 바 차트
4. **특성 기반 네트워크**: 특정 특성(예: 최근 대화)의 유사도만을 기반으로 한 네트워크 그래프
5. **임베딩 시각화**: 사용자 임베딩을 2D 또는 3D 공간에 시각화
6. **추천 결과 비교**: 추천 결과를 특성별로 비교하는 차트
7. **추천 레이더 차트**: 추천 사용자의 특성을 레이더 차트로 비교

## 캐시 사용

시각화 결과는 기본적으로 `visualization_cache` 디렉토리에 캐시됩니다. 캐시를 사용하지 않으려면 다음과 같이 설정합니다:

```python
visualizer = MeetupVisualization(recommender=recommender, use_cache=False)
```

또는 명령행에서:

```bash
python run_visualization.py --no-cache
```

## 기여하기
프로젝트 기여에 관한 문의는 프로젝트 관리자에게 연락해 주세요.
