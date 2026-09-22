# Production-Ready FastAPI RAG Chatbot

`rag-chatbot/.antigravity/rules.md`에 정의된 20년 차 엔지니어링 아키텍트 원칙(유지보수성, 가용성, 장애 격리, 예측 가능한 리소스 소모)을 준수하여 구현된 프로덕션 수준의 RAG(Retrieval-Augmented Generation) 챗봇 백엔드입니다.

---

## 🏗️ 아키텍처 개요

```
[Client]
   │
   ▼ HTTP / SSE
[RequestContextMiddleware] ── X-Request-ID 발급 & Latency/JSON 구조화 로깅
   │
   ▼
[API Router (/api/v1)] ──── FastAPI Depends ────▶ [Service Layer]
   │                                                     │
   ├── /health & /health/ready                           ├── DocumentService (SHA-256 멱등성 청킹)
   ├── /documents (text, file, stats)                    └── RAGService (임계치 폴백, 토큰 예산 제어)
   └── /chat & /chat/stream (SSE)                                │
                                         ┌───────────────────────┴───────────────────────┐
                                         ▼                                               ▼
                              [ChromaVectorStore]                               [BaseLLMClient]
                               anyio.to_thread 래핑                              (Gemini / Ollama / Mock)
```

### 적용된 핵심 엔지니어링 원칙 (`rules.md`)

1. **계층형 단방향 의존성 및 DI (`rules.md` §1.1)**:
   - `Router` -> `Service` -> `VectorStore / LLMClient` 단방향 의존성 유지.
   - `FastAPI.Depends()`를 통해 서비스 인스턴스를 주입받으며 핸들러 내 무거운 객체 인스턴스화 배제.
2. **FastAPI `lifespan` 컨텍스트 매니저 (`rules.md` §1.2)**:
   - `on_event`를 사용하지 않고 `lifespan`에서 ChromaDB 및 외부 세션을 1회 초기화 후 `app.state`에 캐싱.
   - 앱 종료 시 Graceful Shutdown 처리.
3. **이벤트 루프 블로킹 원천 방지 (`rules.md` §2.1)**:
   - ChromaDB I/O 및 CPU 바운드 청킹 작업을 `anyio.to_thread.run_sync()`로 워커 스레드에 완전 오프로딩.
4. **SSE 스트리밍 & 클라이언트 연결 끊김 감지 (`rules.md` §2.2)**:
   - `/api/v1/chat/stream` 엔드포인트에서 청크 생성 시 `await request.is_disconnected()`를 체크하여 클라이언트 이탈 시 즉시 생성 취소(비용 누수 방지).
   - SSE 표준 규격 (`data: {...}\n\n`, 에러 시 `event: error\ndata: ...\n\n`) 준수.
5. **RAG 파이프라인 가드레일 (`rules.md` §3)**:
   - **Context Fallback**: 코사인 유사도 점수가 기준치(`SIMILARITY_THRESHOLD`) 미만일 경우 기본 사전문구("해당 문서를 찾을 수 없습니다.") 반환 (환각 방지).
   - **Context Window Overflow 방지**: 토큰 예산(`MAX_CONTEXT_TOKENS`)을 사전 계산하여 초과 청크 Truncate.
   - **SHA-256 멱등성**: 문서 수집 시 파일/본문 해시를 검증하여 중복 임베딩 및 적재 방지.
6. **일관된 예외 및 관측성 (`rules.md` §4)**:
   - 커스텀 비즈니스 예외(`AppException`) 및 통일된 JSON 에러 규격.
   - JSON 구조화 로깅, `X-Request-ID` 추적, API Key 및 PII 마스킹.
7. **설정 및 컨테이너 보안 (`rules.md` §5)**:
   - `pydantic-settings` strict type validation.
   - Non-root `appuser` (UID 10001) 및 Multi-stage Dockerfile.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 변수 설정 (`.env`)

`.env.example`을 복사하여 `.env`를 생성합니다.

```bash
cp .env.example .env
```

#### 모드별 설정 가이드:

- **1) 로컬 개발 & 테스트 (Mock 모드 - 기본값, 외부 의존성 없음)**:
  ```env
  LLM_PROVIDER=mock
  ```
- **2) Google Gemini SaaS 모드**:
  ```env
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=your_gemini_api_key
  GEMINI_MODEL_NAME=gemini-3.8-flash
  GEMINI_EMBEDDING_MODEL=gemini-embedding-001
  ```
- **3) 로컬 LLM 모드 (테스트베드 - Ollama, vLLM 등 OpenAI 호환)**:
  ```env
  LLM_PROVIDER=openai_compatible
  OPENAI_BASE_URL=http://localhost:11434/v1
  OPENAI_API_KEY=ollama
  OPENAI_MODEL_NAME=llama3.1:8b
  OPENAI_EMBEDDING_MODEL=nomic-embed-text
  ```

---

### 2. 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- 대화형 Swagger API 문서: `http://localhost:8000/docs`
- 헬스 체크: `http://localhost:8000/api/v1/health`

---

### 3. Docker로 실행

```bash
# Docker Compose 실행 (ChromaDB 데이터는 ./data/chroma에 영속화)
docker compose up -d --build

# 로그 확인
docker compose logs -f
```

---

## 🧪 테스트 실행

```bash
pytest -v
```

12개의 통합/단위 테스트(Liveness/Readiness, SHA-256 멱등성, SSE 스트리밍, 클라이언트 연결 끊김 중단, 임계치 폴백, 토큰 버짓 절삭, 에러 응답 규격)가 포함되어 있습니다.

---

## 📡 API 사용 예시

### 1. 텍스트 문서 등록 (멱등성 보장)
```bash
curl -X POST "http://localhost:8000/api/v1/documents/text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "사내 규정",
    "content": "재택근무는 주 2회까지 신청 가능하며, 부서장의 사전 승인이 필요합니다."
  }'
```

동일한 내용을 한 번 더 전송하면 중복 적재되지 않고 `"is_duplicate": true`와 기존 `document_id`가 반환됩니다.

### 2. RAG 질문 (단일 응답)
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "재택근무는 며칠까지 가능한가요?",
    "top_k": 3,
    "score_threshold": 0.65
  }'
```

### 3. RAG 질문 (SSE 실시간 스트리밍)
```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "재택근무 신청 절차는 어떻게 되나요?"
  }'
```
응답:
```
data: {"text": "재택", "done": false, "fallback_triggered": false, "retrieved_chunks": null}

data: {"text": "근무는", "done": false, "fallback_triggered": false, "retrieved_chunks": null}
...
data: {"text": null, "done": true, "fallback_triggered": false, "retrieved_chunks": [...]}
```
