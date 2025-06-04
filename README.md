# Nephrology RAG System (AKI 기반 질의응답 + 수치 입력)

이 프로젝트는 신장내과 질환 중 AKI (급성신손상)에 관한 질병 정보와 혈액검사 수치를 기반으로 한 RAG (Retrieval-Augmented Generation) 시스템입니다.

---

## 📁 프로젝트 구성

```
nephro_rag_v2/
│
├── app/
│   └── app_rag.py                  # Streamlit 기반 UI 메인 앱
│
├── data/
│   └── docs/
│       └── aki_guide_ko_summary.txt  # 질의응답용 한글 요약문서
│
├── vector_store_aki_ko/
│   ├── index.faiss                # FAISS 인덱스 파일
│   └── index.pkl                  # 문서 임베딩 메타정보
│
├── create_embeddings_aki_ko.py    # 임베딩 생성 스크립트
├── requirements.txt               # 실행 환경 패키지 리스트
└── README.md                      # 프로젝트 설명
```

---

## ✅ 주요 기능

- AKI 관련 문서 요약본을 바탕으로 GPT 질의응답
- 혈액검사 수치 (20개 항목) 기반 입력 UI
- 유사도 기반 문서 검색 + GPT 답변 제공

---

## 🛠 설치 방법

```bash
# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 필수 패키지 설치
pip install -r requirements.txt
```

---

## ▶️ 실행 방법

```bash
streamlit run app/app_rag.py
```

---

## 📌 참고 사항

- `docs_ko/aki_guide_ko_summary.txt`는 한국어로 요약된 AKI 가이드 문서입니다.
- OpenAI API 키는 `.env` 또는 코드 내부에 직접 지정해야 합니다 (`OPENAI_API_KEY`).
- FAISS 인덱스와 메타정보 파일(index.faiss, index.pkl)은 미리 생성되어 있어야 합니다.

---

## 📬 문의

질문이나 오류 제보는 Issues 탭에 남겨주세요.