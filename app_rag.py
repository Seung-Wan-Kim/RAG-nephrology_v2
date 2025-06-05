import streamlit as st
import pandas as pd
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# 설정
# 현재 저장소 구조상 인덱스 파일(index.faiss, index.pkl)이
# 프로젝트 루트에 위치하므로 해당 경로를 사용한다.
VECTOR_STORE_PATH = "."  # 벡터 스토어가 위치한 디렉토리 경로
OPENAI_API_KEY = "your_api_key_here"  # 필요시 환경변수로 대체

# 전체 수치 항목 목록 (20여 개 항목)
TEST_ITEMS = [
    "BUN", "Creatinine", "B/C ratio", "eGFR", "Na", "K", "Cl", "CO2", "Ca", "IP",
    "Hb", "PTH", "Vitamin D", "ALP", "LDH", "Lactate", "CPK", "C3", "C4"
]

# 문서 기반 QA 시스템 초기화
def initialize_qa():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    result = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    if isinstance(result, tuple):
        db = result[0]
    else:
        db = result

    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    return db, chain

# 수치 입력창 렌더링
def render_numeric_input():
    st.subheader("🧪 검사 수치 입력")
    num_cols = 5
    input_data = {}
    rows = (len(TEST_ITEMS) + num_cols - 1) // num_cols

    for row_idx in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            idx = row_idx * num_cols + col_idx
            if idx < len(TEST_ITEMS):
                item = TEST_ITEMS[idx]
                with cols[col_idx]:
                    val = st.text_input(f"{item}", key=f"{item}")
                    if val:
                        try:
                            input_data[item] = float(val)
                        except ValueError:
                            st.warning(f"⚠️ '{item}'는 숫자만 입력 가능합니다.")
    return input_data

# 문서 기반 질의응답
def rag_answer(db, chain, query):
    docs = db.similarity_search(query, k=3)
    if docs:
        result = chain.run(input_documents=docs, question=query)
        return result
    return "❌ 문서에서 관련된 정보를 찾을 수 없습니다."

# Streamlit UI
def main():
    st.title("🩺 Nephrology RAG 시스템")

    with st.expander("1. 🔬 혈액검사 수치 기반 입력"):
        user_inputs = render_numeric_input()
        if user_inputs:
            st.success(f"✅ 입력된 수치: {json.dumps(user_inputs, indent=2, ensure_ascii=False)}")

    with st.expander("2. 💬 문서 기반 질의응답"):
        db, chain = initialize_qa()
        question = st.text_input("질문을 입력하세요:")
        if st.button("질문하기") and question:
            with st.spinner("답변 생성 중..."):
                answer = rag_answer(db, chain, question)
                st.write("📌 답변:", answer)

if __name__ == "__main__":
    main()
