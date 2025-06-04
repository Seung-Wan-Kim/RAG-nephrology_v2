from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os

# 수정된 경로
TEXT_PATH = "data/docs/aki_guide_ko_summary.txt"
VECTOR_STORE_DIR = "vector_store_aki_ko"

# 문서 확인
if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(f"❌ 문서를 찾을 수 없습니다: {TEXT_PATH}")

# 문서 로딩
loader = TextLoader(TEXT_PATH, encoding="utf-8")
documents = loader.load()

# 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# OpenAI 임베딩
embedding_model = OpenAIEmbeddings(openai_api_key="sk-...")  # ← 실제 키로 입력

# FAISS 벡터 생성 및 저장
db = FAISS.from_documents(docs, embedding_model)
db.save_local(VECTOR_STORE_DIR)

# index.pkl 수동 저장
with open(os.path.join(VECTOR_STORE_DIR, "index.pkl"), "wb") as f:
    pickle.dump(db, f)

print("✅ 벡터 저장 완료 (index.faiss, index.pkl)")

