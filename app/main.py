# app/main.py
import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Any

# --- CẤU HÌNH ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_INDEX_NAME = "egov-chatbot"
EMBED_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
GENAI_MODEL_NAME = "gemini-2.5-flash"
ARTIFACTS_DIR = "/app/artifacts"
ID_MAP_PATH = f"{ARTIFACTS_DIR}/id_to_record.pkl"

# --- TẢI TÀI NGUYÊN KHI KHỞI ĐỘNG ---
RAG_RESOURCES = {}

def load_resources():
    print("Khởi động server: Đang tải tài nguyên...")
    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        raise ValueError("API Keys chưa được thiết lập!")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    RAG_RESOURCES["pinecone_index"] = pc.Index(PINECONE_INDEX_NAME)
    RAG_RESOURCES["embedder"] = SentenceTransformer(EMBED_MODEL_NAME)
    
    with open(ID_MAP_PATH, "rb") as f:
        RAG_RESOURCES["id_map"] = pickle.load(f)
        
    genai.configure(api_key=GEMINI_API_KEY)
    RAG_RESOURCES["genai_model"] = genai.GenerativeModel(GENAI_MODEL_NAME)
    print("✅ Tải tài nguyên thành công!")

app = FastAPI(title="eGovernment Chatbot API")

@app.on_event("startup")
async def startup_event():
    load_resources()

# --- HÀM TRUY VẤN PINECONE ---
def retrieve_from_pinecone(query: str, top_k=5):
    embedder = RAG_RESOURCES["embedder"]
    index = RAG_RESOURCES["pinecone_index"]
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results.get('matches', [])

# --- API MODELS ---
class QueryRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

# --- API ENDPOINT CHÍNH ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    query = request.question
    
    retrieved_docs = retrieve_from_pinecone(query, top_k=5)

    if not retrieved_docs:
        return AnswerResponse(answer="Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn.", sources=[])

    context = ""
    sources = set()
    for doc in retrieved_docs:
        metadata = doc.get('metadata', {})
        raw_text = metadata.get('raw_text', '')
        title = metadata.get('title', 'Không rõ')
        field = metadata.get('field', 'Không rõ')
        parent_id = metadata.get('parent_id')

        context += f"Trích đoạn từ thủ tục '{title}', mục '{field}':\n{raw_text}\n\n"
        if parent_id:
            sources.add(parent_id)

    prompt = (
        "Bạn là một trợ lý AI chuyên về thủ tục hành chính Việt Nam. "
        "Chỉ dựa vào các trích đoạn được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ.\n\n"
        f"--- BỐI CẢNH ---\n{context}"
        f"--- CÂU HỎI ---\n{query}\n\n"
        "--- TRẢ LỜI ---"
    )

    genai_model = RAG_RESOURCES["genai_model"]
    try:
        response = genai_model.generate_content(prompt)
        final_answer = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi mô hình Gemini: {e}")

    # Thêm nguồn vào cuối câu trả lời nếu có
    if sources:
        final_answer += "\n\n**Nguồn tham khảo:**\n" + "\n".join(f"- {s}" for s in sorted(list(sources)))

    return AnswerResponse(answer=final_answer, sources=list(sources))

@app.get("/")
def read_root():
    return {"status": "eGovernment Chatbot API is running"}