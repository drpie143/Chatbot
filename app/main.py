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

# --- HÀM TRUY VẤN PINECONE (GIỮ NGUYÊN) ---
def retrieve_from_pinecone(query: str, top_k=5):
    embedder = RAG_RESOURCES["embedder"]
    index = RAG_RESOURCES["pinecone_index"]
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results.get('matches', [])

# --- HÀM MỚI ĐỂ LẤY NỘI DUNG ĐẦY ĐỦ TỪ ID ---
def get_full_procedure_text_by_id(input_id: str) -> str:
    """Hàm tra cứu dữ liệu thô SIÊU NHANH từ file id_to_record.pkl"""
    field_map = {
        "ten_thu_tuc": "Tên thủ tục",
        "cach_thuc_thuc_hien": "Cách thức thực hiện",
        "thanh_phan_ho_so": "Thành phần hồ sơ",
        "trinh_tu_thuc_hien": "Trình tự thực hiện",
        "co_quan_thuc_hien": "Cơ quan thực hiện",
        "yeu_cau_dieu_kien": "Yêu cầu, điều kiện",
        "thu_tuc_lien_quan": "Thủ tục liên quan",
    }
    record = RAG_RESOURCES["id_map"].get(input_id)
    if not record:
        return ""
    
    parts = []
    for k, v in record.items():
        if v and k in field_map:
            parts.append(f"**{field_map[k]}**:\n{str(v).strip()}")
    return "\n\n".join(parts)


# --- API MODELS ---
class QueryRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

# --- API ENDPOINT CHÍNH (ĐÃ SỬA LỖI LOGIC) ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    query = request.question
    
    retrieved_docs = retrieve_from_pinecone(query, top_k=5)

    if not retrieved_docs:
        return AnswerResponse(answer="Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn.", sources=[])

    context = ""
    sources = set()
    unique_parent_ids = set() # Dùng để tránh lấy trùng lặp nội dung
    
    for doc in retrieved_docs:
        metadata = doc.get('metadata', {})
        parent_id = metadata.get('parent_id')

        # Lấy toàn bộ nội dung từ file id_to_record.pkl
        if parent_id and parent_id not in unique_parent_ids:
            full_text = get_full_procedure_text_by_id(parent_id)
            if full_text:
                context += f"--- TRÍCH ĐOẠN THỦ TỤC LIÊN QUAN ---\n"
                context += full_text + "\n\n"
                sources.add(parent_id)
                unique_parent_ids.add(parent_id)
    
    # Nếu không có bối cảnh từ ID, dùng tạm metadata
    if not context:
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            context += f"Trích đoạn từ thủ tục '{metadata.get('title', 'N/A')}', mục '{metadata.get('field', 'N/A')}':\n{metadata.get('raw_text', '')}\n\n"
            if metadata.get('parent_id'):
                sources.add(metadata.get('parent_id'))
                
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