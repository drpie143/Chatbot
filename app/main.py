# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import os
import pickle
import gzip
import re
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from typing import List, Dict, Any

# --- 2. CẤU HÌNH ---
# Lấy API Key từ biến môi trường để bảo mật
API_KEY = os.environ.get("GEMINI_API_KEY")

# Tên model và đường dẫn artifacts bên trong Docker container
EMBED_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
GENAI_MODEL_NAME = "gemini-1.5-flash"
ARTIFACTS_DIR = "/app/artifacts"

FAISS_PATH = f"{ARTIFACTS_DIR}/index.faiss"
METAS_PATH = f"{ARTIFACTS_DIR}/metas.pkl.gz"
BM25_PATH = f"{ARTIFACTS_DIR}/bm25.pkl.gz"
ID_MAP_PATH = f"{ARTIFACTS_DIR}/id_to_record.pkl"  # File tra cứu nhanh

# --- 3. TẢI TÀI NGUYÊN MỘT LẦN KHI KHỞI ĐỘNG ---
RAG_RESOURCES = {}  # Kho chứa các tài nguyên đã tải

def load_resources():
    """Tải tất cả các model và file index vào bộ nhớ một lần duy nhất."""
    print("Khởi động server: Đang tải các tài nguyên RAG...")

    RAG_RESOURCES["embedder"] = SentenceTransformer(EMBED_MODEL_NAME)
    RAG_RESOURCES["faiss_index"] = faiss.read_index(FAISS_PATH)

    with gzip.open(METAS_PATH, "rb") as f:
        RAG_RESOURCES["metadatas"] = pickle.load(f)
    with gzip.open(BM25_PATH, "rb") as f:
        RAG_RESOURCES["bm25"] = pickle.load(f)
    with open(ID_MAP_PATH, "rb") as f:
        RAG_RESOURCES["id_map"] = pickle.load(f)

    if not API_KEY:
        raise ValueError("API Key của Gemini chưa được thiết lập!")
    genai.configure(api_key=API_KEY)
    RAG_RESOURCES["genai_model"] = genai.GenerativeModel(GENAI_MODEL_NAME)

    print("✅ Tải tài nguyên RAG thành công!")

# --- 4. KHỞI TẠO ỨNG DỤNG FASTAPI ---
app = FastAPI(title="eGovernment Chatbot API")

@app.on_event("startup")
async def startup_event():
    load_resources()

# --- 5. CÁC HÀM LOGIC ---
def minmax_scale(arr):
    arr = np.array(arr, dtype="float32")
    if len(arr) == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-6:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)

def retrieve(query: str, top_k=10, w_vec=0.7, w_bm25=0.3):
    faiss_index = RAG_RESOURCES["faiss_index"]
    bm25_loaded = RAG_RESOURCES["bm25"]
    embedder = RAG_RESOURCES["embedder"]

    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(qv, top_k * 5)
    vec_scores, vec_idx = D[0].tolist(), I[0].tolist()

    tokenized_query = query.split()
    bm25_scores_all = bm25_loaded.get_scores(tokenized_query)
    bm25_top_idx = np.argsort(-bm25_scores_all)[:top_k * 5].tolist()

    union_idx = list(dict.fromkeys(vec_idx + bm25_top_idx))
    vec_map = {i: s for i, s in zip(vec_idx, vec_scores)}
    vec_list = [vec_map.get(i, 0.0) for i in union_idx]
    bm25_list = [bm25_scores_all[i] for i in union_idx]

    vec_scaled = minmax_scale(vec_list)
    bm25_scaled = minmax_scale(bm25_list)
    fused = w_vec * vec_scaled + w_bm25 * bm25_scaled
    order = np.argsort(-fused)

    results = [(union_idx[i], float(fused[i])) for i in order[:top_k]]
    return results

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

def get_context_from_results(results: list) -> tuple[str, list]:
    """Tạo bối cảnh và thu thập nguồn từ kết quả truy hồi."""
    context = ""
    sources = set()
    metadatas = RAG_RESOURCES["metadatas"]

    unique_parent_ids = set()
    for res_id, score in results:
        parent_id = metadatas[res_id].get("parent_id")
        if parent_id and parent_id not in unique_parent_ids:
            full_text = get_full_procedure_text_by_id(parent_id)
            if full_text:
                context += f"--- TRÍCH ĐOẠN THỦ TỤC LIÊN QUAN ---\n"
                context += full_text + "\n\n"
                sources.add(parent_id)
                unique_parent_ids.add(parent_id)

    return context, list(sources)

# --- 6. ĐỊNH NGHĨA API MODELS ---
class QueryRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

# --- 7. API ENDPOINT CHÍNH ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    query = request.question

    # 1. Truy hồi thông tin (Retrieval)
    retrieved_results = retrieve(query, top_k=3)

    if not retrieved_results:
        return AnswerResponse(answer="Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn.", sources=[])

    # 2. Tạo bối cảnh và lấy nguồn (Augmentation)
    context, sources = get_context_from_results(retrieved_results)

    # 3. Xây dựng Prompt cuối cùng
    prompt = (
        "Bạn là một trợ lý AI chuyên về thủ tục hành chính Việt Nam. "
        "Trả lời tiếng Việt, chính xác, dựa hoàn toàn vào DỮ LIỆU bao gồm các thủ tục liên quan. "
        "Trình bày gọn, có gạch đầu dòng nếu phù hợp, và luôn đính kèm các Nguồn (đường link) xuất hiện trong dữ liệu ở cuối. "
        "Nếu thông tin không đủ, hãy nói rằng bạn không tìm thấy thông tin trong các tài liệu được cung cấp.\n\n"
        f"--- BỐI CẢNH ---\n{context}"
        f"--- CÂU HỎI CỦA NGƯỜI DÙNG ---\n{query}\n\n"
        "--- TRẢ LỜI ---"
    )

    # 4. Gọi mô hình Gemini để sinh câu trả lời (Generation)
    genai_model = RAG_RESOURCES["genai_model"]
    try:
        response = genai_model.generate_content(prompt)
        final_answer = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi mô hình Gemini: {e}")

    return AnswerResponse(answer=final_answer, sources=sources)

@app.get("/")
def read_root():
    return {"status": "eGovernment Chatbot API is running"}