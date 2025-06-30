Bước 1: Điều hướng đến thư mục Project
cd "thư mục chứa code"
hoặc
mở cmd ở đường link file

Bước 2: Tạo một môi trường ảo mới
python -m venv pdf_rag_env

Bước 3: Kích hoạt môi trường ảo
pdf_rag_env\Scripts\activate

Bước 4: Cài đặt các thư viện cần thiết
* pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
(hoạt động vs NVIDA cuda 12.1)
* pip install bitsandbytes # Quan trọng cho lượng tử hóa GPU
* pip install streamlit transformers accelerate langchain-huggingface langchain-community langchain-experimental langchain-chroma langchain pypdf
* pip install sentence-transformers # vector embeddings, semantic search, semantic textual similarity, paraphrase mining, cross-encoder reranker
* pip install tiktoken sentencepiece # thư viện support conversion

Bước 5: Xác nhận là ở vs code đã nhận môi trường ảo
ở góc phải dưới xác nhận là đã có 3.11.9(pdf_rag_env)
trỏ chuột đến thì hiển thị .\pdf_rag_env\Scripts/python.exe

Các lần tiếp theo:

1. Mở Command Prompt/Terminal.
2. Điều hướng đến thư mục project: cd "thư mục chứa code" hoặc mở cmd ở đường link file
3. Kích hoạt môi trường ảo: pdf_rag_env\Scripts\activate
4. chạy lệnh streamlit run ....py 
