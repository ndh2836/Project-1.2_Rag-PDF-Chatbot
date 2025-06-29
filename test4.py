import torch
import tempfile
import os
import streamlit as st

# Xây dựng vector database
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Xây dựng RAG Chain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker

# Xây dựng Vector Database
from langchain_chroma import Chroma
from langchain import hub

# Khởi tạo Session State
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hàm tải Embedding Model (cache model embeddings, tránh việc tải lại nhiều lần)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

# Hàm tải LLM
@st.cache_resource
def load_llm():
    """
    Tải và cache mô hình ngôn ngữ lớn (LLM) từ HuggingFace.
    Bao gồm cơ chế fallback sang CPU nếu GPU không khả dụng.
    Sử dụng @st.cache_resource để chỉ tải một lần duy nhất.
    """
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    st.info(f"Đang tải LLM: {MODEL_NAME}...")

    # Kiểm tra xem có GPU và CUDA có sẵn không
    if torch.cuda.is_available():
        st.info("Phát hiện GPU và CUDA. Đang tải mô hình với lượng tử hóa 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto", # Tự động chọn thiết bị (GPU nếu có)
        )
    else:
        st.warning("Không tìm thấy GPU hoặc CUDA Toolkit. Mô hình sẽ chạy trên CPU, có thể rất chậm và tốn RAM.")
        # Khi chạy trên CPU, không sử dụng BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32, # Thường dùng float32 cho CPU
            low_cpu_mem_usage=True,
            device_map="cpu", # Buộc chạy trên CPU
        )
        # Lưu ý: Khi chạy trên CPU, mô hình 7B có thể tốn rất nhiều RAM (khoảng 14GB cho float32)

    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device=0 if torch.cuda.is_available() else -1
    )
    # Trả về mô hình pipeline
    return HuggingFacePipeline(pipeline=model_pipeline)


# Hàm xử lý PDF
def process_pdf(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload_file.getvalue())
        tmp_file_path = tmp_file.name

    # Tạo đối tượng PyPDFLoader để tải tài liệu PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Xử lý tài liệu PDF để tạo vector database
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    # Chia tài liệu thành các đoạn nhỏ hơn
    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")# Lấy prompt từ LangChain Hub
    
    # Hàm format_docs không cần thiết trực tiếp ở đây nữa vì ConversationalRetrievalChain sẽ tự xử lý
    # def format_docs(docs):
    #   return "\n\n".join(doc.page_content for doc in docs)

    # Tạo ConversationalRetrievalChain có bộ nhớ
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=retriever,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        combine_docs_chain_kwargs={"prompt": prompt},
        chain_type="stuff"
    )

    os.unlink(tmp_file_path)
    return qa_chain, len(docs) #Trả về qa_chain

# Giao diện người dùng Streamlit
st.set_page_config(page_title="PDF RAG Chatbot", page_icon=":robot:", layout="wide")

st.title("Hỏi đáp PDF với AI 🤖")

st.markdown("""
**Ứng dụng AI hỗ trợ hỏi đáp trực tiếp với nội dung tải lên từ file PDF bằng tiếng Việt.**
**Hướng dẫn sử dụng:**
1.  Tải lên file PDF chứa nội dung bạn muốn hỏi đáp.
2.  Nhấn nút "Xử lý PDF" và chờ trong giây lát.
3.  Nhập câu hỏi của bạn vào ô bên dưới và nhấn Enter.
---
""")

# Tải model
if not st.session_state.model_loaded:
    with st.spinner("Đang tải các mô hình AI, vui lòng chờ..."):
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.model_loaded = True
        st.success("Model đã được tải thành công!")

# Tải file PDF
uploaded_file = st.file_uploader("Tải lên file PDF", type=["pdf"])
if uploaded_file and st.button("Xử lý PDF"):
    with st.spinner("Đang xử lý PDF..."):
        st.session_state.rag_chain, num_docs = process_pdf(uploaded_file)
        st.session_state.messages = []
        st.success(f"Đã xử lý {num_docs} đoạn từ tài liệu PDF.")

# Giao diện hỏi đáp
if st.session_state.rag_chain:
    # Hiển thị lịch sử nói chuyện
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nhập câu hỏi
    question = st.chat_input("Nhập câu hỏi của bạn:")

    if question:
        # Thêm câu hỏi vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Đang trả lời..."):
            try:
                # Gọi rag_chain.invoke. Không cần truyền chat_history vào đây
                # vì ConversationalRetrievalChain tự quản lý bộ nhớ
                output = st.session_state.rag_chain.invoke({"question": question})

                answer = output.get("answer", "Không tìm thấy câu trả lời.") # Dùng .get để tránh lỗi nếu key không tồn tại

                # Thêm câu trả lời của AI vào lịch sử chat hiển thị
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Hiển thị câu trả lời của AI
                with st.chat_message("assistant"):
                    st.markdown(answer)

            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi tạo câu trả lời: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Xin lỗi, đã có lỗi xảy ra: {e}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Xin lỗi, đã có lỗi xảy ra: {e}")
