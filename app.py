import io
import os
import shutil
import zipfile
from pathlib import PurePosixPath

import streamlit as st
import rarfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Flutter Dev Assistant", page_icon="⚡", layout="wide")
st.title("⚡ Flutter Codebase & Architecture Assistant (RAG)")

# 1. Embedding ve Vector DB Yükleme
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


@st.cache_resource
def load_db():
    embeddings = load_embeddings()
    vector_db = Chroma(persist_directory="./chroma_db_utf8", embedding_function=embeddings)
    return vector_db


def build_uploaded_db(uploaded_zip):
    documents = []
    ignored_directories = {
        ".git", ".dart_tool", "build", "pods", "node_modules",
        ".gradle", "ephemeral", ".idea", ".plugin_symlinks"
    }
    archive_bytes = io.BytesIO(uploaded_zip.getvalue())
    if uploaded_zip.name.lower().endswith(".rar"):
        unrar_path = shutil.which("unrar") or r"C:\Program Files\WinRAR\UnRAR.exe"
        if not os.path.exists(unrar_path):
            raise ValueError("RAR açmak için WinRAR/UnRAR bulunamadı.")
        rarfile.UNRAR_TOOL = unrar_path
        archive = rarfile.RarFile(archive_bytes)
    else:
        archive = zipfile.ZipFile(archive_bytes)

    with archive:
        for zip_info in archive.infolist():
            path = PurePosixPath(zip_info.filename)
            if zip_info.is_dir() or ".." in path.parts:
                continue
            if any(part.casefold() in ignored_directories for part in path.parts):
                continue
            if path.suffix.lower() not in {".dart", ".md", ".yaml", ".yml", ".json"}:
                continue
            try:
                content = archive.read(zip_info).decode("utf-8")
            except UnicodeDecodeError:
                continue
            documents.append(Document(
                page_content=content,
                metadata={"source": str(path), "project_file": True}
            ))

    if not documents:
        raise ValueError("ZIP içinde desteklenen Flutter dosyası bulunamadı.")

    code_documents = [doc for doc in documents if doc.metadata["source"].endswith(".dart")]
    other_documents = [doc for doc in documents if doc not in code_documents]
    chunks = []
    if code_documents:
        dart_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150,
            separators=["\nclass ", "\nvoid ", "\nfinal ", "\n", " ", ""]
        )
        chunks.extend(dart_splitter.split_documents(code_documents))
    if other_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=75)
        chunks.extend(text_splitter.split_documents(other_documents))

    return Chroma.from_documents(documents=chunks, embedding=load_embeddings())


st.sidebar.header("Flutter Projesi")
uploaded_project = st.sidebar.file_uploader(
    "Proje ZIP dosyasını yükle",
    type=["zip", "rar"],
    help="Flutter projenizin ZIP veya RAR dosyasını yükleyin. lib/, pubspec.yaml ve dokümantasyon dosyaları indekslenir."
)

if uploaded_project is not None:
    upload_key = f"v2:{uploaded_project.name}:{uploaded_project.size}"
    if st.session_state.get("project_key") != upload_key:
        try:
            with st.spinner("Flutter projesi indeksleniyor..."):
                st.session_state.project_db = build_uploaded_db(uploaded_project)
                st.session_state.project_key = upload_key
            st.sidebar.success("Proje hazır")
        except (ValueError, zipfile.BadZipFile, rarfile.Error) as error:
            st.sidebar.error(str(error))

vector_db = st.session_state.get("project_db", load_db())
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# 2. Local LLM (Ollama)
llm = Ollama(model="qwen2.5:3b")

# 3. Prompt Template (LCEL Formatı)
template = """Sen kıdemli bir Flutter ve Dart yazılım mimarısın.
Aşağıda sana verilen Bağlam (Context) bilgilerini ve proje mimari kurallarını kullanarak kullanıcının sorusuna cevap ver.

Bağlam (Context):
{context}

Soru: {question}

Cevap Kuralları:
- Cevaplarında proje kurallarına kesinlikle uy.
- Eğer cevabı bağlam içerisinde bulamıyorsan veya emin değilsen "Verilen dokümanlarda bu bilgi bulunmuyor" de, uydurma.
- Kod örnekleri verirken temiz, Dart/Flutter standartlarına uygun kod yaz.

Cevap:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Modern LCEL Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Proje veya mimari kurallar hakkında bir soru sor..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Vektör veritabanı taranıyor ve lokal LLM ile yanıt üretiliyor..."):
            # Get response and source docs
            source_docs = retriever.invoke(user_query)
            answer = rag_chain.invoke(user_query)

            st.markdown(answer)

            with st.expander("📚 Kullanılan Kaynaklar (Retrieved Context)"):
                for idx, doc in enumerate(source_docs):
                    source_name = doc.metadata.get("source", "Bilinmeyen Kaynak")
                    st.write(f"**Kaynak {idx+1}:** `{source_name}`")
                    st.code(doc.page_content, language="markdown")

    st.session_state.messages.append({"role": "assistant", "content": answer})