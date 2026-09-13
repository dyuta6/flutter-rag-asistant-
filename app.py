import io
import logging
import os
import re
import shutil
import zipfile
import urllib.request
from pathlib import PurePosixPath

import streamlit as st
import rarfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Flutter Dev Assistant", page_icon="⚡", layout="wide")
st.title("⚡ Flutter Codebase & Architecture Assistant (RAG)")

# 1. Embedding ve Vector DB Yükleme
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_db():
    embeddings = load_embeddings()
    return Chroma(persist_directory="./chroma_db_utf8", embedding_function=embeddings)


def process_documents_to_chroma(documents):
    """
    Belgelerden genel proje özeti çıkarır, chunk'lara böler ve Chroma DB'ye aktarır.
    """
    all_sources = sorted(doc.metadata["source"] for doc in documents)
    
    # Feature, Core ve Presentation modüllerini dinamik tespit et
    detected_features = sorted({
        parts[parts.index("features") + 1]
        for s in all_sources
        for parts in [PurePosixPath(s).parts]
        if "features" in parts and parts.index("features") + 1 < len(parts)
    })
    
    detected_core_dirs = sorted({
        parts[parts.index("lib") + 1]
        for s in all_sources
        for parts in [PurePosixPath(s).parts]
        if "lib" in parts and parts.index("lib") + 1 < len(parts) and parts[parts.index("lib") + 1] != "features"
    })
    
    # Projenin ana UseCase, Repository, Page ve Service dosyalarını listele
    key_components = [
        s for s in all_sources
        if any(marker in s.lower() for marker in ["usecase", "repository", "datasource", "page", "service", "model", "controller"])
    ]
    
    pubspec_content = ""
    readme_content = ""
    for doc in documents:
        src = doc.metadata["source"].lower()
        if "pubspec.yaml" in src and not pubspec_content:
            pubspec_content = doc.page_content[:800]
        elif "readme.md" in src and not readme_content and len(doc.page_content.strip()) > 30:
            readme_content = doc.page_content[:800]

    overview = (
        "GENEL PROJE MİMARİSİ VE İŞLEVSEL ÖZETİ:\n"
        f"Toplam İndekslenen Dosya Sayısı: {len(all_sources)}\n"
        f"Tespit Edilen Ana Modüller/Özellikler (Features): {', '.join(detected_features) if detected_features else 'Standart modüller'}\n"
        f"Çekirdek Dizinler (Core): {', '.join(detected_core_dirs)}\n"
        f"Ana İş Mantığı ve Ekran Bileşenleri:\n" + "\n".join(f"- {c}" for c in key_components[:30]) + "\n\n"
        f"PUBSPEC YAPILANDIRMASI:\n{pubspec_content}\n\n"
        f"README METNİ:\n{readme_content}\n\n"
        f"DİĞER DOSYALAR:\n" + "\n".join(all_sources[:100])
    )
    
    overview_document = Document(
        page_content=overview,
        metadata={"source": "PROJECT_OVERVIEW", "project_file": True}
    )

    code_documents = [doc for doc in documents if doc.metadata["source"].endswith(".dart")]
    other_documents = [doc for doc in documents if doc not in code_documents]
    chunks = []
    
    if code_documents:
        dart_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120,
            separators=["\nclass ", "\nabstract class ", "\nvoid ", "\nfinal ", "\n", " ", ""]
        )
        chunks.extend(dart_splitter.split_documents(code_documents))
    if other_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
        chunks.extend(text_splitter.split_documents(other_documents))

    chunks.append(overview_document)

    for chunk in chunks:
        source = chunk.metadata.get("source", "Bilinmeyen Kaynak")
        chunk.page_content = f"Kaynak dosya: {source}\n{chunk.page_content}"

    return Chroma.from_documents(documents=chunks, embedding=load_embeddings())


def build_uploaded_db(uploaded_archive):
    """
    Herhangi bir Flutter ZIP veya RAR projesini dinamik olarak ayrıştırır ve indeksler.
    """
    documents = []
    ignored_directories = {
        ".git", ".dart_tool", "build", "pods", "node_modules",
        ".gradle", "ephemeral", ".idea", ".plugin_symlinks", ".dartServer"
    }
    archive_bytes = io.BytesIO(uploaded_archive.getvalue())
    if uploaded_archive.name.lower().endswith(".rar"):
        unrar_path = shutil.which("unrar") or r"C:\Program Files\WinRAR\UnRAR.exe"
        if not os.path.exists(unrar_path):
            raise ValueError("RAR açmak için WinRAR/UnRAR bulunamadı.")
        rarfile.UNRAR_TOOL = unrar_path
        archive = rarfile.RarFile(archive_bytes)
    else:
        archive = zipfile.ZipFile(archive_bytes)

    with archive:
        for file_info in archive.infolist():
            path = PurePosixPath(file_info.filename)
            if file_info.is_dir() or ".." in path.parts:
                continue
            if any(part.casefold() in ignored_directories for part in path.parts):
                continue
            if path.suffix.lower() not in {".dart", ".md", ".yaml", ".yml", ".json"}:
                continue
            try:
                content = archive.read(file_info).decode("utf-8", errors="ignore")
            except Exception:
                continue
            documents.append(Document(
                page_content=content,
                metadata={"source": str(path), "project_file": True}
            ))

    if not documents:
        raise ValueError("Yüklenen arşivde desteklenen Flutter/Dart dosyası bulunamadı.")

    return process_documents_to_chroma(documents)


def build_github_db(repo_url):
    """
    Public GitHub reposunu ZIP olarak indirip doğrudan hafızada indeksler.
    Örnek format: https://github.com/flutter/samples veya flutter/samples
    """
    clean_url = repo_url.strip().rstrip("/")
    # Kullanıcı sadece 'owner/repo' yazmışsa veya tam URL girmişse düzenle
    if clean_url.startswith("https://github.com/"):
        parts = clean_url.replace("https://github.com/", "").split("/")
    elif clean_url.startswith("github.com/"):
        parts = clean_url.replace("github.com/", "").split("/")
    else:
        parts = clean_url.split("/")

    if len(parts) < 2:
        raise ValueError("Geçersiz GitHub repo formatı. Örn: 'https://github.com/flutter/samples' veya 'flutter/samples'")

    owner, repo = parts[0], parts[1]
    
    # Ana dalı tespit etmek için sırasıyla main ve master zipurl'leri denenir
    zip_urls = [
        f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip",
        f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
    ]

    zip_bytes = None
    last_error = None

    req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in zip_urls:
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    zip_bytes = io.BytesIO(resp.read())
                    break
        except Exception as e:
            last_error = e

    if zip_bytes is None:
        raise ValueError(f"GitHub reposu indirilemedi ({owner}/{repo}). Repo public olduğundan ve 'main'/'master' dalının bulunduğundan emin olun. Hata: {last_error}")

    documents = []
    ignored_directories = {
        ".git", ".dart_tool", "build", "pods", "node_modules",
        ".gradle", "ephemeral", ".idea", ".plugin_symlinks", ".dartServer"
    }

    with zipfile.ZipFile(zip_bytes) as archive:
        for file_info in archive.infolist():
            path = PurePosixPath(file_info.filename)
            if file_info.is_dir() or ".." in path.parts:
                continue
            if any(part.casefold() in ignored_directories for part in path.parts):
                continue
            if path.suffix.lower() not in {".dart", ".md", ".yaml", ".yml", ".json"}:
                continue
            try:
                content = archive.read(file_info).decode("utf-8", errors="ignore")
            except Exception:
                continue
            documents.append(Document(
                page_content=content,
                metadata={"source": str(path), "project_file": True}
            ))

    if not documents:
        raise ValueError(f"'{owner}/{repo}' reposunda desteklenen Flutter/Dart dosyası bulunamadı.")

    return process_documents_to_chroma(documents)


# Sidebar Proje Yükleyici
st.sidebar.header("📁 Flutter Projesi Yükle")

upload_mode = st.sidebar.radio(
    "Proje Kaynağı Seçin:",
    ["📦 Dosya Yükle (ZIP / RAR)", "🌐 GitHub Reposu (Public)"],
    index=0
)

if upload_mode == "📦 Dosya Yükle (ZIP / RAR)":
    uploaded_project = st.sidebar.file_uploader(
        "Proje ZIP veya RAR dosyasını yükleyin",
        type=["zip", "rar"],
        help="Flutter projenizin ZIP/RAR arşivini yükleyin. Kodlar, bağımlılıklar ve mimari dinamik olarak taranır."
    )

    if uploaded_project is not None:
        upload_key = f"v4:{uploaded_project.name}:{uploaded_project.size}"
        if st.session_state.get("project_key") != upload_key:
            try:
                with st.spinner("Flutter projesi analiz ediliyor ve vektör veri tabanı oluşturuluyor..."):
                    st.session_state.project_db = build_uploaded_db(uploaded_project)
                    st.session_state.project_key = upload_key
                st.sidebar.success(f"'{uploaded_project.name}' başarıyla indekslendi.")
            except Exception as error:
                st.sidebar.error(f"Hata: {str(error)}")

else:
    github_url = st.sidebar.text_input(
        "GitHub Repo URL'si",
        placeholder="https://github.com/flutter/samples",
        help="Herhangi bir public Flutter reposunun GitHub linkini girin."
    )
    fetch_btn = st.sidebar.button("🚀 Repoyu İndir ve İncele", use_container_width=True)

    if fetch_btn and github_url:
        gh_key = f"gh:{github_url.strip().lower()}"
        if st.session_state.get("project_key") != gh_key:
            try:
                with st.spinner(f"GitHub reposu indiriliyor ve analiz ediliyor ({github_url})..."):
                    st.session_state.project_db = build_github_db(github_url)
                    st.session_state.project_key = gh_key
                st.sidebar.success("GitHub reposu başarıyla indirildi ve indekslendi!")
            except Exception as error:
                st.sidebar.error(f"Hata: {str(error)}")

vector_db = st.session_state.get("project_db")
if vector_db is None:
    vector_db = load_db()

# BM25 + Vektör Hibrit Arama Motoru
_bm25_vector_db = None
_bm25_documents = []
_bm25_index = None


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_çğıöşü]+", text.casefold())


def get_bm25_index():
    global _bm25_vector_db, _bm25_documents, _bm25_index
    if _bm25_vector_db is vector_db and _bm25_index is not None:
        return _bm25_documents, _bm25_index

    stored_data = vector_db.get(include=["documents", "metadatas"])
    _bm25_documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            stored_data.get("documents", []),
            stored_data.get("metadatas", []),
        )
    ]
    _bm25_index = BM25Okapi([tokenize(doc.page_content) for doc in _bm25_documents])
    _bm25_vector_db = vector_db
    return _bm25_documents, _bm25_index


def retrieve_docs(query, k=6):
    """
    Evrensel Hibrit Arama (Dense Semantic Vector + Sparse BM25 + Lexical Matching)
    """
    query_terms = [t for t in tokenize(query) if len(t) > 2]
    
    # 1. Semantik Arama
    semantic_candidates = vector_db.similarity_search(query, k=25)
    
    # 2. BM25 Arama
    bm25_documents, bm25_index = get_bm25_index()
    bm25_scores = bm25_index.get_scores(tokenize(query))
    top_bm25_indices = sorted(range(len(bm25_scores)), key=bm25_scores.__getitem__, reverse=True)[:25]
    
    candidates = list(semantic_candidates)
    candidates.extend(bm25_documents[i] for i in top_bm25_indices)
    
    # Proje amacı / Genel mimari / Ne işe yarar soruları için doğrudan README, pubspec ve OVERVIEW önceliği
    is_general_query = any(w in query.casefold() for w in ["amaç", "nedir", "özet", "ne işe yarar", "mimari", "proje", "uygulama"])
    if is_general_query:
        overview_data = vector_db.get(where={"source": "PROJECT_OVERVIEW"})
        if overview_data and overview_data.get("documents"):
            candidates.insert(0, Document(
                page_content=overview_data["documents"][0],
                metadata=overview_data["metadatas"][0] if overview_data.get("metadatas") else {"source": "PROJECT_OVERVIEW"}
            ))
        for doc in bm25_documents:
            s = doc.metadata.get("source", "").lower()
            if "readme" in s or "pubspec.yaml" in s:
                candidates.insert(0, doc)

    # Tekilleştirme ve Hibrit Skorlama
    unique_candidates = {}
    for doc in candidates:
        source = doc.metadata.get("source", "")
        key = (source, doc.page_content[:150])
        if key not in unique_candidates:
            # Skor hesaplama: Semantik eşleşme + Terim sıklığı
            term_score = sum(t in doc.page_content.casefold() or t in source.casefold() for t in query_terms)
            if is_general_query and any(kw in source.lower() for kw in ["project_overview", "readme", "pubspec"]):
                term_score += 10
            unique_candidates[key] = (doc, term_score)

    ranked_docs = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs[:k]]


def format_docs(docs, max_chars=3500):
    formatted = []
    total_chars = 0
    for doc in docs:
        source = doc.metadata.get("source", "Bilinmeyen Kaynak")
        chunk = doc.page_content
        entry = f"--- [DOSYA: {source}] ---\n{chunk}"
        if total_chars + len(entry) > max_chars:
            break
        formatted.append(entry)
        total_chars += len(entry)
    return "\n\n".join(formatted)


# 2. Local LLM (Ollama - qwen2.5:3b)
llm = OllamaLLM(
    model="qwen2.5:3b",
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.18,
    num_predict=400,
    num_ctx=4096,
    num_gpu=99,
)

# 3. Evrensel Flutter Uzmanı Prompt Şablonu
system_template = """Sen kıdemli bir Flutter ve Dart yazılım mimarısın.
Aşağıda kullanıcının yüklediği Flutter projesinden/paketinden çıkarılan kod parçacıkları, modüller, sınıflar ve genel özet (Bağlam) verilmiştir.

Sana verilen bu bağlamı inceleyerek kullanıcının sorusunu doğrudan, net, akıcı ve Türkçe olarak yanıtla.

Bağlam (Context):
{context}

Soru: {question}

Kurallar:
- Sadece Türkçe yanıt ver.
- Asla aynı kelimeleri veya cümleleri arka arkaya tekrarlama.
- Projenin amacını, sunduğu özellikleri ve temel widget/sınıf yapılarını doğrudan ve özet şekilde açıkla.

Türkçe Cevap:"""

prompt = ChatPromptTemplate.from_template(system_template)

def clean_response(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()

rag_chain = (
    {"context": lambda q: format_docs(retrieve_docs(q)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_response)
)

# 4. Streamlit Sohbet Arayüzü (Streaming Desteği ile)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Proje veya Flutter mimarisi hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        retrieved_docs = retrieve_docs(user_query)
        
        with st.spinner("Proje kodları analiz ediliyor ve yanıt üretiliyor..."):
            full_response = rag_chain.invoke(user_query)
            st.markdown(full_response)

        with st.expander("📚 Kullanılan Kaynak Dosyalar (Retrieved Context)"):
            for idx, doc in enumerate(retrieved_docs):
                source_name = doc.metadata.get("source", "Bilinmeyen Kaynak")
                st.write(f"**Kaynak {idx+1}:** `{source_name}`")
                st.code(doc.page_content, language="dart")

    st.session_state.messages.append({"role": "assistant", "content": full_response})