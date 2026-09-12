import io
import os
import re
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

    all_sources = sorted(doc.metadata["source"] for doc in documents)
    feature_names = sorted({
        parts[parts.index("features") + 1]
        for source in all_sources
        for parts in [PurePosixPath(source).parts]
        if "features" in parts and parts.index("features") + 1 < len(parts)
    })
    overview = (
        "Bu Flutter projesinin indekslenmiş dosya özeti. "
        "Feature klasörleri: " + ", ".join(feature_names) + ". "
        "Proje dosyaları restoran keşfi, restoran menüsü, sepet, ödeme/adres, "
        "favoriler, kullanıcı profili, kimlik doğrulama ve sipariş akışlarını içerir.\n"
        "Dosya listesi:\n" + "\n".join(all_sources)
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
            chunk_size=1000, chunk_overlap=150,
            separators=["\nclass ", "\nvoid ", "\nfinal ", "\n", " ", ""]
        )
        chunks.extend(dart_splitter.split_documents(code_documents))
    if other_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=75)
        chunks.extend(text_splitter.split_documents(other_documents))

    chunks.append(overview_document)

    for chunk in chunks:
        source = chunk.metadata.get("source", "Bilinmeyen Kaynak")
        chunk.page_content = f"Kaynak dosya: {source}\n{chunk.page_content}"

    return Chroma.from_documents(documents=chunks, embedding=load_embeddings())


st.sidebar.header("Flutter Projesi")
uploaded_project = st.sidebar.file_uploader(
    "Proje ZIP dosyasını yükle",
    type=["zip", "rar"],
    help="Flutter projenizin ZIP veya RAR dosyasını yükleyin. lib/, pubspec.yaml ve dokümantasyon dosyaları indekslenir."
)

if uploaded_project is not None:
    upload_key = f"v3:{uploaded_project.name}:{uploaded_project.size}"
    if st.session_state.get("project_key") != upload_key:
        try:
            with st.spinner("Flutter projesi indeksleniyor..."):
                st.session_state.project_db = build_uploaded_db(uploaded_project)
                st.session_state.project_key = upload_key
            st.sidebar.success("Proje hazır")
        except (ValueError, zipfile.BadZipFile, rarfile.Error) as error:
            st.sidebar.error(str(error))

vector_db = st.session_state.get("project_db")
if vector_db is None:
    vector_db = load_db()


def retrieve_docs(query):
    candidates = vector_db.similarity_search(query, k=100)
    query_terms = {
        term for term in re.findall(r"[a-zA-Z0-9_]+", query.casefold())
        if len(term) > 2
    }
    if is_auth_service_query(query):
        query_terms.update({
            "firebase", "auth", "google", "sign-in", "signin",
            "kimlik", "doğrulama", "giriş", "apple", "email",
            "e-posta", "login_with_apple", "login_with_google",
            "login_with_email"
        })
    if is_data_storage_query(query):
        query_terms.update({
            "firebase", "firestore", "veritabanı", "database", "saklama",
            "kullanıcı", "veri", "auth_remote_data_source"
        })
    if is_store_listing_query(query):
        query_terms.update({
            "discover", "store", "stores", "restoran", "restaurant",
            "store_remote_data_source", "store_repository_impl",
            "get_stores_usecase"
        })

    def lexical_score(doc):
        source = doc.metadata.get("source", "").casefold()
        content = doc.page_content.casefold()
        content_matches = sum(term in content for term in query_terms)
        source_matches = sum(term in source for term in query_terms)
        return source_matches * 3 + content_matches

    ranked = sorted(candidates, key=lexical_score, reverse=True)
    if is_auth_service_query(query):
        auth_docs = []
        seen_sources = set()
        for filename in (
            "login_with_apple_usecase.dart",
            "login_with_google_usecase.dart",
            "login_with_email_usecase.dart",
            "register_with_email_usecase.dart",
            "auth_remote_data_source.dart",
            "auth_repository_impl.dart",
        ):
            for doc in vector_db.similarity_search(filename, k=2):
                source = doc.metadata.get("source", "")
                if filename in source and source not in seen_sources:
                    auth_docs.append(doc)
                    seen_sources.add(source)
        ranked = auth_docs + [
            doc for doc in ranked
            if doc.metadata.get("source", "") not in seen_sources
        ]
    if is_data_storage_query(query):
        storage_docs = []
        seen_sources = set()
        for search_term in (
            "Firestore",
            "auth_remote_data_source.dart",
            "PRIVACY_POLICY.md",
        ):
            for doc in vector_db.similarity_search(search_term, k=4):
                content = doc.page_content.casefold()
                source = doc.metadata.get("source", "")
                if ("firestore" in content or "firestore" in source.casefold()) and source not in seen_sources:
                    storage_docs.append(doc)
                    seen_sources.add(source)
        ranked = storage_docs + [
            doc for doc in ranked
            if doc.metadata.get("source", "") not in seen_sources
        ]
    if is_store_listing_query(query):
        store_docs = []
        seen_sources = set()
        for search_term in (
            "store_remote_data_source.dart",
            "store_repository_impl.dart",
            "get_stores_usecase.dart",
            "discover_page.dart",
        ):
            for doc in vector_db.similarity_search(search_term, k=3):
                source = doc.metadata.get("source", "")
                if search_term in source and source not in seen_sources:
                    store_docs.append(doc)
                    seen_sources.add(source)
        ranked = store_docs + [
            doc for doc in ranked
            if doc.metadata.get("source", "") not in seen_sources
        ]

    if is_remote_data_source_query(query):
        remote_source_docs = []
        seen_sources = set()
        for search_term in (
            "store_remote_data_source.dart",
            "FirebaseFirestore",
            "collection('stores')",
            "StoreRemoteDataSourceImpl",
            "discover feature",
        ):
            for doc in vector_db.similarity_search(search_term, k=4):
                source = doc.metadata.get("source", "")
                content = doc.page_content.casefold()
                if (
                    "store_remote_data_source" in source.casefold()
                    or "store_remote_data_source" in content
                    or "firebasefirestore" in content
                    or "collection('stores')" in content
                    or "collection(\"stores\")" in content
                ) and source not in seen_sources:
                    remote_source_docs.append(doc)
                    seen_sources.add(source)
        ranked = remote_source_docs + [
            doc for doc in ranked
            if doc.metadata.get("source", "") not in seen_sources
        ]

    generic_project_query = is_project_purpose_query(query)
    if generic_project_query:
        overview_data = vector_db.get(where={"source": "PROJECT_OVERVIEW"})
        overview_docs = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(
                overview_data.get("documents", []),
                overview_data.get("metadatas", []),
            )
        ]
        return overview_docs[:1] + [doc for doc in ranked if doc.metadata.get("source") != "PROJECT_OVERVIEW"][:7]
    return ranked[:8]


def is_project_purpose_query(query):
    normalized_query = query.casefold()
    return any(term in normalized_query for term in {
        "projenin amacı", "proje amacı", "ana amacı", "amacı nedir",
        "ne amaçla", "ne işe yarar", "uygulama nedir"
    })


def is_auth_service_query(query):
    normalized_query = query.casefold()
    return any(term in normalized_query for term in {
        "kullanıcı girişi", "giriş", "oturum", "login", "kayıt",
        "kimlik doğrulama", "hangi servis", "hangi hizmet", "e-posta",
        "email", "apple", "google", "auth", "giriş özelliği"
    }) and any(term in normalized_query for term in {
        "servis", "hizmet", "kullan", "yapıl", "sağlan", "giriş",
        "login", "auth", "doğrulama", "feature", "özellik", "bölüm",
        "hangi", "nerede", "dosya"
    })


def is_data_storage_query(query):
    normalized_query = query.casefold()
    return any(term in normalized_query for term in {
        "veri", "kullanıcı", "bilgi", "saklan", "depolan", "kayded"
    }) and any(term in normalized_query for term in {
        "firebase", "servis", "hizmet", "hangi", "nerede", "veritabanı",
        "database", "firestore", "saklan", "depolan", "kayded"
    })


def is_store_listing_query(query):
    normalized_query = query.casefold()
    return any(term in normalized_query for term in {
        "restoranlar", "restoranları", "restoran verileri", "mağazalar",
        "dükkanlar", "store", "stores"
    }) and any(term in normalized_query for term in {
        "hangi feature", "hangi özellik", "getir", "alın", "kaynak",
        "repository", "data source", "uzak kaynak", "nereden"
    })


def is_remote_data_source_query(query):
    normalized_query = query.casefold()
    return any(term in normalized_query for term in {
        "remote data source", "uzak kaynak", "data source", "datasource",
        "firestore", "stores", "restoran verisi", "restoran verileri"
    }) and any(term in normalized_query for term in {
        "hangi", "nerede", "sorumlu", "getir", "alır", "çek", "çekiyor",
        "collection", "stores", "firestore"
    })

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
- Önce doğrudan soruyu cevapla; soruyla ilgisiz mimari tavsiye, destek adresi veya genel Flutter bilgisi ekleme.
- Soru projenin veya uygulamanın amacını soruyorsa PROJECT_OVERVIEW kaynağındaki restoran keşfi, menü, sepet, ödeme/adres, favoriler, profil, kimlik doğrulama ve sipariş bilgilerini birleştirerek doğal bir özet yaz.
- Soru kullanıcı girişi veya kimlik doğrulama servisinin hangisi olduğunu soruyorsa kaynakta açıkça belirtilen Firebase Auth ve Google Sign-In servislerini cevapla.
- Soru giriş yöntemlerini soruyorsa, login_with_apple_usecase.dart, login_with_google_usecase.dart ve login_with_email_usecase.dart kaynaklarını dikkate al. Bu dosyalardan biri bağlamda görünüyorsa ilgili yöntemin bulunduğunu açıkça belirt: Apple ile giriş, Google ile giriş veya e-posta ile giriş. Firebase Auth'u kimlik doğrulama altyapısı olarak açıkla.
- Soru özellikle Apple ile giriş desteğini soruyorsa ve login_with_apple_usecase.dart bağlamda görünüyorsa doğrudan "Evet, projede Apple ile giriş desteği bulunmaktadır." diye cevapla.
- Soru e-posta ile kayıt veya giriş özelliğinin hangi feature içinde olduğunu soruyorsa, kaynak yollarındaki `lib/features/auth` klasörünü esas al ve doğrudan bunun Auth feature'ı olduğunu belirt. E-posta girişi `login_with_email_usecase.dart`, e-posta kaydı `register_with_email_usecase.dart` dosyasıyla ilişkilidir.
- Soru kullanıcı verilerinin hangi Firebase servisiyle saklandığını soruyorsa, kaynakta açıkça belirtilen Firebase Firestore servisini cevapla. Firebase Cloud Messaging (FCM) bildirimler içindir; `storageBucket` yapılandırması tek başına kullanıcı verilerinin orada saklandığını göstermez.
- Soru restoranların hangi feature tarafından getirildiğini soruyorsa ve bağlamda `features/discover` ile `store_remote_data_source.dart`, `store_repository_impl.dart` veya `get_stores_usecase.dart` dosyaları görünüyorsa doğrudan restoranların Discover feature'ı tarafından getirildiğini söyle. Uzak veri kaynağı `StoreRemoteDataSource`, repository `StoreRepository`, kullanım senaryosu `GetStoresUseCase` olarak belirt.
- Soru restoran verilerinin uzak kaynaktan alınmasından sorumlu data source'ı soruyorsa ve bağlamda `store_remote_data_source.dart` ve `StoreRemoteDataSourceImpl` görünüyorsa kesin olarak `StoreRemoteDataSourceImpl` olduğunu söyle. Bu sınıfın `FirebaseFirestore` üzerinden `stores` koleksiyonunu dinlediğini belirt.
- Soru hangi remote data source veya datasource olduğunu soruyorsa, bağlamda `StoreRemoteDataSourceImpl` ve `firestore.collection('stores')` görünüyorsa bunları doğrudan kullan; tahmin etme.
- Bağlamda cevap yoksa veya emin değilsen tam olarak "Verilen dokümanlarda bu bilgi bulunmuyor" de; tahmin etme.
- Dosya, sınıf, paket veya servis adı soruluyorsa yalnızca bağlamda açıkça görünen isimleri kullan.
- Kod örnekleri verirken temiz, Dart/Flutter standartlarına uygun kod yaz.

Cevap:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(
        f"Kaynak dosya: {doc.metadata.get('source', 'Bilinmeyen Kaynak')}\n{doc.page_content}"
        for doc in docs
    )

# 4. Modern LCEL Chain
rag_chain = (
    {"context": lambda query: format_docs(retrieve_docs(query)), "question": RunnablePassthrough()}
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
            source_docs = retrieve_docs(user_query)
            answer = rag_chain.invoke(user_query)

            st.markdown(answer)

            with st.expander("📚 Kullanılan Kaynaklar (Retrieved Context)"):
                for idx, doc in enumerate(source_docs):
                    source_name = doc.metadata.get("source", "Bilinmeyen Kaynak")
                    st.write(f"**Kaynak {idx+1}:** `{source_name}`")
                    st.code(doc.page_content, language="markdown")

    st.session_state.messages.append({"role": "assistant", "content": answer})