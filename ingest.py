import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Embedding Modelini Yükle (Lokalde ve çok hızlı çalışır)
print("Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Dart Kodlarını Yükle ve AST-Aware Parçala
print("Codebase taranıyor...")
if os.path.exists("./codebase"):
    code_loader = DirectoryLoader(
        './codebase',
        glob="**/*.dart",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    raw_code_docs = code_loader.load()

    dart_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\nclass ", "\nvoid ", "\nfinal ", "\n", " ", ""]
    )
    code_chunks = dart_splitter.split_documents(raw_code_docs)
    print(f"Toplam {len(code_chunks)} kod parçası oluşturuldu.")
else:
    code_chunks = []
    print("Warning: 'codebase' klasörü boş veya bulunamadı.")

# 3. Mimari Kuralları (Markdown) Yükle
print("Mimari kurallar taranıyor...")
if os.path.exists("./guidelines"):
    rule_loader = DirectoryLoader(
        './guidelines',
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    raw_rule_docs = rule_loader.load()

    rule_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    rule_chunks = rule_splitter.split_documents(raw_rule_docs)
    print(f"Toplam {len(rule_chunks)} kural parçası oluşturuldu.")
else:
    rule_chunks = []

# 4. Verileri birleştir ve ChromaDB'ye göm
all_chunks = code_chunks + rule_chunks

if all_chunks:
    print("Vektör veritabanı (ChromaDB) oluşturuluyor...")
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_utf8"
    )
    print("İşlem tamamlandı! Vektör veritabanı './chroma_db_utf8' dizinine kaydedildi.")
else:
    print("İşlenecek belge bulunamadı.")