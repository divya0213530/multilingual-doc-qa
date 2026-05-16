# ingest.py
# PURPOSE: Load a document, split into chunks, 
#          convert to vectors, store in ChromaDB

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings



def ingest_file(file_path: str):
    """
    Takes a file path (PDF or TXT),
    processes it and stores in ChromaDB vector database.
    """

    # ── STEP A: LOAD THE FILE ─────────────────────────────
    # Check file type and use the right loader
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        # PyPDFLoader reads PDF and converts each page to text
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        # TextLoader reads plain .txt files
        # encoding="utf-8" is important for Telugu/Hindi text

    docs = loader.load()
    # docs is a list of Document objects
    # Each Document has:
    #   .page_content → the actual text
    #   .metadata     → {"source": "filename.pdf", "page": 0}

    print(f"📄 Loaded {len(docs)} page(s) from {file_path}")


    # ── STEP B: SPLIT INTO CHUNKS ─────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,    
        # Each chunk = max 400 characters
        # Why 400? Small enough for precise retrieval,
        # large enough to have meaningful context

        chunk_overlap=60   
        # Last 60 chars of chunk N = first 60 chars of chunk N+1
        # Why overlap? So sentences at boundaries aren't cut off
    )

    chunks = splitter.split_documents(docs)
    # chunks is a list of smaller Document objects
    # Example: 10-page PDF might become 80 chunks

    print(f"✂️  Split into {len(chunks)} chunks")


    # ── STEP C: CREATE EMBEDDING MODEL ───────────────────
    embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
        # This model:
        # - Downloads once (~120MB) from HuggingFace
        # - Runs locally after that (no internet needed)
        # - Supports 50+ languages including Telugu & Hindi
        # - Converts text → 384 numbers (vector)
        # - Same meaning in different languages = similar vectors
    )


    # ── STEP D: EMBED CHUNKS & STORE IN CHROMADB ─────────
    vectorstore = Chroma.from_documents(
        chunks,                        # our text chunks
        embeddings,                    # embedding model
        persist_directory="./db"       
        # Saves the database to a folder called "db"
        # So data survives even after you close the app
    )
    # What ChromaDB does internally:
    # 1. Calls embedding model on each chunk → gets vector
    # 2. Stores (text + vector + metadata) together
    # 3. Builds a search index for fast retrieval

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB (./db folder)")
    return vectorstore