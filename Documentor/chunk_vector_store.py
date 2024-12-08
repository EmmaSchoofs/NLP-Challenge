from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.config import Settings

class ChunkVectorStore:
    def __init__(self) -> None:
        # Initialize the ChromaDB client
        self.client = chromadb.PersistentClient(path="chromadb")
        self.client.heartbeat()
        self.collection = self.client.get_or_create_collection(name="my_collection")

    def split_into_chunks(self, file_path: str):
        # Load and split the PDF into chunks
        doc = PyPDFLoader(file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
        chunks = text_splitter.split_documents(doc)
        
        # Filter complex metadata (you need to define this function separately)
        chunks = self.filter_complex_metadata(chunks)

        return chunks

    def filter_complex_metadata(self, chunks):
        # Placeholder for metadata filtering logic
        # Modify as per your needs
        return chunks

    def store_to_vector_database(self, chunks):
        # Ensure page_content is stringified and metadata is valid
        documents = [str(chunk.page_content) if not isinstance(chunk.page_content, str) else chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata if isinstance(chunk.metadata, dict) else {} for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return self.collection

    def query_vector_database(self, query_text, n_results=2):
        # Query the ChromaDB collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results