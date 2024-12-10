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

        # Filter complex metadata
        chunks = self.filter_complex_metadata(chunks)

        return chunks

    def filter_complex_metadata(self, chunks):
        # This function filters and simplifies metadata based on your needs.
        for chunk in chunks:
            # Example: Keep only necessary metadata fields (e.g., page number, title)
            # This is just a basic filter, you can extend it as required.
            chunk.metadata = {key: value for key, value in chunk.metadata.items() if key in ['page_number', 'author']}
        return chunks

    def store_to_vector_database(self, chunks):
        # Ensure page_content is stringified and metadata is valid
        documents = [str(chunk.page_content) if not isinstance(chunk.page_content, str) else chunk.page_content for chunk in chunks]
        
        # Ensure metadata is a non-empty dict
        metadatas = []
        for chunk in chunks:
            # If no metadata exists, create a placeholder
            if not chunk.metadata:
                chunk.metadata = {'source': 'unknown'}  # Default metadata (you can adjust this)
            elif not isinstance(chunk.metadata, dict):
                chunk.metadata = {'source': str(chunk.metadata)}  # Convert non-dict metadata into a dict
            metadatas.append(chunk.metadata)
        
        ids = [f"doc_{i}" for i in range(len(chunks))]

        # Add documents to ChromaDB collection
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return self.collection

    def query_vector_database(self, query_input, n_results=2):
        """
        Query the ChromaDB collection. Accepts either a string query or a dictionary.
        """
        # If the input is a dictionary, extract context and question
        if isinstance(query_input, dict):
            query_text = query_input.get("question", "")
        else:
            query_text = query_input  # Default case for string input
        
        # Ensure we pass only the query text to the vector store
        results = self.collection.query(
            query_texts=[query_text],  # Pass query string properly
            n_results=n_results
        )

        # Ensure documents are returned correctly
        if "documents" not in results or not results["documents"]:
            raise ValueError("No documents found in the query results.")

        return results
