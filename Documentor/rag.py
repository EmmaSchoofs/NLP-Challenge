from chunk_vector_store import ChunkVectorStore as cvs
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import json
from langchain import hub

class Rag:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self) -> None:
        self.csv_obj = cvs()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.model = ChatOllama(model="mistral")

    @staticmethod
    def ensure_string(input_data):
        if isinstance(input_data, str):
            return input_data  # Already a string
        elif isinstance(input_data, dict):
            return json.dumps(input_data)  # Convert dictionary to JSON string
        elif isinstance(input_data, list):
            return "\n".join(map(str, input_data))  # Join list items into a single string
        else:
            return str(input_data)  # Convert other types to string

    def set_retriever(self):
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Call `store_to_vector_database` first.")
        # Set retriever as a method of the class
        self.retriever = self.retrieve_from_store

    def retrieve_from_store(self, query, n_results=5):
        """This function queries the vector store using the provided query."""
        results = self.vector_store.query(
            query_texts=[query],  # Pass query string properly
            n_results=n_results
        )
        # Ensure documents are strings
        if "documents" not in results:
            raise ValueError("No documents returned by the vector store query.")
        results["documents"] = [self.ensure_string(doc) for doc in results["documents"]]
        return results

    def augment(self):
        """Augments the context to the original prompt and sets up the chain."""
        # Use retriever as a method for retrieving context
        context_runnable = RunnablePassthrough(self.retriever)  # Ensure that retriever can be used here

        # Now augment context and question
        self.chain = (context_runnable | self.prompt | self.model | StrOutputParser())

    def ask(self, query: str):
        """Run the RAG pipeline to answer a question."""
        if not self.chain:
            print("Chain is not initialized!")
            return "Chain is not initialized. Please upload a PDF file first."

        # Query the retriever for context
        results = self.retriever(query, n_results=5)

        # Validate and extract the documents
        if "documents" not in results or not results["documents"]:
            return "No relevant documents found in the vector store."

        # Join all documents into a single context string
        context = self.ensure_string(results["documents"][0])
        query = self.ensure_string(query)

        # Format context and query as a single string for input to the chain
         # query_input = f"Context: {context}\nQuestion: {query}"
        
        query_input = {
            "context": context,
            "question": query
        }       


        # Run the chain with the retrieved context
        response = self.chain.invoke(query_input)  # Pass the dictionary to the chain

        return response

    def feed(self, file_path):
        """Store the file into the vector database."""
        chunks = self.csv_obj.split_into_chunks(file_path)
        print(f"Chunks created: {len(chunks)}")

        self.vector_store = self.csv_obj.store_to_vector_database(chunks)
        print(f"Vector store initialized: {self.vector_store is not None}")

        if not self.vector_store:
            raise ValueError("Failed to initialize the vector store.")

        self.set_retriever()
        self.augment()

    def clear(self):
        """Clears the vector store and resets components."""
        self.vector_store = None
        self.chain = None
        self.retriever = None