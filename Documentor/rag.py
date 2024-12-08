from chunk_vector_store import ChunkVectorStore as cvs
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import json

class Rag:

  vector_store = None
  retriever = None
  chain = None

  def __init__(self) -> None:
    self.csv_obj = cvs()
    self.prompt = PromptTemplate.from_template(
      """
      <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
      to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
        maximum and keep the answer concise. [/INST] </s>
      [INST] Question: {question}
      Context: {context}
      Answer: [/INST]
      """
    )
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

      # Set retriever as a method of the class, now we pass the query properly
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

  # Augment the context to original prompt.
  def augment(self):
    self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
    | self.prompt
    | self.model
    | StrOutputParser())


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

      # Run the chain with the retrieved context
      response = self.chain.invoke({"context": context, "question": query})

      return response

  # Stores the file into vector database.
  def feed(self, file_path):
      chunks = self.csv_obj.split_into_chunks(file_path)
      print(f"Chunks created: {len(chunks)}")

      self.vector_store = self.csv_obj.store_to_vector_database(chunks)
      print(f"Vector store initialized: {self.vector_store is not None}")

      if not self.vector_store:
          raise ValueError("Failed to initialize the vector store.")

      self.set_retriever()
      self.augment() 


  def clear(self):
    self.vector_store = None
    self.chain = None
    self.retriever = None