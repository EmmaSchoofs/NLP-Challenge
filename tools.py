import os
from dotenv import load_dotenv
import PyPDF2
import markdown
from sentence_transformers import SentenceTransformer
import requests
import logging


# Load environment variables
load_dotenv()

class PDFExtractionTool:
    def extract_text(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
        return text


class RAGTool:
    def __init__(self, llm_tool, vector_db_tool, embedder_model='all-MiniLM-L6-v2'):
        self.llm_tool = llm_tool
        self.vector_db_tool = vector_db_tool
        self.embedder = SentenceTransformer(embedder_model)

    def generate_answer(self, question):
        # Generate query vector
        query_vector = self.embedder.encode(question)
        
        # Retrieve relevant content
        relevant_content = self.vector_db_tool.retrieve(query_vector)
        
        # Generate answer using the LLM
        answer = self.llm_tool.generate_with_context(question, relevant_content)
        return answer


class MarkdownFormatter:
    def format(self, content):
        return markdown.markdown(content)



class SummaryTool:
    def __init__(self, llm_tool):
        self.llm_tool = llm_tool

    def generate_summary(self, content):
        return self.llm_tool.generate_summary(content)



class QuizTool:
    def __init__(self, llm_tool):
        self.llm_tool = llm_tool

    def generate_quiz(self, content):
        return self.llm_tool.generate_quiz(content)


class GroqLLMTool:
    api_key = os.getenv("GROQ_API_KEY")

    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api.groq.com/openai/v1/chat"

    def generate(self, prompt, max_tokens=100, temperature=0.7):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = requests.post(f"{self.base_url}/completions", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("text", "")
        except requests.exceptions.RequestException as e:
            logging.error(f"Groq API Error: {e}")
            return "An error occurred while processing your request."
