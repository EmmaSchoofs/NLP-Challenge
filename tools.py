import os
from dotenv import load_dotenv
import PyPDF2
import markdown
from sentence_transformers import SentenceTransformer
import requests
import logging
from crewai_tools import BaseTool
from typing import Any, Optional


# Load environment variables
load_dotenv()

class PDFExtractionTool(BaseTool):
    name: str = "PDF Extraction Tool"
    description: str = "This tool will extract text from a pdf file."
    pdf_path: str

    def _run(self):
        text = ""
        with open(self.pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
        text = text.strip()
        print(text)
        return text


class MarkdownFormatter(BaseTool):
    name: str = "Markdown Formatter"
    description: str = "This tool will convert any content to markdown."

    def _run(self, content: Optional[str] = None):
        if not content:
            return "No content provided"
        return markdown.markdown(content)



class SummaryTool(BaseTool):
    name: str = "Summary Tool"
    description: str = "This tool will give a summary of any content using an llm tool."
    llm_tool: Any

    def _run(self, content: Optional[str] = None):
        if not content:
            return "No content provided."
        return self.llm_tool.generate_summary(content)


# Map tools to their classes
tool_functions = {
    "PDF Extraction Tool": lambda pdf_path: PDFExtractionTool(pdf_path=pdf_path),
    "Markdown Formatter": MarkdownFormatter,
    "Summary Tool": lambda llm_tool, content = None: SummaryTool(llm_tool=llm_tool, content=content),
}


logging.basicConfig(level=logging.DEBUG)
logging.debug(f"Registered tools: {list(tool_functions.keys())}")