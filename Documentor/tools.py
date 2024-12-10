import os
from dotenv import load_dotenv
import PyPDF2
import markdown
import logging
from sentence_transformers import SentenceTransformer
from crewai_tools import BaseTool
from typing import Any, Optional

# Load environment variables
load_dotenv()

class PDFExtractionTool(BaseTool):
    name: str = "PDF Extraction Tool"
    description: str = "This tool extracts text from a PDF file."
    pdf_path: str

    def _run(self):
        """Extracts text from a PDF file specified by pdf_path."""
        text = ""
        try:
            with open(self.pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            return "An error occurred while processing the PDF file."

        return text.strip()

class MarkdownFormatter(BaseTool):
    name: str = "Markdown Formatter"
    description: str = "This tool converts text content to Markdown format."

    def _run(self, content: Optional[str] = None):
        """Converts provided content into Markdown format."""
        if not content:
            logging.warning("No content provided to MarkdownFormatter.")
            return "No content provided."
        return markdown.markdown(content)

class SummaryTool(BaseTool):
    name: str = "Summary Tool"
    description: str = "This tool summarizes content using a provided LLM tool."
    llm_tool: Any

    def _run(self, content: Optional[str] = None):
        """Generates a summary of the provided content using an LLM tool."""
        if not content:
            logging.warning("No content provided to SummaryTool.")
            return "No content provided."
        try:
            return self.llm_tool.generate_summary(content)
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return "An error occurred while generating the summary."

# Tool functions mapped to their respective tool classes
def create_pdf_extraction_tool(pdf_path):
    """Helper to instantiate PDFExtractionTool."""
    return PDFExtractionTool(pdf_path=pdf_path)

def create_summary_tool(llm_tool):
    """Helper to instantiate SummaryTool with a specified LLM tool."""
    return SummaryTool(llm_tool=llm_tool)

def create_markdown_formatter():
    """Helper to instantiate MarkdownFormatter."""
    return MarkdownFormatter()

tool_functions = {
    "PDFExtractionTool": create_pdf_extraction_tool,
    "MarkdownFormatter": create_markdown_formatter,
    "SummaryTool": create_summary_tool,
}

logging.basicConfig(level=logging.DEBUG)
logging.debug(f"Registered tools: {list(tool_functions.keys())}")