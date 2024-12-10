# test_tools.py
from tools import tool_functions

pdf_tool = tool_functions["PDF Extraction Tool"](pdf_path="test.pdf")
content = pdf_tool._run()
print("PDF Tool Content:", content)

markdown_tool = tool_functions["Markdown Formatter"]()
formatted = markdown_tool._run("**Test Markdown**")
print("Markdown Tool Content:", formatted)
