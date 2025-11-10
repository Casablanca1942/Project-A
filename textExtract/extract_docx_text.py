from docx import Document
from pathlib import Path

def extract_text_from_docx(docx_path: str, output_txt_path: str):
    doc = Document(docx_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    Path(output_txt_path).write_text(full_text, encoding="utf-8")
    print(f"[✔] 已提取文本并保存到: {output_txt_path}")

if __name__ == "__main__":
    extract_text_from_docx(
        docx_path="The Intelligent Investor.docx",
        output_txt_path="TII.txt"
    )
