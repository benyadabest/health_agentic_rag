import streamlit as st
from pathlib import Path
import json
from PyPDF2 import PdfReader
import os
import base64

st.set_page_config(
    page_title="Document Summaries",
    page_icon="📄",
    layout="wide"
)

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

SUMMARIES_PATH = DOCS_DIR / "summaries.json"

def read_file_safe(file_path):
    """Read file with fallback encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
            
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')

def extract_text_from_pdf(file_path):
    """Extract readable text content from PDF"""
    try:
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return "\n\n".join(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def save_document_analysis(file_name, analysis, text_content):
    """Save document analysis and text content"""
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    
    text_path = docs_dir / f"{file_name}.txt"
    with open(text_path, "w", encoding='utf-8') as f:
        f.write(text_content)
    
    analysis["file_path"] = str(text_path)
    
    summaries_path = docs_dir / "summaries.json"
    if summaries_path.exists():
        with open(summaries_path, "r", encoding='utf-8') as f:
            summaries = json.load(f)
    else:
        summaries = {}
    
    summaries[file_name] = analysis
    with open(summaries_path, "w", encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

def load_summaries():
    """Load all document summaries"""
    if SUMMARIES_PATH.exists():
        with open(SUMMARIES_PATH, "r", encoding='utf-8') as f:
            return json.load(f)
    return {}

def main():
    st.title("📄 Document Summaries and Topics")
    
    summaries = load_summaries()
    
    if not summaries:
        st.info("No documents have been uploaded yet. Please upload documents in the main chat page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Documents")
        selected_doc = st.radio(
            "Select a document to view",
            options=list(summaries.keys()),
            format_func=lambda x: x[:30] + "..." if len(x) > 30 else x
        )
    
    with col2:
        if selected_doc:
            doc_data = summaries[selected_doc]
            
            st.subheader(f"Document: {selected_doc}")
            
            with st.expander("📝 Summary", expanded=True):
                st.markdown(doc_data["summary"])
            
            st.subheader("🏷️ Key Topics")
            
            cols = st.columns(2) 
            for i, topic in enumerate(doc_data["topics"]):
                col = cols[i % 2]  
                with col:
                    if st.button(
                        f"🔍 {topic}",
                        key=f"{selected_doc}-{topic}",
                        help=f"Click to discuss {topic} in the context of {selected_doc}",
                        use_container_width=True
                    ):
                        doc_path = Path(doc_data["file_path"]) 
                        doc_text = ""
                        if doc_path.exists():
                            reader = PdfReader(doc_path)
                            text_content = []
                            for page in reader.pages:
                                text_content.append(page.extract_text())
                            doc_text = "\n\n".join(text_content)
                            
                        st.session_state["doc_query"] = {
                            "topic": topic,
                            "content": doc_text,
                            "doc_name": selected_doc
                        }
                        st.switch_page("app.py")
            
            st.subheader("📄 Document Preview")
            pdf_path = DOCS_DIR / selected_doc

            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                # Download button
                st.download_button(
                    "⬇️ Download PDF",
                    pdf_bytes,
                    file_name=selected_doc,
                    mime="application/pdf",
                )
                
                # Display PDF using base64 encoded data in an iframe
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.error("Original PDF file not found")

if __name__ == "__main__":
    main()