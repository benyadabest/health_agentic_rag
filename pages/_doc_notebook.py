import streamlit as st
import base64
from PyPDF2 import PdfReader

st.set_page_config(
    page_title="Document Summaries",
    page_icon="📄",
    layout="wide"
)

def read_file_safe(file_path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
            
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')

def main():
    st.title("📄 Document Summaries and Topics")
    
    if "uploaded_files" not in st.session_state:
        st.info("No documents have been uploaded yet. Please upload documents in the main chat page.")
        return
    
    if not st.session_state.uploaded_files:
        st.info("No documents have been uploaded yet. Please upload documents in the main chat page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Documents")
        selected_doc = st.radio(
            "Select a document to view",
            options=list(st.session_state.uploaded_files.keys()),
            format_func=lambda x: x[:30] + "..." if len(x) > 30 else x
        )
    
    with col2:
        if selected_doc:
            doc_data = st.session_state.uploaded_files[selected_doc]
            
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
                        st.session_state["doc_query"] = {
                            "topic": topic,
                            "content": doc_data["content"],
                            "doc_name": selected_doc
                        }
                        st.switch_page("app.py")
            
            st.subheader("📄 Document Content")
            st.text_area(
                "Full Document Text",
                value=doc_data["content"],
                height=400,
                disabled=True
            )

if __name__ == "__main__":
    main()