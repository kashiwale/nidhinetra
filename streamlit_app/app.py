import streamlit as st
st.set_page_config(page_title="NidhiNetra", layout="wide")
from pathlib import Path
from nidhinetra.core.ingest import extract_text_from_pdf, ingest_pdf_to_chroma, search_documents
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
st.write("API Key present:", bool(os.getenv("OPENAI_API_KEY")))

# Optional: sanity check in dev mode
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY not found. Please check your .env file.")

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)


st.title("📂 NidhiNetra - Tax Document Insight Portal")

st.markdown("##### _By Ishani & Arjun_ ✨")

# Select PDF File
pdf_folder = Path("nidhinetra/data/returns")
pdf_files = list(pdf_folder.glob("*.pdf"))
pdf_names = [f.name for f in pdf_files]

selected_pdf = st.selectbox("Select a Tax Return PDF", pdf_names)

option = st.radio("Choose Operation", ["📜 Extract Text", "🔍 Semantic Search"])

if selected_pdf:
    full_path = str(pdf_folder / selected_pdf)
    namespace = "llc_returns"
    user_id = "user_dhirendra"
    year = "2024"

    if option == "📜 Extract Text":
        st.info("✨ Extracting pure text magic from the document...")
        with st.spinner("📜 Unfolding the document..."):
            content = extract_text_from_pdf(full_path)
            st.text_area("📄 Raw Extracted Text", content[:5000], height=400)

    elif option == "🔍 Semantic Search":
        st.info("🔍 Sending your document to the Chroma universe...")
        with st.spinner("🚀 Ingesting into Chroma..."):
            count = ingest_pdf_to_chroma(
                pdf_path=full_path,
                namespace=namespace,
                user_id=user_id,
                year=year
            )
            st.success(f"✅ Ingested {count} chunks into Chroma!")

        query = st.text_input("Ask something about the document (e.g. 'What is the net income?')")
        if st.button("🔎 Search"):
            with st.spinner("🧠 Thinking through the semantic fog..."):
                results = search_documents(query=query, namespace=namespace, user_id=user_id, year=year)
                for i, (doc, score) in enumerate(results):
                    st.markdown(f"**Result {i+1}** (Score: {score:.2f})")
                    st.code(doc.page_content[:800])
                    st.markdown("---")
