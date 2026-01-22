import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#setup configuration
st.set_page_config(page_title="Journal Assistant", layout="wide")
st.title("Journaling Assistant with Citation")

#process pdf function
def process_pdf(uploaded_file):
    all_docs = []
    progress_text = "Reading Files..."
    loading_bar = st.progress(0, progress_text)
    total_files = len(uploaded_file)

    for i, file in enumerate(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file.name
        
        all_docs.extend(docs)

        os.remove(temp_path)
        loading_bar.progress((i+1)/total_files, text=f"Reading file {file.name}")
    
    #split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_docs = text_splitter.split_documents(all_docs)

    #creating vector database
    loading_bar.progress(0.9, text="Embedding process..")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(final_docs, embeddings)

    loading_bar.empty()
    return vector_db

#sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Api Key", type="password")
    uploaded_file = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process file"):
        if not api_key:
            st.error("Input Api Key!")
        elif not uploaded_file:
            st.error("Input File!")
        else:
            with st.spinner("Processing file"):
                st.write("Reading documents...")
                st.session_state["vectordb"] = process_pdf(uploaded_file)
                st.session_state["ready"] = True
                st.success(f"Succes reading {len(uploaded_file)} file")

#chat area
if "ready" in st.session_state and st.session_state["ready"]:

    question = st.chat_input("Ask your question here!")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("ai"):
            context_window = ""
            sources = []

            db = st.session_state["vectordb"]
            docs = db.similarity_search(question, k=3)

            for doc in docs:
                context_window += doc.page_content + "\n\n"
                file_name = doc.metadata.get("source", "unknown")
                file_page = doc.metadata.get("page", 0) + 1
                sources.append(f"**{file_name}** (page. {file_page})")

            prompt = f"""Act as an researcher,answer this question based on the given context.
                        if there are no matches context, answer "Information does not found!".

                        Context: {context_window}

                        Question: {question}
                        """

            llm = ChatGoogleGenerativeAI(
                model = "models/gemini-2.5-flash",
                temperature = 0,
                api_key=api_key
            )
            response = llm.invoke(prompt)

            st.markdown(response.content)

            if response.content != "Information does not found!":
                st.markdown("---")
                st.markdown("Citation: ")
                for sc in set(sources):
                    st.markdown(f"- {sc}")

else:
    st.info("Please upload PDFs on the left bar!")