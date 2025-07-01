import traceback
from urllib import parse

import streamlit as st
from pymilvus import DataType, FieldSchema

from utilities.chunking import chunk_pdf_and_upload
from utilities.embedding_hf_infer import get_embeddings
from utilities.llm_helper import LLMHelper
from utilities.milvus import MilvusHelper

try:
    menu_items = {
        "Get help": None,
        "Report a bug": None,
        "About": """
        ## Embeddings App
        Embedding testing application.
        """,
    }
    st.set_page_config(layout="wide", menu_items=menu_items)

    llm_helper = LLMHelper()

    st.session_state["data_files"] = []

    with st.expander("Add a single document to the knowledge base", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload a document to add it to the knowledge base", type=["pdf", "txt"]
        )
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # Chunking the PDF
            chunks = chunk_pdf_and_upload(bytes_data)

            # Add the embeddings to the knowledge base
            embeddings = get_embeddings(chunks)
            print("Successfully obtained the embeddings of every chunk")

            # Uploading to Milvus
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(
                    name="embeddings",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=len(embeddings[0]),
                ),
            ]
            milvus_client = MilvusHelper(fields)
            data = [
                {
                    "id": i,
                    "text": chunks[i],
                    "metadata": {"fileName": uploaded_file.name},
                    "embeddings": embeddings[i],
                }
                for i in range(len(chunks))
            ]

            milvus_client.upload_data(data)

            st.session_state["data_files"].append(uploaded_file.name)

            st.success(
                f"File {uploaded_file.name} embeddings added to the knowledge base."
            )

except Exception as e:
    st.error(traceback.format_exc())
