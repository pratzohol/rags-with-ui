import streamlit as st
import os, json, re, io
from os import path
import traceback
from utilities.llm_helper import LLMHelper
from utilities.embedding_hf_infer import get_embeddings
from utilities.chunking import chunk_pdf_and_upload
from utilities.milvus import MilvusHelper
import uuid
from urllib import parse
from pymilvus import DataType, FieldSchema


def upload_file(file_content, filename:str, bucket:str,index_name:str):
    # Upload a new file
    st.session_state['filename'] = filename
    st.session_state['file_url'] = llm_helper.blob_client.put_objects([file_content],bucket,filename,chunk=False,index_name=index_name)


try:
    menu_items = {
        'Get help': None,
        'Report a bug': None,
        'About': '''
        ## Embeddings App
        Embedding testing application.
        '''
    }
    st.set_page_config(layout="wide", menu_items=menu_items)

    llm_helper = LLMHelper()

    bucket_name = os.getenv('BUCKET_NAME')
    existing_indices=llm_helper.blob_client.get_all_files(bucket_name)
    existing_indices.append("CREATE NEW INDEX")

    index_name=st.selectbox("Existing Indices in the Bucket", existing_indices)
    if index_name=="CREATE NEW INDEX":
        index_name=st.text_input(f"Enter the new index name for the selected bucket - {bucket_name}")


    with st.expander("Add a single document to the knowledge base", expanded=True):
        uploaded_file = st.file_uploader("Upload a document to add it to the knowledge base", type=['pdf', 'txt'])
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # if st.session_state.get('filename', '') != uploaded_file.name:
            upload_file(bytes_data, uploaded_file.name,bucket_name,index_name)

            # Chunking the PDF
            chunks, chunks_url, chunks_download_url = chunk_pdf_and_upload(bytes_data, bucket_name, uploaded_file.name, index_name)

            # Add the embeddings to the knowledge base
            embeddings = get_embeddings(chunks)
            print("Successfully obtained the embeddings of every chunk")

            # Uploading to Milvus
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="urls", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
            ]
            milvus_client = MilvusHelper(fields)
            data = [{"id": i, "urls": chunks_url[i], "source": chunks_download_url[i], "text": chunks[i], "embeddings": embeddings[i]} for i in range(len(chunks))]

            milvus_client.upload_data(data)

            st.success(f"File {uploaded_file.name} embeddings added to the knowledge base.")

except Exception as e:
    st.error(traceback.format_exc())
