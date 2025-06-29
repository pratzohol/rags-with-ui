import streamlit as st
import os
import traceback
from utilities.llm_helper import LLMHelper
import streamlit.components.v1 as components
from urllib import parse

def delete_embeddings_of_file(file_to_delete):
    # Query RediSearch to get all the embeddings - lazy loading
    if 'data_files_embeddings' not in st.session_state:
        st.session_state['data_files_embeddings'] = llm_helper.get_all_documents(k=1000)

    if st.session_state['data_files_embeddings'].shape[0] == 0:
        return

    for converted_file_extension in ['.txt']:
        file_to_delete = 'converted/' + file_to_delete + converted_file_extension

        embeddings_to_delete = st.session_state['data_files_embeddings'][st.session_state['data_files_embeddings']['filename'] == file_to_delete]['key'].tolist()
        embeddings_to_delete = list(map(lambda x: f"{x}", embeddings_to_delete))
        if len(embeddings_to_delete) > 0:
            llm_helper.vector_store.delete_keys(embeddings_to_delete)
            # remove all embeddings lines for the filename from session state
            st.session_state['data_files_embeddings'] = st.session_state['data_files_embeddings'].drop(st.session_state['data_files_embeddings'][st.session_state['data_files_embeddings']['filename'] == file_to_delete].index)

def delete_file_and_embeddings(bucket_name='',filename=''):
    if filename == '':
        filename = st.session_state['file_and_embeddings_to_drop'] # get the current selected filename

    file_dict = [d for d in st.session_state['data_files'] if (d == filename or d.startswith(filename.split('.')[0] + '_chunk_'))]

    if len(file_dict) > 0:
        for file in file_dict:
            source_file = file
            try:
                llm_helper.blob_client.delete_file(bucket_name,source_file)
            except Exception as e:
                st.error(f"Error deleting file: {source_file} - {e}")

    # update the list of filenames to remove the deleted filename
    st.session_state['data_files'] = [d for d in st.session_state['data_files'] if d!= '{filename}' and not d.startswith(filename.split('.')[0] + '_chunk_')]


def delete_all_files_and_embeddings():
    files_list = st.session_state['data_files']
    for filename_dict in files_list:
        delete_file_and_embeddings(filename_dict['filename'])

try:
    # Set page layout to wide screen and menu item
    menu_items = {
	'Get help': None,
	'Report a bug': None,
	'About': '''
	 ## Embeddings App

	Document Reader Sample Demo.
	'''
    }
    st.set_page_config(layout="wide", menu_items=menu_items)

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    llm_helper = LLMHelper()

    st.session_state['data_files'] = llm_helper.blob_client.get_all_files(bucket_name,index=False)

    if len(st.session_state['data_files']) == 0 and bucket_name!="--Select--":
        st.warning("No files found. Go to the 'Add Document' tab to insert your docs.")

    else:
        filenames_list = [s for s in st.session_state['data_files'] if '_chunk_' not in s]
        st.dataframe(filenames_list, use_container_width=True)

        st.text("")
        st.text("")
        st.text("")

        filename=st.selectbox("Select filename to delete", filenames_list, key="file_and_embeddings_to_drop")

        st.text("")
        st.button("Delete file and its embeddings", on_click=delete_file_and_embeddings,args=(bucket_name,filename,))
        st.text("")
        st.text("")

        if len(filenames_list) > 1:
            st.button("Delete all files (with their embeddings)", type="secondary", on_click=delete_all_files_and_embeddings, args=None, kwargs=None)

except Exception as e:
    st.error(traceback.format_exc())
