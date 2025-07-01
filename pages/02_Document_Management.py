import os
import traceback

import streamlit as st

from utilities.llm_helper import LLMHelper


def delete_file_and_embeddings(filename=""):
    if filename == "":
        filename = st.session_state[
            "file_and_embeddings_to_drop"
        ]  # get the current selected filename

    # update the list of filenames to remove the deleted filename
    st.session_state["data_files"] = [
        d
        for d in st.session_state["data_files"]
        if d != "{filename}" and not d.startswith(filename.split(".")[0] + "_chunk_")
    ]


def delete_all_files_and_embeddings():
    files_list = st.session_state["data_files"]
    for filename in files_list:
        delete_file_and_embeddings(filename)


try:
    # Set page layout to wide screen and menu item
    menu_items = {
        "Get help": None,
        "Report a bug": None,
        "About": """
	 ## Embeddings App

	Document Reader Sample Demo.
	""",
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

    if len(st.session_state["data_files"]) == 0:
        st.warning("No files found. Go to the 'Add Document' tab to insert your docs.")
    else:
        filenames_list = [
            s for s in st.session_state["data_files"] if "_chunk_" not in s
        ]
        st.dataframe(filenames_list, use_container_width=True)

        st.text("")
        st.text("")
        st.text("")

        filename = st.selectbox(
            "Select filename to delete",
            filenames_list,
            key="file_and_embeddings_to_drop",
        )

        st.text("")
        st.button(
            "Delete file and its embeddings",
            on_click=delete_file_and_embeddings,
            args=(filename,),
        )
        st.text("")
        st.text("")

        if len(filenames_list) > 1:
            st.button(
                "Delete all files (with their embeddings)",
                type="secondary",
                on_click=delete_all_files_and_embeddings,
                args=None,
                kwargs=None,
            )

except Exception as e:
    st.error(traceback.format_exc())
