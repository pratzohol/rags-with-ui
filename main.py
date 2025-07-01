import os
import traceback

import regex as re
import streamlit as st

from utilities.llm_helper import LLMHelper


def check_variables_in_prompt():
    # Check if "summaries" is present in the string custom_prompt
    if "{summaries}" not in st.session_state.custom_prompt:
        st.warning(
            """Your custom prompt doesn't contain the variable "{summaries}".
        This variable is used to add the content of the documents retrieved from the VectorStore to the prompt.
        Please add it to your custom prompt to use the app.
        Reverting to default prompt."""
        )
        st.session_state.custom_prompt = ""
    if "{question}" not in st.session_state.custom_prompt:
        st.warning(
            """Your custom prompt doesn't contain the variable "{question}".
        This variable is used to add the user's question to the prompt.
        Please add it to your custom prompt to use the app.
        Reverting to default prompt."""
        )
        st.session_state.custom_prompt = ""


def ask_followup_question(followup_question):
    st.session_state.askedquestion = followup_question
    st.session_state["input_message_key"] = st.session_state["input_message_key"] + 1


def questionAsked():
    st.session_state.askedquestion = st.session_state[
        "input" + str(st.session_state["input_message_key"])
    ]


def main():
    try:
        default_question = ""
        default_answer = ""

        if "question" not in st.session_state:
            st.session_state["question"] = default_question
        if "response" not in st.session_state:
            st.session_state["response"] = default_answer
        if "context" not in st.session_state:
            st.session_state["context"] = ""
        if "custom_prompt" not in st.session_state:
            st.session_state["custom_prompt"] = ""
        if "custom_temperature" not in st.session_state:
            st.session_state["custom_temperature"] = float(
                os.getenv("OPENAI_TEMPERATURE", 0.7)
            )

        if "sources" not in st.session_state:
            st.session_state["sources"] = ""
        if "followup_questions" not in st.session_state:
            st.session_state["followup_questions"] = []
        if "input_message_key" not in st.session_state:
            st.session_state["input_message_key"] = 1
        if "askedquestion" not in st.session_state:
            st.session_state.askedquestion = default_question

        # Set page layout to wide screen and menu item
        menu_items = {
            "Get help": None,
            "Report a bug": None,
            "About": """
            ## Embeddings App
            Embedding testing application.
            """,
        }
        st.set_page_config(layout="wide", menu_items=menu_items)

        llm_helper = LLMHelper(
            custom_prompt=st.session_state.custom_prompt,
            temperature=st.session_state.custom_temperature,
        )

        # Custom prompt variables
        custom_prompt_placeholder = """{summaries}
        Please reply to the question using only the text above.
        Question: {question}
        Answer:"""
        custom_prompt_help = """You can configure a custom prompt by adding the variables {summaries} and {question} to the prompt.
            {summaries} will be replaced with the content of the documents retrieved from the VectorStore.
            {question} will be replaced with the user's question.
        """

        col1, col2, col3 = st.columns([1, 2, 1])
        _, col2, col3 = st.columns([2, 2, 2])

        with col3:
            with st.expander("Settings"):
                st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    key="custom_temperature",
                )
                st.text_area(
                    "Custom Prompt",
                    key="custom_prompt",
                    on_change=check_variables_in_prompt,
                    placeholder=custom_prompt_placeholder,
                    help=custom_prompt_help,
                    height=150,
                )

        question = st.text_input(
            "Mistral-7B Semantic Answer",
            value=st.session_state["askedquestion"],
            key="input" + str(st.session_state["input_message_key"]),
            on_change=questionAsked,
        )

        # Answer the question if any
        if st.session_state.askedquestion != "":
            st.session_state["question"] = st.session_state.askedquestion
            st.session_state.askedquestion = ""
            (
                st.session_state["question"],
                st.session_state["response"],
                st.session_state["context"],
                st.session_state["sources"],
            ) = llm_helper.get_semantic_answer_lang_chain(
                st.session_state["question"], []
            )
            st.session_state["response"], followup_questions_list = (
                llm_helper.extract_followupquestions(st.session_state["response"])
            )
            st.session_state["followup_questions"] = followup_questions_list

        sourceList = []

        # Display the sources and context - even if the page is reloaded
        if st.session_state["sources"] or st.session_state["context"]:
            (
                st.session_state["response"],
                sourceList,
                matchedSourcesList,
                linkList,
                filenameList,
            ) = llm_helper.get_links_filenames(
                st.session_state["response"], st.session_state["sources"]
            )
            st.write("<br>", unsafe_allow_html=True)
            st.markdown("Answer: " + st.session_state["response"])

        # Display proposed follow-up questions which can be clicked on to ask that question automatically
        if len(st.session_state["followup_questions"]) > 0:
            st.write("<br>", unsafe_allow_html=True)
            st.markdown("**Proposed follow-up questions:**")
        with st.container():
            for questionId, followup_question in enumerate(
                st.session_state["followup_questions"]
            ):
                if followup_question:
                    str_followup_question = re.sub(
                        r"(^|[^\\\\])'", r"\1\\'", followup_question
                    )
                    st.button(
                        str_followup_question,
                        key=1000 + questionId,
                        on_click=ask_followup_question,
                        args=(followup_question,),
                    )

        if st.session_state["sources"] or st.session_state["context"]:
            # Buttons to display the context used to answer
            st.write("<br>", unsafe_allow_html=True)
            st.markdown("**Document sources:**")
            for id in range(len(sourceList)):
                st.markdown(f"[{id+1}] {sourceList[id]}")

            # Details on the question and answer context
            st.write("<br><br>", unsafe_allow_html=True)
            with st.expander("Question and Answer Context"):
                if (
                    not st.session_state["context"] is None
                    and st.session_state["context"] != []
                ):
                    for content_source in st.session_state["context"].keys():
                        st.markdown(f"#### {content_source}")
                        for context_text in st.session_state["context"][content_source]:
                            st.markdown(f"{context_text}")

                st.markdown(f"SOURCES: {st.session_state['sources']}")

        for questionId, followup_question in enumerate(
            st.session_state["followup_questions"]
        ):
            if followup_question:
                str_followup_question = re.sub(
                    r"(^|[^\\\\])'", r"\1\\'", followup_question
                )

    except Exception:
        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
