import re

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_milvus.vectorstores import Milvus

from utilities.customprompt import PROMPT
from utilities.embedding_hf_infer import EmbeddingModel
from utilities.llm_hf_infer import MistralLLM


class LLMHelper:
    def __init__(
        self,
        embeddings=EmbeddingModel("thenlper/gte-small"),
        llm=MistralLLM,
        k: int = 5,
    ):
        self.embeddings = embeddings
        self.llm = llm
        self.k = k

        # Initialize Milvus as vector store
        connection_args = {"uri": "rag.db"}
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"nlist": 2},
        }

        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name="vector_emb",
            index_params=index_params,
            consistency_level="Strong",
            primary_field="id",
            connection_args=connection_args,
        )

    def get_semantic_answer_lang_chain(self, question, chat_history):
        question_generator = LLMChain(
            llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False
        )
        doc_chain = load_qa_with_sources_chain(
            self.llm, chain_type="stuff", verbose=False, prompt=PROMPT
        )
        chain = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
        )
        result = chain({"question": question, "chat_history": chat_history})

        breakpoint()
        sources = "\n".join(
            set(map(lambda x: x.metadata["fileName"], result["source_documents"]))
        )
        return question, result["answer"].split("Answer: ")[1], sources

    def get_completion(self, prompt, **kwargs):
        return self.llm(prompt)

    def extract_followupquestions(self, answer):
        followupTag = answer.find("Follow-up Questions")
        followupQuestions = answer.find("<<")

        # take min of followupTag and folloupQuestions if not -1 to avoid taking the followup questions if there is no followupTag
        followupTag = (
            min(followupTag, followupQuestions)
            if followupTag != -1 and followupQuestions != -1
            else max(followupTag, followupQuestions)
        )
        answer_without_followupquestions = (
            answer[:followupTag] if followupTag != -1 else answer
        )
        followup_questions = answer[followupTag:].strip() if followupTag != -1 else ""

        # Extract the followup questions as a list
        pattern = r"\<\<(.*?)\>\>"
        match = re.search(pattern, followup_questions)
        followup_questions_list = []
        while match:
            followup_questions_list.append(
                followup_questions[match.start() + 2 : match.end() - 2]
            )
            followup_questions = followup_questions[match.end() :]
            match = re.search(pattern, followup_questions)

        if followup_questions_list != "":
            # Extract follow up question
            pattern = r"\d. (.*)"
            match = re.search(pattern, followup_questions)
            while match:
                followup_questions_list.append(
                    followup_questions[match.start() + 3 : match.end()]
                )
                followup_questions = followup_questions[match.end() :]
                match = re.search(pattern, followup_questions)

        if followup_questions_list != "":
            pattern = r"Follow-up Question: (.*)"
            match = re.search(pattern, followup_questions)
            while match:
                followup_questions_list.append(
                    followup_questions[match.start() + 19 : match.end()]
                )
                followup_questions = followup_questions[match.end() :]
                match = re.search(pattern, followup_questions)

        # Special case when 'Follow-up questions:' appears in the answer after the <<
        followupTag = answer_without_followupquestions.lower().find(
            "follow-up questions"
        )
        if followupTag != -1:
            answer_without_followupquestions = answer_without_followupquestions[
                :followupTag
            ]
        followupTag = answer_without_followupquestions.lower().find(
            "follow up questions"
        )  # LLM can make variations...
        if followupTag != -1:
            answer_without_followupquestions = answer_without_followupquestions[
                :followupTag
            ]

        return answer_without_followupquestions, followup_questions_list

    def insert_citations_in_answer(self, answer, filenameList):
        filenameList_lowered = [
            x.lower() for x in filenameList
        ]  # LLM can make case mitakes in returing the filename of the source

        matched_sources = []
        pattern = r"\[\[(.*?)\]\]"
        match = re.search(pattern, answer)
        while match:
            filename = match.group(1).split(".")[
                0
            ]  # remove any extension to the name of the source document
            if filename in filenameList:
                if filename not in matched_sources:
                    matched_sources.append(filename.lower())
                filenameIndex = filenameList.index(filename) + 1
                answer = (
                    answer[: match.start()]
                    + "$^{"
                    + f"{filenameIndex}"
                    + "}$"
                    + answer[match.end() :]
                )
            else:
                answer = (
                    answer[: match.start()]
                    + "$^{"
                    + f"{filename.lower()}"
                    + "}$"
                    + answer[match.end() :]
                )
            match = re.search(pattern, answer)

        # When page is reloaded search for references already added to the answer (e.g. '${(id+1)}')
        for id, filename in enumerate(filenameList_lowered):
            reference = "$^{" + f"{id+1}" + "}$"
            if reference in answer and not filename in matched_sources:
                matched_sources.append(filename)

        return answer, matched_sources, filenameList_lowered

    def get_links_filenames(self, answer, sources):
        split_sources = sources.split(
            "\n"
        )  # soures are expected to be of format '  \n  [filename1.ext](sourcelink1)  \n [filename2.ext](sourcelink2)  \n  [filename3.ext](sourcelink3)  \n '
        srcList = []
        linkList = []
        filenameList = []
        for src in split_sources:
            if src != "":
                srcList.append(src)
                linkList.append(src)
                filename = src.split("/")[-1]  # retrieve the source filename.
                answer = answer.replace(
                    src, filename
                )  # if LLM added a path to the filename, remove it from the answer
                filenameList.append(filename)

        answer, matchedSourcesList, filenameList = self.insert_citations_in_answer(
            answer, filenameList
        )  # Add (1), (2), (3) to the answer to indicate the source of the answer

        return answer, srcList, matchedSourcesList, linkList, filenameList

    def clean_encoding(self, text):
        try:
            encoding = "ISO-8859-1"
            encodedtext = text.encode(encoding)
            encodedtext = encodedtext.decode("utf-8")
        except Exception as e:
            encodedtext = text
        return encodedtext
