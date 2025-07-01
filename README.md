# RAGs-with-UI

Simple chatbot UI implementing RAG pipeline using `mistralai/Mistral-7B-Instruct-v0.1` as base LLM and `Milvus` as vector database with the feature to chunk and upload .pdf files as our internal knowledge base.



## Notes

To use the LLM, you need to go to this link https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 and request access (granted immediately), then run `huggingface-cli login` in terminal and paste the read token of huggingace (https://huggingface.co/settings/tokens).