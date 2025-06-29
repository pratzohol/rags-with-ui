from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field

class LangchainLLM(LLM):
    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=self.headers)
        messages=[{"role":"system",
                   "content":"You are an AI Assistant. Find relevant information from the context provided by the user and then answer the question"},
                   {"role":"user",
                    "content":prompt}]
        response = client.chat.completions.create(model=self.model_name, messages=messages)
        print("Successfully generated the output of LLMs")
        return response.choices[0].message.content

    @property
    def _llm_type(self):
        return "Runnable instance of LLMs"
    
    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}