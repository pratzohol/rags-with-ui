import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(input_texts, model_name="thenlper/gte-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str = "thenlper/gte-small"):
        self.model_name = model_name

    def embed_documents(self, texts):
        return get_embeddings(texts, model_name=self.model_name)

    def embed_query(self, text: str):
        emb_query = get_embeddings([text], model_name=self.model_name)[0]
        return emb_query
