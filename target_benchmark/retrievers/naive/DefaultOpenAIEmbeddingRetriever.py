from typing import Dict, Union, List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
from target_benchmark.retrievers.utils import markdown_table_str


class OpenAIEmbedder(AbsStandardEmbeddingRetriever):
    def __init__(
        self,
        expected_corpus_format: str = "nested array",
        num_rows: Union[int, None] = None,
        embedding_batch_size: Optional[int] = None
    ):
        super().__init__(expected_corpus_format=expected_corpus_format, embedding_batch_size=embedding_batch_size)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.num_rows = num_rows

    def embed_query(
        self,
        query: str,
        dataset_name: str,
        **kwargs,
    ) -> np.ndarray:
        emb = self.create_embedding(query)
        return np.array(emb)

    def batch_embed_queries(self, queries: List[str], dataset_name: str) -> List[np.ndarray]:
        return self._create_embeddings(queries)

    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        table_str = markdown_table_str(corpus_entry["table"], self.num_rows)
        emb = self.create_embedding(table_str)
        return np.array(emb)

    def batch_embed_corpora(self, dataset_name: str, corpus_entries: List[Dict]) -> List[np.ndarray]:
        table_strs = [markdown_table_str(corpus_entry["table"], self.num_rows) for corpus_entry in corpus_entries]
        return self._create_embeddings(table_strs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=32),
    )
    def create_embedding(self, text: str):
        return self.embedding_model.embed_query(text)

    def _create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return [np.array(emb) for emb in self.embedding_model.embed_documents(texts)]
