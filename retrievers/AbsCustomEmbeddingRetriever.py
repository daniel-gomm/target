from dataset_loaders.LoadersDataModels import QueryForTasksDataModel
from retrievers.AbsRetrieverBase import AbsRetrieverBase
from retrievers.RetrieversDataModels import RetrievalResultDataModel

from abc import abstractmethod
from typing import List, Dict, Iterable, Tuple


class AbsCustomEmbeddingRetriever(AbsRetrieverBase):
    """
    This interface includes the retrieve method and an encode method that doesn't expect a return value. If your retrieval tool already has table embedding/encoding persistence built in, this is the preferred class to inherit from for your retriever. At retrieval time, it is assumed that the **table embeddings are no longer needed to be provided** for the retrieval to work.
    Some possible reasons to inherit from this class and not `AbsStandardizedEmbeddingRetriever`:
    - you have a custom format of embedding for the tables (ie directory structure, file formats, etc).
    - your tool already deals with the persistence of the embedding.
    """

    def __init__(self, expected_corpus_format: str = "nested array"):
        """
        Parameters:
            expected_corpus_format (str, optional): a string indicating what corpus format (ie nested array, dictionary, pandas df, etc.) the `embed_corpus` function expects from its input.
        """
        super().__init__(expected_corpus_format=expected_corpus_format)

    def retrieve_batch(
        self,
        queries: List[QueryForTasksDataModel],
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[RetrievalResultDataModel]:
        retrieval_results = []
        for query in queries:
            retrieval_results.append(
                RetrievalResultDataModel(
                    dataset_name=dataset_name,
                    query_id=query.query_id,
                    retrieval_results=self.retrieve(
                        query.query, dataset_name, top_k, **kwargs
                    ),
                )
            )
        return retrieval_results

    @abstractmethod
    def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
        **kwargs,
    ) -> List[Tuple[int, str]]:
        """
        Directly retrieves the corresponding tables for the query. Works under the assumption that the embeddings are available when this function is called, and the retriever should be able to get the right tables with the query provided without any additional information about the corpus.

        Parameters:
            query (str): the actual query string.

            dataset_name (str): identifier for the dataset that these queries come from. since retrieval evaluation can be done for multiple datasets, use this as a way of choosing which dataset's corpus to retrieve from.

            top_k (int): the top k tables to retrieve for each query

            any additional kwargs you'd like to include.

        Returns:
            List[Tuple[int, str]]: the list of tuples each identifying one table retrieved, each tuple is the (database id, table id) of the retrieved table.
        """
        pass

    @abstractmethod
    def embed_corpus(self, dataset_name: str, corpus: Iterable[Tuple]) -> None:
        """
        The function to embed the given corpus. This will be called in the evaluation pipeline before any retrieval. The corpus given will be in the same format as self.expected_corpus_format for flexibility.

        Parameters:
            dataset_name (str): the name of the corpus dataset.
            corpus (Iterable[Dict[str, object]]): an iterable of tuples, each being a batch of entries in the corpus dataset, containing database id, table id, the table contents (which the user can assume is in the format of self.expected_corpus_format), and context metadata (in this order in the tuple).

        Returns:
            nothing. the persistence of the embedding must be dealt with the logic of this function itself, and the `retrieve` function should also know about the embedding results of this function so that retrieval can be done.
        """
        pass
