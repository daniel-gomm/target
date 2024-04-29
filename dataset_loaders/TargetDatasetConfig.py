from dataset_loaders.LoadersDataModels import HFDatasetConfigDataModel

DEFAULT_FETAQA_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="fetaqa",
    hf_corpus_dataset_path="target-benchmark/fetaqa-corpus",
    hf_queries_dataset_path="target-benchmark/fetaqa-queries",
)

DEFAULT_TABFACT_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="tab-fact",
    hf_corpus_dataset_path="target-benchmark/tabfact-corpus",
    hf_queries_dataset_path="target-benchmark/tabfact-queries",
)

DEFAULT_WIKITQ_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="wikitq",
    hf_corpus_dataset_path="target-benchmark/wikitq-corpus",
    hf_queries_dataset_path="target-benchmark/wikitq-queries",
)

DEFAULT_INFAGENTDA_DATASET_CONFIG = HFDatasetConfigDataModel(
    dataset_name="infiagent-da",
    hf_corpus_dataset_path="target-benchmark/infiagentda-corpus",
    hf_queries_dataset_path="target-benchmark/infiagentda-queries",
)