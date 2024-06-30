from dataset_loaders import Text2SQLDatasetLoader
from dataset_loaders.AbsDatasetLoader import AbsDatasetLoader
from dataset_loaders.LoadersDataModels import (
    DatasetConfigDataModel,
)
from dataset_loaders.TargetDatasetConfig import *

from dictionary_keys import (
    ANSWER_COL_NAME,
    QUERY_COL_NAME,
    QUERY_ID_COL_NAME,
    DATABASE_ID_COL_NAME,
    DIFFICULTY_COL_NAME,
)

from generators.AbsGenerator import AbsGenerator
from generators.DefaultGenerator import DefaultGenerator
from generators.GeneratorPrompts import TEXT2SQL_SYSTEM_PROMPT, TEXT2SQL_USER_PROMPT
from generators.GeneratorsDataModels import DownstreamGeneratedResultDataModel

from retrievers.RetrieversDataModels import RetrievalResultDataModel

from tasks.AbsTask import AbsTask
from tasks.TasksDataModels import (
    Text2SQLTaskPerformanceDataModel,
)
from tasks.utils import evaluate_sql_execution
from pathlib import Path
import sqlite3
from typing import List, Dict, Literal, Union


class Text2SQLTask(AbsTask):

    AVAILABLE_METRICS = set(["execution_accuracy", "execution_ves"])
    DEFAULT_METRICS = set(["execution_accuracy"])

    def __init__(
        self,
        datasets_config: Dict[str, Dict[str, str]] = None,
        overwrite_default_datasets: bool = False,
        task_generator: AbsGenerator = None,
        metrics: Union[str, List[str]] = list(DEFAULT_METRICS),
        **kwargs,
    ):
        assert (
            datasets_config == None
        ), "currently text2sql task doesn't accept custom dataset config. update coming soon..."
        if task_generator == None:
            task_generator = DefaultGenerator(
                system_message=TEXT2SQL_SYSTEM_PROMPT,
                user_message=TEXT2SQL_USER_PROMPT,
            )
        super().__init__(
            task_name=self.get_default_task_name(),
            datasets_config=None,
            overwrite_default_datasets=False,
            task_generator=task_generator,
            **kwargs,
        )

        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            if metric not in Text2SQLTask.AVAILABLE_METRICS:
                raise ValueError(
                    f"the metric {metric} is not one of the available metrics!"
                )
        self.include_ves = False
        if "execution_ves" in metrics:
            self.include_ves = True

        # two lists, pred_sql contains the predicted sql queries, and ref_sql contains the ground truth sql queries.
        self.pred_sql = []
        self.ref_sql = []
        self.difficulties = []
        self.current_dataset: str = None
        self.database_dirs: Dict[str, str] = None

    @classmethod
    def get_default_task_name(cls) -> str:
        return "Text to SQL Task"

    @classmethod
    def get_available_metrics(cls) -> str:
        return str(Text2SQLTask.AVAILABLE_METRICS)

    def setup_database_dirs(self, dataloaders: Dict[str, Text2SQLDatasetLoader]):
        self.database_dirs = {
            name: loader.path_to_database_dir for name, loader in dataloaders.items()
        }

    def _get_default_dataset_config(self) -> Dict[str, DatasetConfigDataModel]:
        """
        Returns the default dataset config for fact verification.
        Includes the following datasets:
            TabFact
            TODO: more to come
        """
        return {  # TODO: FIX with actual text2sql dataset configs
            DEFAULT_TABFACT_DATASET_CONFIG.dataset_name: DEFAULT_TABFACT_DATASET_CONFIG,
        }

    def _get_schema(self, dataset_name: str, database_id: str):
        if dataset_name not in self.database_dirs:
            raise ValueError(
                f"dataset {dataset_name} does not have a database directory setup."
            )
        db_path = Path(
            self.database_dirs[dataset_name], database_id, f"{database_id}.sqlite"
        )
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_schema WHERE type='table'")

        # Fetch and print the schema of each table
        tables = cur.fetchall()
        schema_str = f"Database Name: {database_id}\n"
        for table in tables:
            schema_str += f"Table Name: {table[0]}\n Schema:\n{table[1]}\n"
        return schema_str

    def _get_downstream_task_results(
        self,
        query_batch: Dict[str, List],
        retrieval_results: List[RetrievalResultDataModel],
        dataset_name: str,
    ) -> List[DownstreamGeneratedResultDataModel]:
        """
        Given the query and the retrieval results, generate downstream task results. Uses fact verification tasks's default generator to accept or refute the claim, or say there's not enough information.
        """
        if not self.current_dataset:
            self.current_dataset = dataset_name
        return [
            DownstreamGeneratedResultDataModel(
                dataset_name=dataset_name,
                query_id=query_id,
                generated_results=self.task_generator.generate(
                    table_str="\n".join(
                        self._get_schema(id[0]) for id in result.retrieval_results
                    ),
                    query=query_str,
                ),
            )
            for query_id, query_str, result in zip(
                query_batch[QUERY_ID_COL_NAME],
                query_batch[QUERY_COL_NAME],
                retrieval_results,
            )
        ]

    def _update_downstream_task_metrics(
        self,
        query_batch: Dict[str, List],
        downstream_results: List[DownstreamGeneratedResultDataModel],
    ) -> None:
        """
        Update metric tracked for fact verification's performance calculation.
        Specifically, update the `self.pred_answers` and `self.ref_answers` lists
        based on the predicted answers in downstream_results and ground truth answers in query_batch.
        """
        self.pred_answers.extend(
            [
                downstream_answer.generated_results
                for downstream_answer in downstream_results
            ]
        )
        for downstream_answer in downstream_results:
            split_list = downstream_answer.generated_results.split("\t", 1)
            if len(split_list) < 2:
                raise ValueError(
                    f"could not parse the sql and the corresponding database. given result: {downstream_answer.generated_results}"
                )
            self.pred_sql.append((split_list[0], split_list[1]))
        self.ref_sql.extend(
            list(zip(query_batch[ANSWER_COL_NAME], query_batch[DATABASE_ID_COL_NAME]))
        )
        self.difficulties.extend(query_batch[DIFFICULTY_COL_NAME])

    def _calculate_downstream_task_performance(
        self, **kwargs
    ) -> Text2SQLTaskPerformanceDataModel:
        """
        Calculate downstream task metrics for the fact verification task.
        Metrics computed: accuracy, f1, precision, and recall.
        """
        if self.current_dataset not in self.database_dirs:
            raise ValueError(
                f"{self.current_dataset} does not have path to database files."
            )
        db_path = self.database_dirs[self.current_dataset]

        result = Text2SQLTaskPerformanceDataModel(
            scores=evaluate_sql_execution(
                self.pred_sql,
                self.ref_sql,
                self.difficulties,
                db_path,
                1,
                60,
                self.include_ves,
            )
        )

        self.pred_sql = []
        self.ref_sql = []
        self.difficulties = []
        self.current_dataset = None
        return result
