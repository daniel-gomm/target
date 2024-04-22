from generators.AbsGenerator import AbsGenerator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

import os

AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_API_VER = os.getenv("AZURE_OPENAI_API_VERSION")


class DefaultGenerator(AbsGenerator):
    def __init__(self):
        super().__init__()
        self.language_model = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_API_VER,
            temperature=0.0,
        )
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a data analyst who reads tables to answer questions."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    "Table: {table_str}\nQuery: {query_str}"
                ),
            ]
        )
        self.chain = self.chat_template | self.language_model

    def generate(self, table_str: str, query: str) -> str:
        return self.chain.invoke({"table_str": table_str, "query_str": query}).content
