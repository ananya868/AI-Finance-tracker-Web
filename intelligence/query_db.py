from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict 
from langchain.chat_models import init_chat_model
from langchain import hub 
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from typing_extensions import Annotated 

import os 


class QueryDBPipeline: 
    """
    Query Database with a Question
    Args: 
        - question: str 
        - db_uri: SQLDatabase | uri 
        - top_k: int 
    Returns:
        - result: list 
    """
    def __init__(self, question, db_uri, top_k):
        self.question = question
        self.db_uri = db_uri
        self.top_k = top_k
    

    """Components"""
    # define chain state 
    class State(TypedDict):
        question: str 
        query: str 
        result: str 
        answer: str 
    

    class QueryOutput(TypedDict): 
        """Generate SQL query."""
        query: Annotated[str, ..., "Syntactically valid SQL query."]


    def config(self): 
        self.state = {"question": self.question}
        self.db = SQLDatabase.from_uri(self.db_uri)
        # Define LLM
        self.llm = llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        # Query prompt template
        self.query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        assert len(self.query_prompt_template.messages) == 1 
        print(":: Configured QueryDBPipeline ::")
    

    def write_query(self):
        """Generate SQL query to fetch information""" 
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 10,
                "table_info": self.db.get_table_info(),
                "input": self.state["question"],
            }
        )
        structured_llm = self.llm.with_structured_output(self.QueryOutput)
        result = structured_llm.invoke(prompt)
        self.state['query'] = result["query"]
        print(":: Generated SQL query :: -> ", self.state['query'])
        
    
    def execute_query(self): 
        """Execute SQL query""" 
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        self.state["result"] = execute_query_tool.invoke(self.state["query"])
        print(":: Executed SQL query  :: -> ", self.state["result"])


    def generate_answer(self):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {self.state["question"]}\n'
            f'SQL Query: {self.state["query"]}\n'
            f'SQL Result: {self.state["result"]}'
        )
        response = self.llm.invoke(prompt)
        print(":: Generated Answer    :: -> ", response.content)
        return response.content # LLM generated response
    

    def run(self):
        self.config()
        self.write_query()
        self.execute_query()
        return self.generate_answer()
    







    

    