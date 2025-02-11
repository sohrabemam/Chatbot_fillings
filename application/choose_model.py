# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=W0311
# pylint: disable=C0303
# pylint: disable=C0411

#importing the neccessay libraries.
from typing import Literal,Annotated, Sequence,TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import logging
import json
import re 

from application.config.config import Config
from application.chat.chatbot import load_faiss_db,text_chatbot,finance_chatbot
from application.chat.llm_init import llm_manager

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
    
TECH_INDEX_DIR=Config.TECH_INDEX_DIR

llm=llm_manager()
logger = logging.getLogger(__name__)
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


def check_query(state) -> Literal["finance", "text"]:
    """
    Determines whether the question is based on text or finance database
    """
    logging.debug("---CHECK RELEVANCE---")
    # Define a class for the model's output
    class Grade(BaseModel):
        """Custom score for relevance check."""
        relevance_score: str = Field(description="Relevance score, could be 'finance', 'text', etc.")
    llm_with_tool = llm.with_structured_output(Grade)
    # Define the prompt to classify the query
    prompt = PromptTemplate(
    template="""You are a classifier that determines which database to use based on the user query:
    
    - Use the **financial database** if the question is related to:
      - Cash flow statements
      - Balance sheets
      - Stock prices
      - Any other numerical financial data

    - Use the **text database** if the question is related to:
      - The ticker’s business model
      - Revenue of each product
      - Latest updates and growth factors
      - Development pipeline
      - 10-K, 8-K, proxy statements, insider information

    Here is the user's question:  
    **{question}**

    Based on the content of the question, determine the relevant database:
    - If the question is about financial data, assign **'finance'**.
    - If the question is about text-based information, assign **'text'**.
    """,
    input_variables=["question"],
    )


    # Create the chain with the prompt and the LLM
    chain = prompt | llm_with_tool
    print(state["messages"])
    messages = state["messages"]
    question = messages[-1].content
    # Extract the user query from the state
    #question = state["messages"][-1].content
    # Get the relevance score
    scored_result = chain.invoke({"question": question})
    score = scored_result.relevance_score.lower()
    # Return the relevance score (could be 'finance', 'text', or another label)
    print(f"---DECISION: {score.upper()} DATABASE RELEVANT---")
    return score  # Returning the score as lowercase ('finance' or 'text')




def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to whether to use finance or text

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    logging.debug("---CALL AGENT---")
    try:
        messages = state["messages"]
        logging.info(f"Messages before invoking LLM: {messages}")

        response = llm.invoke(messages)
        logging.info(f"Generated response: {response}")

        return {"messages": [response]}

    except Exception as e:
        logging.error(f"Error in agent: {e}")
        raise
  

    
def text(state):
  '''generate answer based on filings
  Args:
   state (messages): The current state
  Returns:
    str:  The updated state with answer to query
  '''
  logging.debug("Generating answer for text-based query")
  try:
      #messages = state["messages"] 
      #query = messages[0].content
      query = state["messages"][-1].content
      if TECH_INDEX_DIR is None:
            raise ValueError("TECH_INDEX_DIR is not set. Please check your environment variables.")
      print(f"TECH_INDEX_DIR: {TECH_INDEX_DIR}") 
      local_dir =TECH_INDEX_DIR
      retriever = load_faiss_db(local_dir)
      answer = text_chatbot(retriever, query)
      answer_type=answer.get("type","")
      message_content = answer.get("data", "")
      final_answer="type"+":"+answer_type+":"+","+message_content
      return {"messages": [final_answer]}

  except Exception as e:
      logging.error(f"Error in text function: {e}")
      raise



def finance(state):
    """
    Generate an answer based on financial data.

    Args:
        state (dict): The current state of messages.

    Returns:
        str: The updated state with the answer to the query.
    """
    logging.debug("Generating answer for finance-based query")
    try:
        #messages = state["messages"]
        #question = messages[0].content
        question = state["messages"][-1].content
        logging.info(f"Query received: {question}")
        answer= finance_chatbot(question)
        logging.info(f"Generated finance answer: {answer}") 
        answer_type=answer.get("type","")        
        message_content = answer.get("data", "")
        message_content_str = json.dumps(message_content,indent=4)
        message_content_json=json.loads(message_content_str)
        if answer_type=="table" :
        # Construct the final answer string
            final_answer = f"{message_content_json}"
        # elif answer_type=="image":
        #     final_answer = f"{answer_type,message_content_json}"
        else:
            final_answer="type"+":"+answer_type+":"+","+str(message_content)
        
        return {"messages": [final_answer]}

        
    except Exception as e:
        logging.error(f"Error in finance function: {e}")
        raise



def create_graph(agent,finance,text):
  '''Creates a graph, joins all nodes to form edges and return the graph'''
  logging.debug("Creating agent graph with nodes: agent, finance, text")
  try:
      workflow = StateGraph(AgentState)
      workflow.add_node("agent", agent)
      workflow.add_node("finance", finance)
      workflow.add_node("text", text)

      workflow.add_edge(START, "agent")
      workflow.add_conditional_edges("agent", check_query)
      workflow.add_edge("finance", END)
      workflow.add_edge("text", END)

      graph = workflow.compile()
      logging.info("Graph created successfully")
      
      return graph

  except Exception as e:
      logging.error(f"Error in create_graph: {e}")
      raise


def response(graph,query):
    '''Takes the structure of graph and query, and returns the response of the query.'''
    logging.debug("Generating response based on the graph")
    try:
        inputs = graph.invoke({"messages": [("user", query)]})
        response_message = inputs["messages"][-1].content
        logging.info(f"Generated response: {response_message}")
        # type_answer = response_message.split(':')[1].strip()
        data_answer = response_message.split(",", 1)[1].strip()
        types = ['image', 'table', 'text']
        type_answer = None  # Default to None if no type is found

        # Iterate over the types and check if any match
        for t in types:
            if t in response_message:
                type_answer = t
                break  # Stop once a match is found    
        logger.info(f"Answer type is {type_answer}")
        logger.info(f"Data is {data_answer}")  
        if type_answer=="table":
            response_json_table = {
                   "data":response_message
                }
            response_json_str=json.dumps(response_json_table)
            parsed_data = json.loads(response_json_str)
            parsed_data["data"] = json.loads(parsed_data["data"].replace("'", '"'))
            logger.info(type(parsed_data))
            # Load JSON string into a Python dictionary
            cleaned_json = json.dumps(parsed_data, indent=4)  
            parsed_data_clean = json.loads(cleaned_json)
            # Extract everything after the first "data" key
            nested_data = parsed_data_clean["data"]
            return nested_data
        
        # elif type_answer == "image":
        #     response_json_image = {
        #         "type":type_answer,
        #         "data": data_answer
        #     }
        #     response_json_str = json.dumps(response_json_image)
        #     parsed_data = json.loads(response_json_str)
            
        #     # # Process nested data within "data" field
        #     # parsed_data["data"] = json.loads(
        #     #     json.dumps(parsed_data["data"]).replace("'", '"')
        #     # )
            
        #     # Optional: Format cleaned JSON for better readability
        #     cleaned_json = json.dumps(parsed_data, indent=4)
        #     parsed_data_clean = json.loads(cleaned_json)
            
        #     # Extract everything after the first "data" key
        #     nested_data = parsed_data_clean["data"]
        #     return nested_data

        else:
                 
            response_json = {
                    "type": type_answer,
                    "data": data_answer
                }
            logger.info("Converted response_json to JSON format.")
                    
            logger.info(type(response_json))
            return response_json    
        
    except Exception as e:
        logging.error(f"Error in response function: {e}")
        raise
   