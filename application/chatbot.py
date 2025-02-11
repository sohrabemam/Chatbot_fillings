# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=W0311
# pylint: disable=C0303
# pylint: disable=C0103

# Importing required modules and setting up logging configuration
import logging
import json
from application.config.config import Config

# Importing necessary LangChain modules and custom helper functions
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

# Adding the project root to the system path for module imports
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Importing custom LLM manager and finance helper functions
from application.chat.llm_init import llm_manager
from application.chat.finance_chatbot_helper import *

# Initialize constants from configuration
database_url = Config.DATABASE_URL
open_ai_key = Config.OPEN_API_KEY
bucket_name = Config.S3_BUCKET

# Initializing the language model manager
llm = llm_manager()


def load_faiss_db(local_dir):
    ''' Load a FAISS vector database from the specified local directory.
        Args:
         local_dir: after fetching the folder from s3 it is stored in a local folder, Return the retriever
        Returns:
         retriever: A retriever object for similarity search. 
         '''
    logging.info(f"Loading FAISS DB from {local_dir}")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=open_ai_key)
        vector_store = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        logging.info("FAISS DB loaded successfully")
        return retriever
    except Exception as e:
        logging.error(f"Error loading FAISS DB: {e}")
        raise


def text_chatbot(retriever, query):
    """Processes user queries using a text-based chatbot integrated with a FAISS retriever.
        Args:
        retriever: FAISS retriever object for similarity-based search.
        query (str): User's query.
        Returns:
        dict: Response data in text format."""
    #joinging the root directory with additional information folder and info.json 
    file_path =os.path.join(PROJECT_ROOT,'additional_information', 'info.json')
    logging.info(f"Processing query: {query}")
    #reading path
    with open(file_path, 'r') as file:
        glossary_dict = json.load(file)
    #instructions to AI
    template = '''
    You are an AI assistant for a financial advisor, specializing in analyzing financial data to support stock  market investment decisions,{glossary_dict} has all the information about the different type of filings, form on which you have to answer user query. Understand how each filing and form is useful to the user. Use this gloassary_dict  to accurately answer usery query. Also provide numbers, trends, percent, as in quantitative data to support your analysis. Make sure the ans is format in a readable format.When no year is mentioned, consider data of most recent date.Avoid giving your opinion,always stick to facts.If specific years or metrics are mentioned, ensure your analysis directly references relevant data for those criteria.Highlight any trends, risks, or notable points that could impact financial decisions. 
    When responding to queries, include details on profit, sales figures, and percentage changes wherever applicable. Ensure the data is presented clearly, concisely, and supports the user's query. Highlight trends or notable statistics where relevant, providing a comprehensive view of the financial metrics.If query ask for last three years,we are currently in 2024, so go three years back.When analysing any query, mention the numerical data.In text based analysis, like summarising a business model,be brief, include all the relevant details,explain everything.While analysing, latest year's data make sure you mention which year you are considering.If data is not available return "No data available".
    {context}

    Question: {question}

    Helpful Answer:
    '''
    try:
        #adding custom prompt to the prompttemplate
        custom_rag_prompt = PromptTemplate.from_template(template)
        #building rag_chain
        rag_chain =({"context": retriever, "question": RunnablePassthrough(), 'glossary_dict': RunnableLambda(lambda x: glossary_dict)} | custom_rag_prompt | llm | StrOutputParser() )
        #invoking rag chain
        response = rag_chain.invoke(query)
        logging.info("Response generated successfully")
        #foramtting the output
        text_output={"type": "text", "data": response} 
        return text_output
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return "Error processing your request."


def finance_chatbot(query):
    """
    Processes finance-related queries using SQL database and generates text or visualizations.
    Args:
        query (str): User's query.
    Returns:
        dict: Response data in text, table, or image format.
    """
    logging.info(f"Received finance query: {query}")
    ticker_path=os.path.join(PROJECT_ROOT, 'additional_information', 'ticker.txt')
    ratio_path=os.path.join(PROJECT_ROOT,'additional_information', 'ratios.txt')
    query_path=os.path.join(PROJECT_ROOT,'additional_information', 'queries.txt')
    # try:
    with open(ticker_path,'r') as file:
        ticker = file.read()
    with open(ratio_path,'r') as file:
        ratio = file.read()
    with open(query_path,'r') as file:
        examples = file.read()
        
    # Connecting to the SQL database
    db = SQLDatabase.from_uri(database_url, sample_rows_in_table_info=3)
    query = str(query)
     # Creating a system message for the assistant
    sys_msg = SystemMessage(content=f'''You are a helpful assistant tasked with executing SQL queries and visualizing results when needed 1. **SQL Query Execution**: If the user asks a question related to data, generate and execute the appropriate SQL query to retrieve the data from the database. Return the SQL query and its result.While resulting errors, make sure that to not assume the number of results to output. Show all the output. 
    1. Keep Temperature as 0.
    2. **Plot or Graph Requests**: If the user requests visualization (using terms like "plot," "graph," "visualize," "chart," "generate"), pass the query and result to `data_format` and `viz_rec` to get the graph.

    3. **Output Formatting**: Ensure all output is structured properly. Return tables as DataFrames.

    4. **Ratio Calculations**: If the user asks for a ratio not found in the database, refer to the provided {ratio} dictionary and calculate it using the formula available.

    5. **Glossary Reference**: For any specific term like "ticker" or others not directly related to data queries, refer to {ticker} for additional information.

    6. **Query Examples**: Use {examples} as a reference for formulating SQL queries when needed.

    7. **Token Limit Handling**: If a token limit error occurs when returning the result, display only the first 100 rows of the output to avoid exceeding the limit.

    8. **Output Limit Conditions**: If no specific limit is mentioned, return the output in full without truncating.

    9. **Multiple Ticker and Year Handling**: If the question mentions multiple tickers (e.g., "AAPL, TSLA, MSFT") or years, handle each ticker or year separately by providing individual results.
    
    10. If the output is 100000000 then the output should be 10 million.
    11.If data is not available return "No data available".
   ''')

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # Creating and running the agent
    agent_executor = create_react_agent(llm, toolkit.get_tools(), state_modifier=sys_msg)


    result = ''
    events = list(agent_executor.stream({"messages": [("user", query)]}, stream_mode="values"))
    last_event_message = events[-1]["messages"][-1]
    content = last_event_message.content if hasattr(last_event_message, 'content') else None
    result = content
    result = str(result)
    df = data_format(query, result)
    logging.info("SQL query executed and formatted successfully")
    # Handling visualization requests
    keywords=["plot", "graph", "visualize", "generate", "create", "show"]
    #if query contains of any keyword then it is a image query
    if any(keyword in query.lower() for keyword in keywords):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        # Ensure DataFrame contains numerical data before plotting
        if df is None or df.empty or not any(df.dtypes.apply(lambda x: x in ['int64', 'float64'])):
            logging.warning("⚠️ No valid numerical data available for plotting.")
            no_image = {
                "type": "image",
                "data": {
                    "url": "No valid data available for plotting.",
                    "alt": "No valid data available.",
                    "caption": "No valid data available for graph generation."
                }
            }
            return no_image

        # Generate alternative text and caption
        alt, caption = alt_caption_text(query, result)
        
        # Try generating the graph
        try:
            viz_response = make_viz(query, result, df, bucket_name, alt, caption)
            if isinstance(viz_response, dict) and viz_response.get("type") == "image":
                logging.info("✅ Visualization successfully generated.")
                return viz_response  
            else:
                logging.warning("❌ Failed to generate plot")
                no_image = {
                    "type": "image",
                    "data": {
                        "url": "Failed to generate plot.",
                        "alt": "Graph generation failed.",
                        "caption": "Graph could not be created due to missing data."
                    }
                }
                return no_image
        except Exception as e:
            logging.error(f"❌ Error generating visualization: {e}")
            return {
                "type": "image",
                "data": {
                    "url": "Error generating plot.",
                    "alt": "Plot generation error.",
                    "caption": "An error occurred while creating the graph."
                }
            }

    else:
        try:
            #loading table in json format
            json_data = generate_table_as_json(df, result)
            #if table is in string then return text output
            if isinstance(json_data, str):
                text_output={"type": "text","data": result}
                return text_output
            else:
                try:
                    #if json has only row return text output
                    num_rows = len(json_data["data"]["rows"])
                    if num_rows == 1:
                        text_output={"type": "text","data": result}
                        return text_output
                    else:
                        #If json has more than one row then return the output in table format
                        table_output={"type": "table","data": json_data}
                        return table_output
                except Exception as e:
                    logging.error(f"Error generating table as JSON: {e}")
                    return "We are experiencing high traffic and can’t process your request right now. Please try again later. Thank you for your patience!"
    
        except Exception as e:
            logging.error(f"Error processing finance query: {e}")
            return "Error processing your request."
