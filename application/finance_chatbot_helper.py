# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=W0311
# pylint: disable=C0303
# pylint: disable=C0103

import os
import uuid
import json
import re
import boto3
import logging
import sys
import matplotlib
import matplotlib.pyplot as plt

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase

from application.config.config import Config
from  application.chat.llm_init import llm_manager
matplotlib.rcParams['backend'] = 'Agg'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

database_url =Config.DATABASE_URL

logger.info("Loading database url")
aws_access_key = Config.AWS_ACCESS_KEY
aws_secret_key = Config.AWS_SECRET_KEY
region_name = Config.REGION_NAME
bucket_name = Config.S3_BUCKET

# Initialize AWS S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name
)

db = SQLDatabase.from_uri(database_url, sample_rows_in_table_info=3)
llm = llm_manager()

def data_format(query, result):
    '''This function formats the data i.e if there is a table in result it will convert it to a pandas DataFrame.'''
    logger.info("Starting data formatting.")
    if result:
        try:
            prompt = PromptTemplate(template="""You are an AI assistant that helps convert {result} into a pandas DataFrame. Given the {query} and {result}, format the results into a structured pandas DataFrame. If the data cannot be structured into a DataFrame, return an appropriate message.he query results are a list of items or key-value pairs that you need to convert into a tabular format. Here are a few examples:1. If the result is a list of products and their prices: "Product A: $30, Product B: $40", convert it to a DataFrame with columns "Product" and "Price" 2. If the result is a list of numbers: "10, 20, 30, 40", convert it into a DataFrame with a column "Value".3. If the result is a dictionary of values: "City: Population, New York: 8.4M, Los Angeles: 3.9M", convert it into a DataFrame with "City" and "Population" columns.If the result does not match any of these types, ask for clarification or indicate that no table can be formed.The output should be the Python code to generate the pandas DataFrame, and the DataFrame should be formatted as a list of rows or a dictionary of column values.Keep Temperature as 0""", input_variables=['result', 'query'])
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"query": query, "result": result})
            pattern = r'```python(.*?)```'
            match = re.findall(pattern, response, re.DOTALL)
            extracted_code = "\n".join(match).strip()
            if not extracted_code:
               logger.warning("No valid Python code found in response.")
               return None
            # Execute the extracted code
            local_context = {}
            exec(extracted_code, globals(), local_context)
            df = local_context.get('df', None)
            if df is not None:
               logger.info("Data formatted successfully into DataFrame.")
               return df
            else:
               logger.warning("DataFrame not found after formatting.")
               return None
        except Exception as e:
            logger.error(f"Error formatting data: {e}")
            return None
    else:
        logger.warning("No result found for formatting.")
        return None

def alt_caption_text(query, result):
    '''This function generates alt text and captions based on the query and result.'''
    logger.info("Starting alt text and caption generation.")
    if result:
        try:
            prompt = PromptTemplate(template="""You are an AI assistant generates alternate text and caption based on {query},{result}.Return alternative text stored in the varibale Alt_text  and caption stored in the variable Caption. alt value is for html not for displayingso caption should be about the image for example 1. for a graph of revenue vs income alt would be AAPL Revenue vs Income Chart and caption would be Apple's Total Revenue vs Total Income for the Last 3 Years. Example 2. 'caption':'Line graph showing Apple's Total Revenue and Net Income for the fiscal years ending in 2021, 2022, and 2023.''alt': 'for the graph based on the given data'""", input_variables=['result', 'query'])
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"query": query, "result": result})
            alt_text_pattern = r"(?i)alt[_ ]?text[: ]\s*(.*)"
            caption_pattern = r"(?i)caption[: ]\s*(.*)"

            alt_text = re.search(alt_text_pattern, response)
            caption = re.search(caption_pattern, response)

            # Check if alt_text or caption was found and log appropriately
            logger.info(f"Alt Text: {alt_text.group(1).strip() if alt_text else 'No alt text generated.'}")
            logger.info(f"Caption: {caption.group(1).strip() if caption else 'No caption generated.'}")

            # Clean up alt text and caption
            alt_text_clean = re.sub(r'\'', '', alt_text.group(1).strip() if alt_text else "No alt text generated.")
            alt_text_clean = re.sub(r'\n', '', alt_text_clean)
            alt_text_clean = re.sub(r'[\*\?\_]=/', '', alt_text_clean)
            alt_text_clean = re.sub(r'[\\]', '', alt_text_clean)

            caption_clean = re.sub(r'\n', '', caption.group(1).strip() if caption else "No caption generated.")
            caption_clean = re.sub(r'[\*\?\_]=/', '', caption_clean)
            caption_clean = re.sub(r'[\\]', '', caption_clean)

            logger.info("Alt text and caption generated successfully.")
            return alt_text_clean, caption_clean
        except Exception as e:
            logger.error(f"Error generating alt text and caption: {e}")
            return "No alt text generated.", "No caption generated."
    else:
        logger.warning("No result found for generating alt text and caption.")
        return "No alt text generated.", "No caption generated."

def upload_to_s3(file_path, bucket_name, s3_key):
    """Function to upload plots to s3 bucket and returns the file URL."""
    logger.info(f"Uploading file {file_path} to S3 bucket {bucket_name} with key {s3_key}.")
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        logger.info(f"File uploaded successfully: {file_url}")
        return file_url
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return None

def make_viz(query, result, df, bucket_name,alt,caption):
    '''Generates the plot based on the result and queries.'''
    logger.info("Generating visualization.")
    # Define the directory path for saving plots
    path = "./plots"
    os.makedirs(path, exist_ok=True)

    # Generate a unique ID for the plot
    
    file_name = f"{uuid.uuid4()}.png"
    file_path = os.path.join(path, file_name)
    s3_key = f"InsightsThread_Chatbot_plots/{file_name}"

    if not result:
        return "No result found, so no plot generated."

    try:
        prompt = PromptTemplate(
            template="""You are an AI assistant that helps convert {query}, {result} into a visualization. 'Use seaborn or matplotlib, whichever gives the best graphs. Use python to create the graph and run the code to generate the plot. Always add plt.savefig("{path}/{file_name}") at the end to save the graph.""",
            input_variables=['query', 'result', 'df', 'path', 'file_name'])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": query, "result": result, "df": df, "path": path, "file_name": file_name})

        # Extract Python code block from the response
        pattern = r'```python(.*?)```'
        match = re.findall(pattern, response, re.DOTALL)

        if not match:
            logger.error("No valid Python code found in the response.")
            return "No valid Python code found in the response"

        # Extract the code and replace plt.show() with plt.savefig()
        extracted_code = "\n".join(match).strip()
        extracted_code = extracted_code.replace('plt.show()', f"plt.savefig('{file_path}')")
        print("file_path", file_path)
        # Execute the extracted code in a safe manner
        exec(extracted_code, globals())
        logger.info(f"Plot generated successfully and saved to {file_path}.")

        # Upload to S3 after saving locally
        print("before s3")
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)
        print("url", s3_url)
        if s3_url:
            print(s3_url)
            # After successful upload, delete the local file
            os.remove(file_path)
            logger.info(f"File {file_name} uploaded to S3 and local file removed.")
            viz_format={"type": "image","data": {"url": s3_url,"alt": alt,"caption": caption}}
            #logger.info("Type of viz_format",type(viz_format))
            logger.info(f"Type of viz_format after loads: {type(viz_format)}")

            return  viz_format # Return the S3 URL of the uploaded plot
        else:
            logger.error("Failed to upload to S3.")
            return "Failed to upload to S3"
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return f"Error in generating plot: {str(e)}"

def snake_to_title_case(snake_str: str) -> str:
    return  " ".join(word.title() for word in snake_str.split('_'))

def generate_table_as_json(df, result):
    '''Takes the df and returns it as json data, if the df has less than one row, result is returned.'''
    logger.info("Generating table as JSON.")
    answer = result
    if df.shape[0] >= 1:
        try:
            df = df.fillna(0)
            df.columns = [snake_to_title_case(col) for col in df.columns]
            headers = df.columns.tolist()
            rows = df.to_dict(orient="records")

            # Construct the final JSON structure
            table_json = {
                "type":"table",
                "data": {
                    "headers":headers,
                    "rows": rows
                }}

            # Convert to JSON string
            #json_str = json.dumps(table_json, indent=4)
            #json_data = json.loads(json_str)
            logger.info("Table successfully converted to JSON.")
            return table_json 
        except Exception as e:
            logger.error(f"Error generating JSON from table: {e}")
            return answer
    else:
        logger.info("DataFrame has no rows, returning original result.")
        return answer
