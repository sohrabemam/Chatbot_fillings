# loading all the libraries
import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from application.choose_model import *
from application.chat.chatbot import *
# Initialize Flask app
app = Flask(__name__)
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS","http://127.0.0.1:3000,http://localhost:3000,https://insighthread.com").split(",")     
CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)
# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
    ]
)
@app.route('/query', methods=['POST'])
def query():
    """
    Flask endpoint to handle user query.
    Expects JSON input with a 'query' field.
    
    Request Body :
      {"query": "what is the Microsoft's revenue for last three  years?" } #table query
      {"query": "Plot Microsoft's revenue for last three  years" } #image query
      {"query": "What is apple's business model" } #text query
      
    """
    try:
        # Get query from the incoming JSON request
        data = request.get_json()
        query_text = data.get('query')
        
        if not query_text:
            return jsonify({"error": "Query is required"}), 400

        # Assuming graph is created in your existing code
        graph = create_graph(agent, finance, text)
        
        # Call the response function with the query and graph
        response_message = response(graph, query_text)
        
        
        
        # Return the response to the user
        return jsonify(response_message)

    except Exception as e:
        logging.error(f"Error in query route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=False)
    
#commands to run app.py on macos
#be in application folder
# export FLASK_APP=app/app.py FLASK_ENV=development && flask run
#gunicorn application.app:app --preload -b 0.0.0.0:8080
#FLASK_ENV=development python application/app.py

#commands to run app.py on windows
#Create virtual environment python -m venv myvenv
# install all dependencies using requirement.txt e.g pip install -r requirement.txt 
# go to application directory using cd commmand 
#  then run this $env:FLASK_APP="app.py"   
# then $env:FLASK_ENV="development"  
# finally run : "flask run" this command will run the server 
# go to postman if not already installed on vs code install it
# move to postman put the http://127.0.0.1:5000/query 
# then change method to post go to body then raw 
# then paste query and change type to json then send this request 
# you get answer from custom trained openai

