import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
llm_model = "gpt-3.5-turbo-0301"
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
llm_replacement_model = OpenAI(temperature=0,model='gpt-3.5-turbo-instruct')
response = index.query(query,llm = llm_replacement_model)
display(Markdown(response))
