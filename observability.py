import os

from dotenv import load_dotenv

load_dotenv(".env")

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_PROJECT'] = 'hw3-mas-tourism-customer-support'
