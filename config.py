import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings
import os
import yaml

class BaseSettingsWrapper(BaseSettings):
    class Config:
        env_file = env_file = "Genai/.env" if os.path.exists("genai/.env") else ".env"
        case_sensitive = True
        extra = "allow"

class MongoConfig(BaseSettingsWrapper):
    MONGODB_URI: str 
    DB_NAME : str = "xerox_stage"
    COLLECTION :str = "Collection-NAme"

class LLMConfig(BaseSettingsWrapper):
    OPENAI_API_KEY : str
    TIKTOKEN_MODEL : str 
    TDGRAPH_EMBEDDING_MODEL : str 
    MODEL: str = "gpt-4o-mini"
    TEMPRATURE : int = 0
    MODEL_0125 : str = "gpt-4o"
    EMBD_MODEL_PATH : str
    
class AppConfig(BaseSettingsWrapper):
    PINECONE_API_KEY: str
    INDEX_NAME: str = "document-analysis"
    MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "models")
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    OLLAMA_API_BASE: str = "http://localhost:11434/api"
    OLLAMA_MODEL: str = "phi3:mini"
    COMPANY_PDF_MAPPING: dict = {
        "enersys": ["EnerSys-2023-10K.pdf", "EnerSys-2017-10K.pdf"],
        "amazon": ["Amazon10k2022.pdf"],
        "apple": ["Apple_10-K-2021.pdf"],
        "nvidia": ["Nvidia.pdf"],
        "tesla": ["Tesla.pdf"],
        "lockheed": ["Lockheed_martin_10k.pdf"],
        "advent": ["Advent_Technologies_2022_10K.pdf"],
        "transdigm": ["TransDigm-2022-10K.pdf"]
    }

mongo_uri =  MongoConfig()
llm_config = LLMConfig()
app_config = AppConfig()
