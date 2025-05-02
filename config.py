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
    

mongo_uri =  MongoConfig()
llm_config = LLMConfig()
