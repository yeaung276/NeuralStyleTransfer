import os
from typing import Optional
from redis_om import get_redis_connection, HashModel, Field

REDIS_DATA_URL = os.environ.get('REDIS_OM_URL')

class JobRedis(HashModel):
    process_id: str = Field(index=True, primary_key=True)
    status: str
    content_file: Optional[str] = Field(default='')
    style_file: Optional[str] = Field(default='')
    result: Optional[str] = Field(default='')
    
    class Meta:
        database: get_redis_connection(url=REDIS_DATA_URL, decode_responses=True)
