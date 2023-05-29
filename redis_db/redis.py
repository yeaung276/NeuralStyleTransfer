from typing import Optional
from redis_om import get_redis_connection, HashModel, Field

redis_db = get_redis_connection(host="localhost", port="6379")

class JobRedis(HashModel):
    process_id: str = Field(index=True, primary_key=True)
    status: str
    content_file: Optional[str] = Field(default='')
    style_file: Optional[str] = Field(default='')
    result: Optional[str] = Field(default='')
    
    class Meta:
        database: redis_db
