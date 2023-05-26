from pydantic import BaseModel

class JobResponse(BaseModel):
    process_id: str
    status: str
