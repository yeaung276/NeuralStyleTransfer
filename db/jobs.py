from typing import Literal, Optional, List
import uuid
from pydantic import BaseModel

class Job(BaseModel):
    process_id: str
    status: Literal['processing', 'success', 'fail']
    content_file: Optional[bytes]
    style_file: Optional[bytes]
    result: Optional[bytes]

class Jobs:
    requests = {}

    @classmethod
    def create(cls, content_file: bytes, style_file: bytes) -> Job:
        job = Job(
            process_id=uuid.uuid1().hex[:8], 
            content_file=content_file, 
            style_file=style_file,
            status='processing'
        )
        cls.requests[job.process_id] = job
        return job
        
    @classmethod
    def update(cls, n_job: Job):
        job = cls.requests.get(n_job.process_id, None)
        if job is not None:
            job.status = n_job.status
            job.result = n_job.result
    
    @classmethod
    def get_all_jobs(cls) -> List[Job]:
        return [value for value in cls.requests.values()]
        
    @classmethod
    def get(cls, job_id: Job) -> Job | None:
        return cls.requests.get(job_id, None)