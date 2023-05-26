from typing import Literal, Optional
from pydantic import BaseModel

class Job(BaseModel):
    process_id: str
    user_id: Optional[str]
    status: Literal['processing', 'success', 'fail']
    result: Optional[bytes]

class Jobs:
    requests = {}

    @classmethod
    def add(cls, job: Job):
        cls.requests[job.process_id] = job
        
    @classmethod
    def update(cls, n_job: Job):
        job = cls.requests.get(n_job.process_id, None)
        if job is not None:
            job.status = n_job.status
            job.result = n_job.result
        
    @classmethod
    def get(cls, job_id: Job) -> Job | None:
        return cls.requests.get(job_id, None)