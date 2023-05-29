from typing import Literal, Optional, List
import uuid
import binascii
from pydantic import BaseModel
from redis_db.redis import JobRedis

class Job(BaseModel):
    process_id: str
    status: Literal['processing', 'success', 'fail']
    content_file: Optional[bytes] = b''
    style_file: Optional[bytes] = b''
    result: Optional[bytes] = b''
    
    @classmethod
    def from_redis(cls, r_job: JobRedis) -> "Job":
        return Job(
            process_id=r_job.process_id,
            status=r_job.status,
            content_file=binascii.unhexlify(r_job.content_file.encode('utf-8')),
            style_file=binascii.unhexlify(r_job.style_file.encode('utf-8')),
            result=binascii.unhexlify(r_job.result.encode('utf-8'))
        )
    
    def to_redis(self) -> JobRedis:
        return JobRedis(
            process_id=self.process_id,
            content_file=binascii.hexlify(self.content_file).decode('utf-8'), 
            style_file=binascii.hexlify(self.style_file).decode('utf-8'), 
            status=self.status,
            result=binascii.hexlify(self.result).decode('utf-8')
        )


class Jobs:
    @classmethod
    def create(cls, content_file: bytes, style_file: bytes) -> Job:
        job = Job(
            process_id=uuid.uuid1().hex[:8],
            content_file=content_file, 
            style_file=style_file, 
            status='processing'
            )
        job.to_redis().save()
        return job
        
    @classmethod
    def update(cls, n_job: Job):
        job = JobRedis.get(n_job.process_id)
        if job is not None:
            job.status = n_job.status
            job.result = binascii.hexlify(n_job.result).decode('utf-8')
            job.save()
    
    @classmethod
    def get_all_jobs(cls) -> List[Job]:
        return [Job.from_redis(JobRedis.get(pk)) for pk in JobRedis.all_pks()]
        
    @classmethod
    def get(cls, job_id: str) -> Job | None:
        redis_model = JobRedis.get(job_id)
        return Job.from_redis(redis_model)