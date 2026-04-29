import uuid
from dataclasses import dataclass, field
from typing import Optional

_jobs: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    status: str = "pending"   # pending | running | done | error
    progress: int = 0          # 0-100
    message: str = ""
    error: Optional[str] = None
    total_rows: int = 0
    processed_rows: int = 0


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    for k, v in kwargs.items():
        setattr(job, k, v)
    if job.total_rows > 0:
        job.progress = int((job.processed_rows / job.total_rows) * 100)
