import asyncio
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from concurrent.futures.process import ProcessPoolExecutor
from db.jobs import Jobs, Job
from background.nst_handler import nst_handler
    
class NSTRequest(BaseModel):
    process_id: str
    content: bytes
    style: bytes


async def generate_image(
    executor: ProcessPoolExecutor, 
    request: NSTRequest,
    ):
    c_img = Image.open(BytesIO(request.content))
    s_img = Image.open(BytesIO(request.style))
    loop = asyncio.get_event_loop()
    try:
        gen_img: Image = await loop.run_in_executor(executor, nst_handler, c_img, s_img)
        img_bin = BytesIO()
        gen_img.save(img_bin, format="png")
        Jobs.update(Job(
            process_id=request.process_id, 
            result=img_bin.getvalue(), 
            status='success'
        ))
    except Exception as e:
        Jobs.update(Job(
            process_id=request.process_id, 
            result=None, 
            status='fail'
        ))