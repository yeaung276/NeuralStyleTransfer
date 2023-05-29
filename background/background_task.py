import asyncio
import logging
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from concurrent.futures.process import ProcessPoolExecutor
from redis_db.jobs import Jobs, Job
from background.nst_handler import nst_handler
from ws.connection_manager import ConnectionManager

class NSTRequest(BaseModel):
    process_id: str
    content: bytes
    style: bytes


async def generate_image(
    executor: ProcessPoolExecutor,
    websocket: ConnectionManager, 
    request: NSTRequest,
    ):
    result_bin = BytesIO()
    c_img = Image.open(BytesIO(request.content))
    s_img = Image.open(BytesIO(request.style))
    loop = asyncio.get_event_loop()
    await websocket.send_message({
        'process_id': request.process_id,
        'status': 'started',
        'message': f'process_id: {request.process_id} started'
    })
    try:
        gen_img: Image = await loop.run_in_executor(executor, nst_handler, c_img, s_img)
        gen_img.save(result_bin, format="png")
        Jobs.update(Job(
            process_id=request.process_id, 
            result=result_bin.getvalue(), 
            status='success'
        ))
        await websocket.send_message({
            'process_id': request.process_id,
            'status': 'success',
            'message': f'process_id: {request.process_id} successfully finished'
            })
    except Exception as e:
        logging.error(e)
        Jobs.update(Job(
            process_id=request.process_id, 
            result=result_bin.getvalue(), 
            status='fail'
        ))
        await websocket.send_message({
            'process_id': request.process_id,
            'status': 'fail',
            'message': f'process_id: {request.process_id}: {e}'
            })