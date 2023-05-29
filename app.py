import uvicorn
from typing import Annotated, List, Literal
from contextlib import asynccontextmanager
from concurrent.futures.process import ProcessPoolExecutor
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi import File, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from background.background_task import generate_image, NSTRequest
from background.nst_handler import setup_nst
from response.job_response import JobResponse
from response.image_response import ImageResponse
from ws.connection_manager import ConnectionManager
from redis_db.jobs import Jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ProcessPoolExecutor()
    app.state.executor.submit(setup_nst)
    yield
    app.state.executor.shutdown()

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

manager = ConnectionManager()

@app.get('/')
def welcome():
    return RedirectResponse('static/index.html')

@app.post('/get_nst_image')
async def generate_nst_image(
    request: Request,
    content_image: Annotated[UploadFile, File], 
    style_image: Annotated[UploadFile, File],
    background_tasks: BackgroundTasks
    ) -> JobResponse:
    c_image = await content_image.read()
    s_image = await style_image.read()

    job = Jobs.create(c_image, s_image)

    background_tasks.add_task(
        generate_image, 
        request.app.state.executor, 
        manager,
        NSTRequest(content=c_image, style=s_image, process_id=job.process_id)
    )
    return job
    
@app.get('/get_jobs')
async def get_jobs() -> List[JobResponse]:
    return Jobs.get_all_jobs()

@app.get('/get_image/${type}/{process_id}')
async def get_image(type: Literal['result', 'content', 'style'], process_id: str) -> ImageResponse:
    job = Jobs.get(process_id)
    if type == 'result':
        return ImageResponse(job.result)
    elif type == 'content':
        return ImageResponse(job.content_file)
    else:
        return ImageResponse(job.style_file)
    
@app.websocket('/ws')
async def websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)