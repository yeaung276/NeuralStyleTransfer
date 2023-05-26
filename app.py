from typing import Annotated
from contextlib import asynccontextmanager
from concurrent.futures.process import ProcessPoolExecutor
# from io import BytesIO
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi import File, Request, BackgroundTasks
# from response.image_response import ImageResponse
# from PIL import Image
# from core.NST import NST
# from core.preprocessor import Preprocessor
from background.background_task import generate_image, NSTRequest
from background.nst_handler import setup_nst
from db.jobs import Jobs, Job


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_nst()
    app.state.executor = ProcessPoolExecutor()
    app.state.executor.submit(setup_nst)
    yield
    app.state.executor.shutdown()

app = FastAPI(lifespan=lifespan)

# content_image = Preprocessor.transform('images/louvre_small.jpg')

# style_image = Preprocessor.transform('images/monet.jpg')

# NST.initialize('imagenet-vgg-verydeep-19.mat')

# NST.set_cost_weights(alpha=0.2, beta=0.8)

# g_img, _ = NST.generate(content_image,style_image,no_iter=200, display=True)

# processed_img = Preprocessor.post_process(g_img)

@app.get('/')
def welcome():
    return "go to {url}/docs for documentation"

@app.post('/get_nst_image')
async def generate_nst_image(
    request: Request,
    content_image: Annotated[UploadFile, File], 
    style_image: Annotated[UploadFile, File],
    background_tasks: BackgroundTasks
    ):
    c_image = await content_image.read()
    s_image = await style_image.read()
    job = Job(process_id='only_job_id', status='processing')
    Jobs.add(job)
    background_tasks.add_task(
        generate_image, 
        request.app.state.executor, 
        NSTRequest(content=c_image, style=s_image, process_id=job.process_id)
    )
    return job
    # bytes_image = Image.open(BytesIO(image))
    # processed = Preprocessor.transform(bytes_image)
    # process_img = Image.fromarray(processed[0,:,:,:])
    # return ImageResponse(process_img)
    
@app.get('/get_job')
async def get_job():
    return Jobs.get('only_job_id')
    
@app.post('/test')
async def test(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_image, request.app.state.executor)
    return 'success'

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)