from typing import Annotated
import uvicorn
from fastapi import FastAPI, UploadFile, File
from response.ImageResponse import ImageResponse
from PIL import Image

app = FastAPI()

@app.get('/')
def welcome():
    return "go to {url}/docs for documentation"

@app.post('/get_nst_image')
async def generate_image(
    content_image: Annotated[UploadFile, File], 
    style_image: Annotated[UploadFile, File]
    ):

    return ImageResponse(Image.open('./output.png'))


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)