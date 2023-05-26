from io import BytesIO
from PIL import Image
from fastapi import status
from fastapi.responses import Response


class ImageResponse(Response):
    def __init__(self, data: bytes) -> None:
        super().__init__(
            status_code=status.HTTP_200_OK,
            content=data,
            media_type="image/png",
        )