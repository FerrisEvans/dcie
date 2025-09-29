from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from starlette import status

from common.logger import log
from engine.simple_file import single_file_handler

router = APIRouter(prefix="/upload")

# import engine.reader

@router.post("/i/", status_code=status.HTTP_200_OK)
async def create_upload_file(files: list[UploadFile]):
    for file in files:
        single_file_handler(file)
    return JSONResponse({"filenames": [file.filename for file in files]})


