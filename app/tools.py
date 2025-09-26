from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from starlette import status

from core.logger import log
router = APIRouter(prefix="/tools")

log.success("tools router initialized")

# import engine.reader

@router.post("/uploadfile/", status_code=status.HTTP_200_OK)
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return JSONResponse({"filename": file.filename, "contents": contents})


@router.get("/hello")
async def hello():
    log.info("hello")
    return JSONResponse({"hello": "world"})