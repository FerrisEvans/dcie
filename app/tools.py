from fastapi import APIRouter, UploadFile, File

from core.logger import log
router = APIRouter(prefix="/tools")

log.success("tools router initialized")

# import engine.reader

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "contents": contents}