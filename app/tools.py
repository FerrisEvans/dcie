from fastapi import APIRouter

from core.logger import log
router = APIRouter(prefix="/v1/tools")

log.success("tools router initialized")

import engine.reader