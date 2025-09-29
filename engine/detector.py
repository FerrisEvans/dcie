import json
from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from common.conn import get_db
from core import ac_detect
from core.models import Ac

def ac_task(text: str, task_id: str, filename: str, db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    results = ac_detect(text)
    ac = Ac(task_id=task_id, filename=filename, res=json.dumps(results, ensure_ascii=False))
    ac.save(db)
    return results


