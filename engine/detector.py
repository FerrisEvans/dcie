import json
from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from common import conf
from common.conn import get_db
from core import ac_detect, classifier
from core.models import Ac, Bert


def ac_task(text: str, task_id: str, filename: str, db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    results = ac_detect(text)
    ac = Ac(task_id=task_id, filename=filename, res=json.dumps(results, ensure_ascii=False))
    ac.save(db)
    return results

def bert_task(text: str, task_id: str, filename: str, db: AsyncSession = Depends(get_db)):
    problem_zones = classifier.trace(text)
    results = []
    for zone in problem_zones:
        if zone['label'] not in conf.bert.exclude_labels:
            bert = Bert(
                task_id=task_id,
                filename=filename,
                label=zone['label'],
                score=zone['score'],
                start_pos=zone['start_pos'],
                end_pos=zone['end_pos'],
                context=zone['context'][:100],
            )
            results.append(bert)
    db.flush()
    db.add_all(results)
    # db.commit() todo 需要判断是否需要在这一行 commit
    return results


