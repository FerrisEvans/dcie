from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app import tools_router
from core.conn import master_async_engine, get_redis
from core.logger import setup_logging, log

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # app.postgres_pool = asyncpg.create_pool(
        #     dsn=conf.master_db_url.unicode_string(),
        #     min_size=conf.database.master.min_size,
        #     max_size=conf.database.master.max_size,
        # )
        # log.success(f"Postgres pool created, idle size: {app.postgres_pool.get_idle_size()}")
        app.redis = await get_redis()
        yield
    finally:
        # await app.postgres_pool.close()
        await master_async_engine.dispose()
        log.success("Postgres pool closed")
        app.redis.close()
        log.success("Redis closed")

app = FastAPI(
    title="dcie",
    description="document compliance identification engine",
    version="0.1.0",
    docs_url="/docs",
    contact={
        "name": "Ferris",
        "email": "ferris.ai@icloud.com",
    },
    lifespan=lifespan,
)


async def run_app():
    await setup_logging()
    app.include_router(tools_router)
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8100,
        log_config=None
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    import asyncio

    asyncio.run(run_app())
