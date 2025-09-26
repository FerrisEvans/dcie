from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app import routers
from core import conf
from core.conn import master_async_engine, get_redis
from core.exceptions import register_exception
from core.logger import setup_logging, log

async def run_app():
    await setup_logging()
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8100,
        log_config=None
    )
    server = uvicorn.Server(config)
    await server.serve()

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

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

# 注册全局异常
register_exception(app=app)
# 跨域
if conf.http.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=conf.http.allow_origins,
        allow_credentials=conf.http.allow_credentials,
        allow_methods=conf.http.allow_methods,
        allow_headers=conf.http.allow_headers,
    )
# api
for r in routers:
    app.include_router(r)

if __name__ == '__main__':
    import asyncio

    asyncio.run(run_app())
