import uvicorn
from fastapi import FastAPI

from core.logger import setup_logging, log

app = FastAPI()


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

@app.get("/hello")
async def hello():
    log.trace("Hello")
    log.debug("Hello")
    log.info("Hello")
    log.success("Hello")
    log.warning("Hello")
    log.error("Hello")
    log.critical("Hello")
    return {"hello": "world"}

if __name__ == '__main__':
    import asyncio

    asyncio.run(run_app())
