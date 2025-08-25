import uvicorn
from fastapi import FastAPI

from app.core.logger import setup_logging, logger

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
    logger.trace("Hello")
    logger.debug("Hello")
    logger.info("Hello")
    logger.success("Hello")
    logger.warning("Hello")
    logger.error("Hello")
    logger.critical("Hello")
    return {"hello": "world"}

if __name__ == '__main__':
    import asyncio

    asyncio.run(run_app())
