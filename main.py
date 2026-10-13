import asyncio
from dotenv import load_dotenv
from asmo import Pipeline
from asmo.logging_service import setup_logging, logger

load_dotenv()
setup_logging()


async def main() -> None:
    logger.info("Pipeline starting")
    result = await Pipeline().run()
    logger.info(f"Pipeline finished — {result.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
