import asyncio
import truststore
from dotenv import load_dotenv
from asmo import Pipeline
from asmo.logging_service import setup_logging, logger

# Use the OS trust store for TLS so requests to CommerceTools succeed behind a
# corporate proxy that injects its own root CA. Must run before any SSL context
# (i.e. any aiohttp session) is created.
truststore.inject_into_ssl()

load_dotenv()
setup_logging()


async def main() -> None:
    logger.info("Pipeline starting")
    result = await Pipeline().run()
    logger.info(f"Pipeline finished — {result.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
