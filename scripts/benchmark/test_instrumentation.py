import asyncio
import logging
import sys
from app.remote.client import RemoteInferenceClient

# Configure logging to stdout to see metrics
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


async def test_metrics():
    client = RemoteInferenceClient()
    messages = [{"role": "user", "content": "Hello"}]

    print("Testing generate...")
    try:
        # Note: this will fail if the remote URL is not reachable,
        # but it will show if the decorator runs before/during the call.
        await client.generate(messages)
    except Exception as e:
        print(f"Generate failed as expected/or not: {e}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_metrics())
