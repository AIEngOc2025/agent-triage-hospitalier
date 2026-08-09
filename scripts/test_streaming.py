import asyncio

import httpx


async def test():
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://localhost:8080/chat",
            json={
                "history": [{"role": "user", "content": "J'ai mal au ventre"}],
                "stream": True,
            },
            timeout=10,
        )
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")


if __name__ == "__main__":
    asyncio.run(test())
