from env import Actor
import asyncio

actor = Actor()
result = asyncio.run(actor.local_evaluate())
print(result)