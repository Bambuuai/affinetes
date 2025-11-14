from env import Actor
import asyncio

actor = Actor()
result = asyncio.run(actor.local_evaluate(task_type='sat', kwargs = {'k':2, 'n':3}))
print(result)