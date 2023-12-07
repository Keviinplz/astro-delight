import asyncio

from telegram import Bot

from .base import RayNotifier


class TelegramNotifier(RayNotifier):
    def __init__(self, token: str, chat_id: int):
        self._bot = Bot(token)
        self._chat_id = chat_id

    def notify(self, message: str) -> None:
        async def notify(bot: Bot, message: str, chat_id: int):
            async with bot:
                await bot.send_message(text=message, parse_mode="MarkDown", chat_id=chat_id) # type: ignore

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'RuntimeError: There is no current event loop...'
            loop = None

        if loop and loop.is_running():
            print(
                "Async event loop already running. Adding coroutine to the event loop."
            )
            tsk = loop.create_task(notify(self._bot, message, self._chat_id))
            # ^-- https://docs.python.org/3/library/asyncio-task.html#task-object
            # Optionally, a callback function can be executed when the coroutine completes
            tsk.add_done_callback(lambda t: print("Message sent"))
        else:
            print("Starting new event loop")
            asyncio.run(notify(self._bot, message, self._chat_id))
