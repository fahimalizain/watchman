import multiprocessing
import abc
import asyncio
import logging


class AbstractMultiprocessing(metaclass=abc.ABCMeta):
    __MULTIPROCESSING_ARGS__ = ['input_queue', 'output_queue']

    def __init__(self, input_queue=None):
        self.input_queue = input_queue or multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None

    @abc.abstractmethod
    def run(self, data=None):
        pass

    def connect(self, next_process: "AbstractMultiprocessing"):
        self.output_queue = next_process.input_queue
        return next_process

    def start_process(self):
        self.process = multiprocessing.Process(
            target=self._run_on_process, kwargs={k: v for k, v in self.__dict__.items() if k in self.__MULTIPROCESSING_ARGS__})
        self.process.start()

    def _run_on_process(self, **data: dict):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.__dict__.update(data)
        self.run()

    def join_process(self):
        if self.process is not None:
            self.process.join()

    def get_from_queue(self):
        while not self.input_queue.empty():
            yield self.input_queue.get()

    def put_in_queue(self, item):
        self.output_queue.put(item)

    def log_info(self, msg, *args, **kwargs):
        return self.logger.info(msg, *args, **kwargs)

    def log_error(self, msg, *args, **kwargs):
        return self.logger.error(msg, *args, **kwargs)


class SyncRunner(AbstractMultiprocessing):
    def run(self, data=None):
        data = data or list(self.get_from_queue())
        for item in data:
            self.put_in_queue(item * 2)


class AsyncRunner(AbstractMultiprocessing):
    def run(self, data=None):
        loop = asyncio.get_event_loop()
        data = data or list(super().get_from_queue())
        loop.run_until_complete(self.async_run(data))

    async def async_run(self, data):
        for item in data:
            self.put_in_queue(item * 2)
            await asyncio.sleep(0.1)

    async def get_from_queue(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.input_queue.get)
