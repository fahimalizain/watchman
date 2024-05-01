from multiprocessing import Process, Queue
from dotenv import load_dotenv

from .mog_cuda import MOGCuda
from .yolo import YoloUnit
from .telegram import Telegram
from .multiprocessing import AbstractMultiprocessing

from .config import get_selective_rtsp_urls, get_all_rtsp_urls

if __name__ == "__main__":
    load_dotenv()
    yolo = YoloUnit()
    telegram = Telegram()
    yolo.connect(telegram)

    runners: list[AbstractMultiprocessing] = [yolo, telegram]
    for channel_name, rtsp_url in get_all_rtsp_urls().items():
        mog_cuda = MOGCuda(channel_name=channel_name, rtsp_url=rtsp_url)
        mog_cuda.connect(yolo)

        runners.append(mog_cuda)

    for runner in runners:
        runner.start_process()

    for runner in runners:
        runner.join_process()
