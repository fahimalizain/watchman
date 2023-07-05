import logging
import os


RTSP_URL_FORMAT = "rtsp://admin:Faaz@123@192.168.1.200:554/cam/realmonitor?channel={channel}&subtype=1"
CAM_CHANNEL_LIST = [
    "side-gate",
    "rear",
    "school",
    "master-bed",
    "sit-out",
    "front-gate",
    "indoor-first-floor",
    "ablution",
]

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.getLogger('cv2').setLevel(logging.WARNING)
# Adjust the level as needed
os.environ["FFREPORT"] = "file=ffmpeg-logs.txt:level=8"


def get_rtsp_url(channel: str):
    if channel not in CAM_CHANNEL_LIST:
        raise ValueError(f"Invalid channel name: {channel}")
    return RTSP_URL_FORMAT.format(channel=CAM_CHANNEL_LIST.index(channel) + 1)


def get_all_rtsp_urls():
    return {channel: get_rtsp_url(channel) for channel in CAM_CHANNEL_LIST}


def get_selective_rtsp_urls():
    return {channel: get_rtsp_url(channel) for channel in [
        "indoor-first-floor",
        # "ablution",
    ]}


def get_mask_filename(channel_name):
    return f"{os.path.dirname(os.path.abspath(__file__))}/masks/mask-{CAM_CHANNEL_LIST.index(channel_name) + 1}-{channel_name}.jpg"
