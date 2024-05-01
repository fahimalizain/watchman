import os
import cv2
import traceback
from typing import List
from telegram.ext import ApplicationBuilder, AIORateLimiter


from .multiprocessing import AsyncRunner
from .utils.SuppressStderr import SuppressStderr


class Telegram(AsyncRunner):

    def setup(self):
        TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

        self.app = ApplicationBuilder().token(TOKEN).rate_limiter(
            AIORateLimiter(overall_max_rate=20, overall_time_period=60, group_max_rate=18, group_time_period=60, max_retries=5)).build()

    async def async_run(self, data):
        self.setup()
        while True:
            try:
                data = await self.get_from_queue()
                if not data:
                    continue

                channel_name = data.get('channel_name')
                labels = data.get('labels')
                frames = data.get('frames')

                with SuppressStderr():
                    mp4 = await self.make_mp4(channel_name, frames)

                await self.send_via_telegram(channel_name, mp4)

            except Exception as e:
                # Log traceback
                tb = "\n".join(traceback.format_tb(e.__traceback__))
                self.log_error(
                    f"Telelgram Error {channel_name} {str(e)}\n{tb}")

    async def send_via_telegram(self, channel_name: str, mp4: bytes):
        TELEGRAM_CHAT_ID = int(os.environ.get('TELEGRAM_CHAT_ID'))

        await self.app.bot.send_animation(
            TELEGRAM_CHAT_ID, mp4, filename=f'{channel_name}.mp4')

    async def make_mp4(self, channel_name: str, cv_frames: List):
        import numpy as np
        import tempfile
        import io

        import os
        # Adjust the level as needed
        os.environ["FFREPORT"] = "file=ffmpeg-logs.txt:level=8"

        if len(cv_frames) == 0:
            self.log_error(f"make_mp4: No frames to process")
            return None
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

        # Get W and H from first frame
        max_dim = 350
        frame_height, frame_width, _ = cv_frames[0].shape
        scale_factor = max_dim / max(frame_width, frame_height)
        new_width, new_height = int(
            frame_width * scale_factor), int(frame_height * scale_factor)

        # Define the codec using VideoWriter_fourcc and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        out = cv2.VideoWriter(temp.name, fourcc, 25, (new_width, new_height))

        for frame in cv_frames:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)

        # Release everything after writing
        out.release()

        # Open the temporary file and write to BytesIO
        with open(temp.name, 'rb') as f:
            bytes_io = io.BytesIO(f.read())

        # Don't forget to remove the temporary file
        temp.close()

        # return None
        return bytes_io
