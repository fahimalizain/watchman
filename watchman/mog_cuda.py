import cv2
import multiprocessing
import traceback
from collections import deque

from .multiprocessing import SyncRunner
from .config import get_mask_filename


class MOGCuda(SyncRunner):

    def __init__(self, channel_name: str, rtsp_url: str, input_queue=None):
        super().__init__(input_queue)
        self.channel_name = channel_name
        self.rtsp_url = rtsp_url

        self.__MULTIPROCESSING_ARGS__ = [
            *self.__MULTIPROCESSING_ARGS__,
            'channel_name', 'rtsp_url',
        ]

    def setup(self):

        # Setup
        # SubType 1 recommended is 1000 minArea
        self.min_area = 2000
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cuda_stream = cv2.cuda.Stream()

        self.motion_frames = []
        self.has_motion = False
        self.frame_cache = deque(maxlen=20)
        self.last_motion_counter = 0
        self.frames_with_motion_counter = 0

        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            srcType=cv2.CV_8UC3, dstType=cv2.CV_8UC3, ksize=(5, 5), sigma1=1)

        # MOG2 flashes a lot
        # bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
        # history=500, varThreshold=512, detectShadows=False)
        self.bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG(
            history=500)

        # Dilator
        # Create a structuring element for dilation
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        d_se = cv2.cuda_GpuMat()
        d_se.upload(se)

        # Load local file 'timemask.png' as a grayscale image
        channel_mask = cv2.imread(get_mask_filename(
            self.channel_name), cv2.IMREAD_GRAYSCALE)
        self.channel_mask = cv2.cuda_GpuMat()
        self.channel_mask.upload(channel_mask)

        # Create the dilate filter
        self.dilate_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8U, se)

    def run(self):
        self.setup()
        self.log_info(
            f"Running RTSPMotionDetector for {self.channel_name}")

        while True:
            try:
                while self.cap.isOpened():
                    # Read a frame from the stream
                    ret, frame = self.cap.read()
                    # If the frame is None, break the loop
                    if frame is None:
                        break

                    _has_motion = self.check_has_motion(
                        frame, draw_contours=False)
                    self.handle_motion(frame, _has_motion)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            except Exception as e:
                # Log traceback
                tb = "\n".join(traceback.format_tb(e.__traceback__))
                self.log_error(
                    f"MOG CUDA Error {self.channel_name} {str(e)}\n{tb}")

            # Release the stream and close all windows
            self.cap.release()
            cv2.destroyAllWindows()

    def check_has_motion(self, frame, draw_contours=False, show_debug_windows=False):
        contours = self._get_contours(
            frame, show_debug_windows=show_debug_windows)

        # Loop over the contours
        has_motion = False
        for contour in contours:
            # If the contour is too small, ignore it
            if cv2.contourArea(contour) < self.min_area:
                continue

            has_motion = True

            if not draw_contours:
                break

            # Compute the bounding box for the contour and draw it on the frame
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if show_debug_windows:
            cv2.imshow("Stream {} Actual".format(self.channel_name), frame)

            bg = self.bg_subtractor.getBackgroundImage(
                stream=cv2.cuda_Stream.Null()).download()
            cv2.imshow("Stream {} Background".format(self.channel_name), bg)
            if has_motion:
                cv2.imshow("Stream {} Motion".format(self.channel_name), frame)

        return has_motion

    def _get_contours(self, frame, show_debug_windows=False):
        # Upload the frame to the GPU
        d_frame = cv2.cuda_GpuMat()
        d_frame.upload(frame)

        # Resize the frame
        # d_frame = cv2.cuda.resize(d_frame, 500)

        # Apply the Gaussian blur
        # gaussian_filter.apply(d_frame, d_frame)

        # Apply the MOG2 background subtractor
        d_fg_mask = self.bg_subtractor.apply(
            d_frame, learningRate=-1, stream=self.cuda_stream)

        # Apply time mask
        cv2.cuda.bitwise_and(d_fg_mask, self.channel_mask,
                             d_fg_mask, stream=self.cuda_stream)

        # Apply a threshold to the foreground mask
        d_thresh = cv2.cuda.threshold(
            d_fg_mask, 30, 255, cv2.THRESH_BINARY, stream=self.cuda_stream)[1]

        # Dilate the thresholded image to fill in holes
        self.dilate_filter.apply(d_thresh, d_thresh)

        # Download the dilated image to host memory
        thresh = d_thresh.download()

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if show_debug_windows:
            cv2.imshow("Stream {} Mask".format(self.channel_name), thresh)

        return contours

    def handle_motion(self, frame, _has_motion=False):
        if self.has_motion:
            self.motion_frames.append(frame)
        else:
            self.frame_cache.append(frame)

        # A fresh motion has started
        if _has_motion and not self.has_motion:
            self.motion_frames = list(self.frame_cache)
            self.frame_cache.clear()
            self.has_motion = True

        # We record 50 frames after motion stops
        if _has_motion:
            self.last_motion_counter = 0
            self.frames_with_motion_counter += 1
        else:
            self.last_motion_counter += 1
            if self.last_motion_counter > 50:
                self.has_motion = False

        # 6 Second max per Animation
        if len(self.motion_frames) > 6 * 25:
            self.has_motion = False

        # Send Motion Frames to GIF Queue
        if not self.has_motion and len(self.motion_frames) > 0:
            # Make sure we have enough frames with Motion to filter out false positives
            # We do this by making sure atleast 10% of the frames have motion
            enough_motion = self.frames_with_motion_counter / \
                len(self.motion_frames) > 0.33

            if enough_motion:
                self.put_in_queue(
                    dict(channel_name=self.channel_name, frames=self.motion_frames))
                self.log_info(
                    f"Generated CVFrames from {self.channel_name} with {len(self.motion_frames)} frames")

            # Reset
            self.motion_frames = []
