# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from datetime import datetime

import cv2


class OutputHandler:
    def __init__(
        self,
        enable_video_save=False,
        enable_rtsp_push=False,
        # 视频保存相关参数
        video_save_path="output/videos",
        video_save_format="mp4",
        video_save_fps=30,
        # RTSP推流相关参数
        rtsp_fps=30,
        rtsp_width=1920,
        rtsp_height=1080,
        rtsp_bitrate=5000,
        rtsp_speed_preset="medium",
        rtsp_output_url="rtsp://localhost:8554/stream",
    ):
        self.enable_video_save = enable_video_save
        self.enable_rtsp_push = enable_rtsp_push
        self.video_writer = None
        self.rtsp_writer = None
        self.frame_count = 0

        # 视频保存参数
        self.video_save_path = video_save_path
        self.video_save_format = video_save_format
        self.video_save_fps = video_save_fps

        # RTSP推流参数
        self.rtsp_fps = rtsp_fps
        self.rtsp_width = rtsp_width
        self.rtsp_height = rtsp_height
        self.rtsp_bitrate = rtsp_bitrate
        self.rtsp_speed_preset = rtsp_speed_preset
        self.rtsp_output_url = rtsp_output_url

    def initialize_writers(self, width, height):
        """初始化视频写入器."""
        # 初始化视频保存写入器
        if self.enable_video_save:
            self._initialize_video_writer(width, height)

        # 初始化RTSP推流写入器
        if self.enable_rtsp_push:
            self._initialize_rtsp_writer()

    def _initialize_video_writer(self, width, height):
        """初始化视频文件保存写入器."""
        os.makedirs(self.video_save_path, exist_ok=True)
        fourcc = self._get_video_codec(self.video_save_format)
        output_path = os.path.join(self.video_save_path, f"output_tracking.{self.video_save_format}")
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.video_save_fps, (width, height))
        print(f"Video writer initialized: {output_path} ({width}x{height})")

    def _initialize_rtsp_writer(self):
        """初始化RTSP推流写入器."""
        rtsp_pipeline_1 = (
            "appsrc ! videoconvert"
            + " ! video/x-raw,format=I420"
            + f" ! x264enc speed-preset={self.rtsp_speed_preset} bitrate={self.rtsp_bitrate} key-int-max="
            + str(self.rtsp_fps * 2)
            + " ! video/x-h264,profile=baseline"
            + f" ! rtspclientsink location={self.rtsp_output_url}"
        )

        # rtsp_pipeline_2 = ('appsrc ! videoconvert' +
        #                 ' ! video/x-raw,format=NV12' +  # GPU 编码器通常需要 NV12 格式
        #                 f' ! nvh264enc preset={self.rtsp_speed_preset} bitrate={self.rtsp_bitrate} gop-size=' +
        #                 str(self.rtsp_fps * 2) +
        #                 ' ! h264parse ! video/x-h264,profile=baseline' +
        #                 f' ! rtspclientsink location={self.rtsp_output_url}')

        self.rtsp_writer = cv2.VideoWriter(
            rtsp_pipeline_1, cv2.CAP_GSTREAMER, 0, self.rtsp_fps, (self.rtsp_width, self.rtsp_height), True
        )

        if not self.rtsp_writer.isOpened():
            print("Warning: RTSP writer failed to open")
            self.rtsp_writer = None
        else:
            print(f"RTSP writer initialized successfully: {self.rtsp_output_url}")

    def _get_video_codec(self, format_type):
        """获取视频编码器."""
        codecs = {
            "avi": cv2.VideoWriter_fourcc(*"XVID"),
            "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
            "mkv": cv2.VideoWriter_fourcc(*"XVID"),
        }
        return codecs.get(format_type, cv2.VideoWriter_fourcc(*"mp4v"))

    def write_frame(self, frame):
        """写入帧到各个输出."""
        self.frame_count += 1

        # 如果还没有初始化写入器，使用第一帧的尺寸进行初始化
        if self.video_writer is None and self.rtsp_writer is None:
            height, width = frame.shape[:2]
            self.initialize_writers(width, height)

        # 写入视频文件
        if self.video_writer is not None:
            self.video_writer.write(frame)

        # 写入RTSP推流
        if self.rtsp_writer is not None:
            # 调整帧尺寸以匹配RTSP推流要求
            rtsp_frame = cv2.resize(frame, (self.rtsp_width, self.rtsp_height))
            self.rtsp_writer.write(rtsp_frame)
            print(f"{datetime.now()} - Frame {self.frame_count} written to RTSP stream")

    def release(self):
        """释放所有资源."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Video writer released")

        if self.rtsp_writer is not None:
            self.rtsp_writer.release()
            self.rtsp_writer = None
            print("RTSP writer released")

        print(f"Total frames processed: {self.frame_count}")
