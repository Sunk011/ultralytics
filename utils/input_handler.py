import time
import cv2

from threading import Thread

class InputStreamHandler:
    """
    VideoStreamHandler class for video capture and streaming from various sources.
    Supports camera, file, RTSP, RTMP, and UDP streams.
    Captures frames in a separate thread and puts them in a queue for processing.
    """
    def __init__(self, source_input, frame_queue):
        self.source_input = source_input  # 视频源输入（摄像头id、文件路径、流地址等）
        self.connection_state = 'NO'
        self.source_type = self._determine_source_type()  # 数据源类型
        
        # 设定不同的图像流来源
        self._initialize_capture()
        
        self.frame_queue = frame_queue  # 帧队列
        self.is_running = False  # 状态标签
        self.should_stop = False # 监听键盘中断，等待退出

    def _determine_source_type(self):
        """判断数据源类型"""
        if self.source_input == 0:
            return 'camera'
        elif isinstance(self.source_input, str):
            if self.source_input.startswith('rtsp://'):
                return 'rtsp'
            elif self.source_input.startswith('rtmp://'):
                return 'rtmp'
            else:
                return 'file'
        elif isinstance(self.source_input, int) and self.source_input > 9:
            return 'udp'
        else:
            return 'unknown'

    def _initialize_capture(self):
        """初始化视频捕获"""
        if self.source_type == 'camera':
            self.video_capture = cv2.VideoCapture(0)
            if self.video_capture.isOpened():
                print('== INFO == VideoStreamHandler => cam isOpened... ')
                self.connection_state = 'YES'
            else:
                print('== ERROR == VideoStreamHandler => cam is not Opened... ')
        
        elif self.source_type == 'file':
            self.video_capture = cv2.VideoCapture(self.source_input, cv2.CAP_FFMPEG)
            if self.video_capture.isOpened():
                print('== INFO == VideoStreamHandler => get video file success... ')
                self.connection_state = 'YES'
            else:
                print('== ERROR == VideoStreamHandler => get video file fail... ')
        
        elif self.source_type in ['rtsp', 'rtmp']:
            # RTSP/RTMP 流处理
            self._connect_stream(self.source_input)
        
        elif self.source_type == 'udp':
            # UDP ts video stream
            udp_url = f'udp://0.0.0.0:{self.source_input}'
            self._connect_stream(udp_url)
        
        else:
            print(f'== ERROR == VideoStreamHandler => Unknown source type: {self.source_type}')

    def _connect_stream(self, stream_url):
        """连接网络流"""
        print(f'== INFO == VideoStreamHandler => Connecting to stream: {stream_url}')
        # 等待流连接成功
        while True:
            self.video_capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not self.video_capture.isOpened():
                print(f'== WARNING == VideoStreamHandler => Stream not opened, retrying... ')
                time.sleep(1)
                continue
            else:
                print(f'== INFO == VideoStreamHandler => Stream connected successfully: {stream_url}')
                self.connection_state = 'YES'
                break

    def Capture_Decode(self):
        first_frame_received = True
        try:
            while not(self.should_stop):
                # self.is_running and self.video_capture.isOpened()
                ret, frame = self.video_capture.read()
                if self.should_stop:
                    print('== INFO == VideoStreamHandler => get should_stop, Capture_Decode while break')
                    break
                if not self.is_running:
                    print('== INFO == VideoStreamHandler => is_running False, Capture_Decode while break')
                    break
                # when cam can not read && No KeyboardInterrupt
                if not ret and (not self.should_stop):
                    print('self.should_stop',self.should_stop)
                    print('self.is_running',self.is_running)
                    print("== WARNING == VideoStreamHandler => Can not Read VideoCapture, Waiting")
                    self.connection_state = 'NO'
                    self.reconnect()
                    continue
                elif ret & first_frame_received == True:
                    print("== SUCCESS == VideoStreamHandler => Get VideoCapture, Start ~ ")
                    self.connection_state = 'YES'
                    first_frame_received = False
                else:
                    # cap_los_count = 0
                    pass
                if self.frame_queue.qsize() < 1: # 队列空 压图片
                    self.frame_queue.put(frame)
            print('== INFO == VideoStreamHandler => =^_^= Cam thread All done, See you ~')
        except Exception as e:
            print('== ERROR == VideoStreamHandler => Camera Thread Error, Reason:', e)
    def get_Stream_Status(self):
        if self.connection_state == 'YES':
            return 1
        else:
            return 0
    def run(self):
        self.is_running = True
        print('== INFO == VideoStreamHandler => ---------- Camera  Thread Start ---------- ')
        self.capture_thread = Thread(target=self.Capture_Decode)
        self.capture_thread.start()
    def stop(self):
        self.is_running = False
        self.connection_state = 'NO'
        self.should_stop = True
        print('== INFO == VideoStreamHandler => <stop> Camera Thread Stoped')
        # cv2.destroyAllWindows()
        self.video_capture.release()
    def reconnect(self):
        """重连数据源"""
        self.is_running = False
        print(f'== WARNING == VideoStreamHandler => Reconnecting to {self.source_type} source')
        self.video_capture.release()
        self.is_running = True
        reconnect_count = 0
        
        while True:
            try:
                reconnect_count += 1
                print(f'== WARNING == VideoStreamHandler => reconnect attempt {reconnect_count}')
                
                if self.source_type == 'udp':
                    stream_url = f'udp://0.0.0.0:{self.source_input}'
                elif self.source_type in ['rtsp', 'rtmp']:
                    stream_url = self.source_input
                elif self.source_type == 'file':
                    stream_url = self.source_input
                elif self.source_type == 'camera':
                    stream_url = 0
                else:
                    print(f'== ERROR == VideoStreamHandler => Cannot reconnect to {self.source_type}')
                    break
                
                self.video_capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                ret, _ = self.video_capture.read()
                
                if not ret:
                    self.connection_state = 'NO'
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    print(f'== SUCCESS == VideoStreamHandler => Reconnected to {self.source_type} source')
                    self.connection_state = 'YES'
                    break
                    
            except KeyboardInterrupt:
                print('== ERROR == VideoStreamHandler => Reconnect interrupted by user')
                break
            except Exception as e:
                print(f'== ERROR == VideoStreamHandler => Reconnect error: {e}')
                time.sleep(2)
                continue

if __name__ == "__main__":
    print("=====Start =====")
    from queue import Queue
    frame_queue = Queue()
    
    camera = InputStreamHandler('rtsp://172.17.0.1:8554/live', frame_queue)
    camera.run()
    
    print(' ---------- Process Thread Start ---------- ')
    frame_count = 0
    try:
        # Simple test: get image resolution to verify stream is working
        while True:
            image = frame_queue.get()
            frame_count += 1
            height, width, channels = image.shape
            print(f"Frame {frame_count}: Image resolution: {width}x{height}, Channels: {channels}")
                
    except KeyboardInterrupt:
        print(" =CTRL= KeyboardInterrupt! done ============ ")
        camera.stop()
    except Exception as e:
        print(" =ERROR= Exception! done ======Reason:", e)
        camera.stop()
    print(' =^_^= All done, See you ~')
