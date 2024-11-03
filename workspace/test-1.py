import cv2
import numpy as np
import os
import json
import time
import logging
import traceback
from picamera2 import Picamera2
from libcamera import controls
from smbus2 import SMBus
from collections import deque

def set_i2c_channel(channel):
    I2C_MUX_ADDRESS = 0x24
    CONTROL_REGISTER = 0x24
    channels = {
        0: 0x02,
        1: 0x12,
        2: 0x22,
        3: 0x32
    }
    
    if channel not in channels:
        logging.error(f"유효하지 않은 채널입니다: {channel}")
        return
    try:
        with SMBus(10) as bus:
            channel_value = channels[channel]
            bus.write_byte_data(I2C_MUX_ADDRESS, CONTROL_REGISTER, channel_value)
            logging.info(f"I2C 채널 {channel} 설정 성공.")
    except Exception as e:
        logging.error(f"I2C 채널 {channel} 설정에 실패했습니다: {e}")

def load_config(json_path):
    try:
        with open(json_path, 'r') as file:
            settings = json.load(file)
        return settings
    except Exception as e:
        print(f"카메라 설정을 로드하는 데 실패했습니다: {e}")
        return None
    
def setup_folders(base_path):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join(base_path, timestamp)
        log_folder_path = os.path.join(folder_path, "log")
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(log_folder_path, exist_ok=True)

        log_level = logging.DEBUG
        log_file_path = os.path.join(log_folder_path, "log.txt")

        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file_path),
                                logging.StreamHandler()
                            ])
        
        logging.info(f"폴더 경로: {folder_path}")
        logging.info(f"로그 파일 경로: {log_file_path}")
        logging.info("폴더 및 로깅 설정 완료.")
        return folder_path
    
    except Exception as e:
        print(f"폴더를 설정하고 로깅을 구성하는 데 실패했습니다: {e}")
        return None

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.mtx = np.array(config["Camera"]["Calibration"]["Mtx"])
        self.dist = np.array(config["Camera"]["Calibration"]["Dist"])
        logging.info("ImageProcessor 생성 완료.") 

    def undistort_image(self, image):
        try:
            if self.mtx is not None and self.dist is not None:
                undistorted_image = cv2.undistort(image, self.mtx, self.dist)
                return undistorted_image
            else:
                logging.warning("캘리브레이션 데이터가 설정되지 않았습니다.")
                return image
        except Exception as e:
            logging.error(f"이미지 보정에 실패했습니다: {e}")
            return image

    def apply_background_subtraction(self, image1, image2, background_img):
        try:
            def to_grayscale(img):
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            background_gray = to_grayscale(background_img)
            image1_gray = to_grayscale(image1)
            image2_gray = to_grayscale(image2)

            blurred_bg = cv2.GaussianBlur(background_gray, (5, 5), 0)
            blurred_img1 = cv2.GaussianBlur(image1_gray, (5, 5), 0)
            blurred_img2 = cv2.GaussianBlur(image2_gray, (5, 5), 0)

            diff1 = cv2.absdiff(blurred_bg, blurred_img1)
            diff2 = cv2.absdiff(blurred_bg, blurred_img2)

            sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_diff1 = cv2.filter2D(diff1, -1, sharpening_kernel)
            sharpened_diff2 = cv2.filter2D(diff2, -1, sharpening_kernel)
            
            _, binary_mask1 = cv2.threshold(
                sharpened_diff1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, binary_mask2 = cv2.threshold(
                sharpened_diff2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_mask1 = cv2.morphologyEx(binary_mask1, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary_mask2 = cv2.morphologyEx(binary_mask2, cv2.MORPH_CLOSE, kernel, iterations=2)

            return binary_mask1, binary_mask2

        except Exception as e:
            logging.error(f"배경 차감 적용에 실패했습니다: {e}")
            return None, None

class MeasurementTool(ImageProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.pixel_size_mm = config["Camera"]['Pixel_Size']
        self.lens_magnification = config["Camera"]["Lens_Magnification"]
        logging.info("MeasurementTool 생성 완료.")

    def calculate_camera_movement(self, image1, image2, background_img):
        try:
            binary_mask1, binary_mask2 = self.apply_background_subtraction(image1, image2, background_img)
            if binary_mask1 is None or binary_mask2 is None:
                logging.error("배경 차감된 마스크를 생성할 수 없습니다.")
                return None
            
            # 이곳에 Optical Flow 등을 이용한 이동 거리 및 회전 각도 계산 추가
            # 예제 값을 반환
            tx_mm = 0.1  # X축 이동 거리 (예시)
            ty_mm = 0.2  # Y축 이동 거리 (예시)
            theta_deg = 0.5  # 회전 각도 (예시)
            
            return {'tx': tx_mm, 'ty': ty_mm, 'theta': theta_deg}

        except Exception as e:
            logging.error(f"카메라 이동 거리 계산에 실패했습니다: {e}")
            return None

    def calculate_combined_movement(self, movements):
        try:
            if not movements:
                logging.error("이동 거리 정보를 통합할 데이터가 없습니다.")
                return None
            
            tx_values = [m['tx'] for m in movements if m['tx'] is not None]
            ty_values = [m['ty'] for m in movements if m['ty'] is not None]
            theta_values = [m['theta'] for m in movements if m['theta'] is not None]

            tx_avg = np.mean(tx_values) if tx_values else 0
            ty_avg = np.mean(ty_values) if ty_values else 0
            theta_avg = np.mean(theta_values) if theta_values else 0

            return {'tx': tx_avg, 'ty': ty_avg, 'theta': theta_avg}
        
        except Exception as e:
            logging.error(f"이동 거리 통합 계산에 실패했습니다: {e}")
            return None

def main():
    try:
        json_path = "/home/hims/TestCam/etc/config.json"
        config = load_config(json_path)
        if not config:
            return

        folder_path = setup_folders(os.path.expanduser(config["Path"]))
        if not folder_path:
            return

        config["Image_Save_Path"] = folder_path
        logging.info("설정 값: %s", json.dumps(config, indent=4, ensure_ascii=False))
        num_cameras = config["Num_Cameras"]

        measurement_tool = MeasurementTool(config)
        images1 = []
        images2 = []
        background_img = None

        if config["Test_Img_Capture"]:
            input("캡처할 위치로 이동하여 Enter 키를 눌러 Test 이미지를 캡처하세요...")
            for j in range(num_cameras):
                set_i2c_channel(j)
                camera_controller = CameraController(config, j)
                background_img = camera_controller.capture_image("background")
                camera_controller.close()
        else:
            for j in range(num_cameras):
                if config[f'Background_Img_Path_{j}'] and os.path.exists(f"{config[f'Background_Img_Path_{j}']}"):
                    background_img = cv2.imread(f"{config[f'Background_Img_Path_{j}']}")
                else:
                    logging.info(f"{j}번 카메라에 대한 배경 이미지가 없습니다. 새로운 이미지를 캡처합니다.")
                    input("배경을 찍을 위치에서 Enter 키를 눌러 배경 이미지를 캡처하세요...")
                    set_i2c_channel(j)
                    camera_controller = CameraController(config, j)
                    background_img = camera_controller.capture_image("background")
                    camera_controller.close()
        
        for capture_index, capture_list in enumerate([images1, images2], start=1):
            for j in range(num_cameras):
                logging.info(f"{j
