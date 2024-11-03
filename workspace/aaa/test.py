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
    """
    I2C 멀티플렉서 채널을 설정하는 함수.
    카메라의 특정 채널로 전환합니다.
    """
    I2C_MUX_ADDRESS = 0x24  # I2C 멀티플렉서 주소
    CONTROL_REGISTER = 0x24  # 제어 레지스터
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
        with SMBus(10) as bus:  # SMBus 10번 버스 사용
            channel_value = channels[channel]
            bus.write_byte_data(I2C_MUX_ADDRESS, CONTROL_REGISTER, channel_value)
            logging.info(f"I2C 채널 {channel} 설정 성공.")
    except Exception as e:
        logging.error(f"I2C 채널 {channel} 설정에 실패했습니다: {e}")

def load_config(json_path):
    """
    JSON 파일에서 카메라 설정 값을 로드하는 함수.
    설정 파일을 불러와서 반환합니다.
    """
    try:
        with open(json_path, 'r') as file:
            settings = json.load(file)
        return settings
    
    except Exception as e:
        print(f"카메라 설정을 로드하는 데 실패했습니다: {e}")
        return None
    
def setup_folders(base_path):
    """
    이미지와 로그 파일을 저장할 폴더를 설정하고 로깅을 구성하는 함수.
    """
    try:
        # 폴더 경로 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join(base_path, timestamp)
        log_folder_path = os.path.join(folder_path, "log")
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(log_folder_path, exist_ok=True)

        # 로깅 설정
        log_level = logging.DEBUG
        log_file_path = os.path.join(log_folder_path, "log.txt")

        # 로깅 구성
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
        self.rvecs = [np.array(rvec) for rvec in config["Camera"]["Calibration"]["Rvecs"]]
        self.tvecs = [np.array(tvec) for tvec in config["Camera"]["Calibration"]["Tvecs"]]
        logging.info("ImageProcessor 생성 완료.") 
         
    def undistort_image(self, image):
        """
        캘리브레이션 데이터로 이미지를 왜곡 보정합니다.
        """
        try:
            if self.mtx is not None and self.dist is not None:
                # 보정된 이미지 반환
                undistorted_image = cv2.undistort(image, self.mtx, self.dist)
                return undistorted_image
            else:
                logging.warning("캘리브레이션 데이터가 설정되지 않았습니다.")
                return image
        except Exception as e:
            logging.error(f"이미지 보정에 실패했습니다: {e}")
            return image
            
    def apply_background_subtraction(self, image1, image2, background_img):
        """배경 이미지와 현재 이미지의 차이를 계산"""
        try:
            # 1. 그레이스케일 변환
            def to_grayscale(img):
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            background_gray = to_grayscale(background_img)
            image1_gray = to_grayscale(image1)
            image2_gray = to_grayscale(image2)

            # 2. Gaussian Blur 적용하여 노이즈 감소
            blurred_bg = cv2.GaussianBlur(background_gray, (5, 5), 0)
            blurred_img1 = cv2.GaussianBlur(image1_gray, (5, 5), 0)
            blurred_img2 = cv2.GaussianBlur(image2_gray, (5, 5), 0)

            # 3. 절대 차이 계산
            diff1 = cv2.absdiff(blurred_bg, blurred_img1)
            diff2 = cv2.absdiff(blurred_bg, blurred_img2)

            # 4. 샤프닝 커널 정의 및 적용
            sharpening_kernel = np.array([[0, -1, 0],
                                        [-1, 5,-1],
                                        [0, -1, 0]])
            sharpened_diff1 = cv2.filter2D(diff1, -1, sharpening_kernel)
            sharpened_diff2 = cv2.filter2D(diff2, -1, sharpening_kernel)
            
            # 5. 이진화를 통해 마스크 생성
            _, binary_mask1 = cv2.threshold(
                sharpened_diff1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, binary_mask2 = cv2.threshold(
                sharpened_diff2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # 6. 모폴로지 연산을 통해 노이즈 제거 및 객체 형태 정리
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_mask1 = cv2.morphologyEx(binary_mask1, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary_mask1 = cv2.morphologyEx(binary_mask1, cv2.MORPH_OPEN, kernel, iterations=2)

            binary_mask2 = cv2.morphologyEx(binary_mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary_mask2 = cv2.morphologyEx(binary_mask2, cv2.MORPH_OPEN, kernel, iterations=2)

            # 7. 작은 객체 제거
            contours1, _ = cv2.findContours(binary_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours1:
                if cv2.contourArea(contour) < 500:  # 임계값 조정 가능
                    cv2.drawContours(binary_mask1, [contour], -1, 0, -1)

            contours2, _ = cv2.findContours(binary_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours2:
                if cv2.contourArea(contour) < 500:
                    cv2.drawContours(binary_mask2, [contour], -1, 0, -1)

            # 8. 이진 마스크 저장
            img_path1 = os.path.join(
                self.config["Image_Save_Path"],
                f"optical_flow_1_sharpen_morph.jpg"
            )
            img_path2 = os.path.join(
                self.config["Image_Save_Path"],
                f"optical_flow_2_sharpen_morph.jpg"
            )
            cv2.imwrite(img_path1, binary_mask1)
            cv2.imwrite(img_path2, binary_mask2)

            return binary_mask1, binary_mask2

        except Exception as e:
            logging.error(f"배경 차감 적용에 실패했습니다: {e}")
            return None, None
 
    def image_mask_check(self, image, mask):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if len(mask.shape) != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return image, mask
    
    def process_image(self, image):
        """이미지를 전처리"""
        if self.mtx is not None and self.dist is not None:
            image = self.undistort_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # image = cv2.GaussianBlur(image, (5, 5), 0)

        # # CLAHE 적용
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # image = clahe.apply(image)

        # PreProcess 설정 가져오기
        preprocess = self.config.get("PreProcess", {})
        
        # 각 전처리 단계 순차적으로 적용
        for step_name, step_config in preprocess.items():
            if step_config.get("Use", False):
                method_name = f"process_{step_name}".lower()
                process_method = getattr(self, method_name, None)
                if callable(process_method):
                    logging.debug(f"{step_name} 전처리를 시작합니다.")
                    image = process_method(image, step_config)
                    if image is None:
                        logging.error(f"{step_name} 전처리 중 오류가 발생했습니다.")
                        return None
                else:
                    logging.warning(f"'{step_name}' 전처리 방법이 구현되지 않았습니다.")
        return image
    
    def process_roi(self, image, config):
        """
        ROI 설정을 적용합니다.
        """
        try:
            coordinates = config.get("Coordinates", {})
            x = coordinates.get("x")
            y = coordinates.get("y")
            w = coordinates.get("w")
            h = coordinates.get("h")

            # 좌표와 크기 유효성 검증
            if None in (x, y, w, h):
                logging.error("ROI의 좌표 또는 크기가 누락되었습니다.")
                return None

            camera_config = self.config.get("Camera", {})
            camera_width = camera_config.get("Width")
            camera_height = camera_config.get("Height")

            # 카메라 설정 유효성 검증
            if None in (camera_width, camera_height):
                logging.error("카메라의 Width 또는 Height 설정이 누락되었습니다.")
                return None

            # ROI가 카메라 크기를 초과하는지 확인
            if x + w > camera_width or y + h > camera_height:
                logging.error("ROI가 이미지 크기를 초과합니다. 설정을 조정하세요.")
                return None

            # ROI 적용
            image_roi = image[y:y+h, x:x+w]
            logging.debug(f"ROI를 적용했습니다: x={x}, y={y}, w={w}, h={h}")
            return image_roi

        except Exception as e:
            logging.exception("ROI 처리 중 오류가 발생했습니다.")
            return None

    def process_zoom(self, image, config):
        """디지털 줌 기능 사용"""
        try:
            zoom_factor = config.get("Zoom_Factor", 1.0)
            if zoom_factor != 1.0:
                # height, width = self.config["Camera"]["Width"], self.config["Camera"]["Height"]
                width, height = image.shape[:2]
                new_width = int(width / zoom_factor)
                new_height = int(height / zoom_factor)
                x1 = (width - new_width) // 2
                y1 = (height - new_height) // 2
                image = image[y1:y1+new_height, x1:x1+new_width]
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
                logging.info(f"디지털 줌 적용: 줌 배율 = {zoom_factor}")
            return image
        except Exception as e:
            logging.error(f"디지털 줌 적용에 실패했습니다: {e}")
            return None

class CameraController(ImageProcessor):
    def __init__(self, config, camera_index):
        super().__init__(config) 
        self.config = config
        self.camera_index = camera_index
        self.picam2 = Picamera2()
        self.image_save_path = config["Image_Save_Path"]
        self.setup_camera()
        logging.info(f"CameraController {self.camera_index} 생성 완료.")

    def setup_camera(self):
        """카메라의 해상도와 기본 설정을 구성하고 카메라 컨트롤을 설정합니다."""
        try:
            configuration = self.picam2.create_still_configuration({
                "size": (self.config["Camera"]["Width"], self.config["Camera"]["Height"])
            })
            
            self.picam2.configure(configuration)
            controls_dict = self.create_camera_control_dict()
            logging.info(f"초점 설정: {controls_dict}")
            self.picam2.set_controls(controls_dict)
            time.sleep(1)
            logging.info(f"{self.camera_index}번 카메라 설정 완료.")
        except Exception as e:
            logging.error(f"{self.camera_index}번 카메라 설정을 적용하는 데 실패했습니다: {e}")


    def create_camera_control_dict(self):
        """카메라 컨트롤을 위한 설정 값들을 딕셔너리로 생성하여 반환합니다."""
        controls_dict = {}
        # 프레임 지속 시간 설정
        frame_duration = self.get_frame_duration_limits()
        if frame_duration:
            controls_dict['FrameDurationLimits'] = frame_duration
        # 노출 시간 설정
        exposure_time = self.get_exposure_time()
        if exposure_time:
            controls_dict['ExposureTime'] = exposure_time
        # 카메라의 감도(민감도)를 설정
        analogue_gain = self.config["Camera"]["AnalogueGain"]
        if analogue_gain:
            controls_dict['AnalogueGain'] = analogue_gain
        # 색 균형 설정
        colour_gains = self.config["Camera"]["ColourGains"]
        if colour_gains:
            controls_dict['ColourGains'] = colour_gains
        # 자동 초점 설정
        if self.config["Camera"]['AF']:
            controls_dict = self.get_autofocus_mode(controls_dict)
        else:
            controls_dict["AfMode"] = controls.AfModeEnum.Manual
            controls_dict["LensPosition"] = self.config.get("Camera", {}).get("LensPosition", 10)

        return controls_dict

    def get_frame_duration_limits(self):
        """FPS 값을 바탕으로 프레임의 지속 시간을 계산하여 반환합니다."""
        fps = self.config["Camera"]["FPS"]
        if fps:
            frame_duration = int(1e6 / fps)
            return (frame_duration, frame_duration)
        return None

    def get_exposure_time(self):
        """설정 파일에서 노출 시간을 가져와 반환합니다."""
        exposure_time = self.config["Camera"]["Exposure_Time"]
        if exposure_time:
            return int(exposure_time * 1e3)
        return None

    def get_autofocus_mode(self, controls_dict):
        """자동 초점 설정 반환"""
        try:
            controls_dict["AfMode"] = controls.AfModeEnum.Auto
            af_mode = self.config["Camera"]["AF_MODE"]
            controls_dict["AfRange"] = getattr(controls.AfRangeEnum, af_mode.get("AfRange", "Normal"))
            controls_dict["AfSpeed"] = getattr(controls.AfSpeedEnum, af_mode.get("AfSpeed", "Normal"))
            controls_dict["AfMetering"] = getattr(controls.AfMeteringEnum, af_mode.get("AfMetering", "Auto"))

            af_windows = af_mode.get("AfWindows", None) 
            if af_windows is not None:
                controls_dict["AfWindows"] = af_windows
                
            return controls_dict

        except Exception as e:
            logging.error(f"{self.camera_index}번 카메라의 자동 초점 컨트롤을 설정하는 데 실패했습니다: {e}")
            return None 
        
    def perform_autofocus(self):
        """자동 초점 사이클을 실행하고 렌즈 위치를 업데이트합니다."""
        try:
            result = self.picam2.autofocus_cycle()
            if result:
                time.sleep(2)
                lens_position = self.get_focal_length()
                if lens_position:
                    self.config["Camera"]['LensPosition'] = lens_position
            else:
                logging.error(f"{self.camera_index}번 카메라의 자동 초점에 실패했습니다.")
        except Exception as e:
            logging.error(f"{self.camera_index}번 카메라의 자동 초점에 실패했습니다: {e}")

    def get_focal_length(self):
        """카메라의 메타데이터에서 초점 거리 정보를 추출합니다."""
        try:
            retries = 3
            for _ in range(retries):
                metadata = self.picam2.capture_metadata()
                focal_length = metadata.get("LensPosition")                
                if focal_length:
                    logging.info(f"{self.camera_index}번 카메라의 초점 거리: {focal_length}")
                    return focal_length
                time.sleep(0.5)
            raise ValueError("초점 거리를 추출할 수 없습니다.")
        except Exception as e:
            logging.error(f"초점 거리를 추출하는 데 실패했습니다: {e}")
            return None

    def capture_image(self, prefix):
        """카메라 이미지를 캡처하고, 이미지 전처리 후 저장합니다."""
        try:
            logging.info(f"{self.camera_index}번 카메라에서 {prefix} 이미지를 캡처 중...")

            # 카메라 시작
            self.picam2.start()
            
            if self.config["Camera"]["AF"]:
                self.perform_autofocus()
                time.sleep(2)

            image = self.picam2.capture_array()
            image = self.process_image(image)
            self.picam2.stop()

            if image is None:
                logging.error(f"{self.camera_index}번 카메라에서 캡처된 이미지가 없습니다.")
                return None

            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_path = os.path.join(self.image_save_path, f"{prefix}_image{self.camera_index}.jpg")
            cv2.imwrite(img_path, image)

            return image
        except Exception as e:
            logging.error(f"{self.camera_index}번 카메라에서 이미지를 캡처하는 데 실패했습니다: {e}")
            return None
                    
    def close(self):
        """카메라 리소스를 해제하고, 카메라 컨트롤러를 종료합니다."""
        try:
            self.picam2.close()
            logging.info(f"CameraController {self.camera_index}가 성공적으로 종료되었습니다.")
        except Exception as e:
            logging.error(f"CameraController {self.camera_index} 종료에 실패했습니다: {e}")

class MeasurementTool(ImageProcessor):
    def __init__(self, config):
        self.config = config
        self.pixel_size_mm = config["Camera"]['Pixel_Size']
        self.lens_magnification = config["Camera"]["Lens_Magnification"]
        self.orb = cv2.ORB_create(
            nfeatures=5000,  # 특징점 수
            scaleFactor=1.2,  # 피라미드 스케일 팩터 
            nlevels=12,  # 피라미드 레벨 수
            edgeThreshold=15,  # 가장자리 임계값 
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=5  # FAST 임계값
        )
        self.feature_params = dict(
                maxCorners=500,
                qualityLevel=0.01,
                minDistance=3,
                blockSize=7
            )
        self.lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
        self.movement_history = deque(maxlen=10)
        logging.info("MeasurementTool 생성 완료.")
    
    def move_calculation(self, image1, image2, prefix, camera_index, movement, background_img):
        """Optical Flow를 이용하여 두 이미지 사이의 객체 이동 및 회전 각도를 계산합니다."""
        try:
            # 1. 배경 차감 적용
            bg_Mask1, bg_Mask2 = self.apply_background_subtraction(image1, image2, background_img)
            image1_gray, bg_Mask1 = self.image_mask_check(image1, bg_Mask1)
            image2_gray, bg_Mask2 = self.image_mask_check(image2, bg_Mask2)

            # 2. 좋은 특징점 검출
            p0 = cv2.goodFeaturesToTrack(image1_gray, mask=bg_Mask1, **self.feature_params)

            if p0 is None:
                logging.error("첫 번째 이미지에서 특징점을 찾을 수 없습니다.")
                return None, None, None

            # 3. 두 번째 이미지에서 특징점 추적
            p1, st, err = cv2.calcOpticalFlowPyrLK(image1_gray, image2_gray, p0, None, **self.lk_params)

            # 4. 좋은 특징점 선택
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 10:
                logging.error("충분한 좋은 특징점 매칭을 찾을 수 없습니다.")
                return None, None, None

            # 5. 어파인 변환 추정을 위한 데이터 준비
            pts_old = np.float32(good_old).reshape(-1, 2)
            pts_new = np.float32(good_new).reshape(-1, 2)

            # 6. 어파인 변환 추정 (회전 및 평행 이동 포함)
            M, inliers = cv2.estimateAffinePartial2D(
                pts_old, pts_new, method=cv2.RANSAC,
                ransacReprojThreshold=2.0, maxIters=5000, confidence=0.99
            )

            if M is None or inliers is None or np.sum(inliers) < 10:
                logging.error("어파인 변환 추정에 실패했습니다.")
                return None, None, None

            # 7. 변환 행렬에서 스케일, 이동량, 회전 각도 추출
            sx = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
            sy = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
            scale = (sx + sy) / 2

            # 8. 줌 배율로 스케일 보정
            if self.config["PreProcess"]["Zoom"]["Use"]:
                scale /= self.config["PreProcess"]["Zoom"]["Zoom_Factor"]

            # 9. 이동량 계산
            tx_pixels = M[0, 2]
            ty_pixels = M[1, 2]

            # 10. 카메라 이동 보정 (y축)
            if movement is not None:
                movement_pixels = movement * self.lens_magnification / self.pixel_size_mm
                ty_pixels_corrected = ty_pixels - movement_pixels
                ty_mm = (ty_pixels_corrected / scale) * self.pixel_size_mm / self.lens_magnification
                logging.info(f"카메라 이동 보정된 Y 이동량: {ty_mm:.2f} mm")
            else:
                ty_mm = (ty_pixels / scale) * self.pixel_size_mm / self.lens_magnification

            # X축 이동량도 필요 시 보정 (movement 파라미터가 x축 이동도 포함한다면)
            # 현재 y축만 보정한다고 가정
            tx_mm = (tx_pixels / scale) * self.pixel_size_mm / self.lens_magnification

            # 11. 회전 각도 계산
            angle_rad = np.arctan2(M[1, 0], M[0, 0])
            theta_deg = np.degrees(angle_rad)

            # 12. 특징점 좌표를 배열 형태로 변환 (인라이어만)
            good_old_pts = pts_old[inliers.flatten() == 1]
            good_new_pts = pts_new[inliers.flatten() == 1]

            # 13. Optical Flow 시각화
            # 두 이미지를 가로로 합쳐 하나의 이미지로 만듭니다.
            height1, width1 = image1_gray.shape
            height2, width2 = image2_gray.shape

            # 두 이미지의 높이를 맞춥니다.
            if height1 != height2 or width1 != width2:
                logging.warning("두 이미지의 크기가 다릅니다. 시각화를 위해 크기를 조정합니다.")
                height = max(height1, height2)
                width = max(width1, width2)
                image1_gray_resized = cv2.resize(image1_gray, (width, height))
                image2_gray_resized = cv2.resize(image2_gray, (width, height))
            else:
                height = height1
                width = width1
                image1_gray_resized = image1_gray
                image2_gray_resized = image2_gray

            combined_img = np.zeros((height, width * 2), dtype=np.uint8)
            combined_img[:, :width] = image1_gray_resized
            combined_img[:, width:] = image2_gray_resized

            # 컬러 이미지로 변환하여 시각화에 사용합니다.
            combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)

            # 두 번째 이미지의 특징점 좌표에 너비만큼 오프셋 추가
            good_new_pts_offset = good_new_pts.copy()
            good_new_pts_offset[:, 0] += width

            # 특징점 이동을 화살표로 시각화
            for i in range(len(good_old_pts)):
                old_pt = tuple(good_old_pts[i].astype(int))
                new_pt = tuple(good_new_pts_offset[i].astype(int))
                movement_vector = (new_pt[0] - old_pt[0], new_pt[1] - old_pt[1])

                # 이동 방향에 따라 색상 설정 (위로 이동: 빨간색, 아래로 이동: 초록색)
                color = (0, 0, 255) if movement_vector[1] > 0 else (0, 255, 0)

                # 화살표 그리기
                cv2.arrowedLine(combined_img_color, old_pt, new_pt, color, 2, tipLength=0.1)

                # 원 그리기
                cv2.circle(combined_img_color, old_pt, 3, (0, 0, 255), -1)  # 빨간색
                cv2.circle(combined_img_color, new_pt, 3, (0, 255, 0), -1)  # 초록색

            # 이동 거리 및 회전 각도 텍스트 추가
            text = f"Movement: X={tx_mm:.2f} mm, Y={ty_mm:.2f} mm | Rotation: {theta_deg:.2f} deg"
            cv2.putText(combined_img_color, text, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 시각화 이미지 저장
            match_img_path = os.path.join(
                self.config["Image_Save_Path"],
                f"{prefix}_optical_flow_{camera_index}.jpg"
            )
            cv2.imwrite(match_img_path, combined_img_color)
            logging.info(f"Optical flow 시각화 이미지 저장: {match_img_path}")
            logging.info(f"객체의 이동: X={tx_mm:.2f} mm, Y={ty_mm:.2f} mm")
            logging.info(f"객체의 회전 각도: {theta_deg:.2f} 도")

        except Exception as e:
            logging.exception(f"Optical Flow를 사용한 객체 이동 계산에 실패했습니다: {e}")
            
    def remove_small_objects(self, binary_mask, min_area=500):
        """이진 마스크에서 작은 객체를 제거합니다."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(binary_mask, [contour], -1, 0, -1)
        return binary_mask
    
    def detect_and_compute_orb(self, gray_image):
        """
        ORB를 사용하여 키포인트 검출 및 디스크립터 추출

        Args:
            gray_image (np.ndarray): 전처리된 그레이스케일 이미지.

        Returns:
            tuple: 키포인트와 디스크립터.
        """
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        if keypoints:
            # 서브픽셀 정밀도 조정
            corners = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1,1,2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            refined_corners = cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
            for i, kp in enumerate(keypoints):
                kp.pt = tuple(refined_corners[i][0])
        return keypoints, descriptors

    def match_features(self, des1, des2):
        """
        디스크립터 매칭

        Args:
            des1 (np.ndarray): 첫 번째 이미지의 디스크립터.
            des2 (np.ndarray): 두 번째 이미지의 디스크립터.

        Returns:
            list: 매칭된 특징점 리스트.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def calculate_homography(self, kp1, kp2, matches, zoom_factor):
        """
        호모그래피 계산 및 이동 벡터 추출

        Args:
            kp1 (list): 첫 번째 이미지의 키포인트.
            kp2 (list): 두 번째 이미지의 키포인트.
            matches (list): 매칭된 키포인트 리스트.

        Returns:
            float: 수직 이동 거리(mm) 또는 None.
        """
        if len(matches) < 4:
            logging.error("호모그래피를 계산하기 위한 충분한 매칭이 없습니다.")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            shift_y = H[1, 2]
            movement_distance_mm = (shift_y * self.pixel_size_mm) / (zoom_factor * self.lens_magnification)

            logging.info(f"ORB 기반 이동 거리: {movement_distance_mm:.2f} mm")
            return movement_distance_mm
        else:
            logging.error("호모그래피 계산에 실패했습니다.")
            return None

    def detect_and_compute_sift(self, image):
        # SIFT 검출기 생성
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        # FLANN 기반 매처 설정
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 매칭 시 확인할 트리 개수
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test 적용
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def calculate_movement_sift(self, foreground1, foreground2, zoom_factor=1.0):
        """
        SIFT 기반 이동 거리 계산
        """
        try:
            # 채널 수 확인 및 그레이스케일 변환
            if len(foreground1.shape) == 3 and foreground1.shape[2] == 3:
                gray1 = cv2.cvtColor(foreground1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = foreground1.copy()
                
            if len(foreground2.shape) == 3 and foreground2.shape[2] == 3:
                gray2 = cv2.cvtColor(foreground2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = foreground2.copy()
                
            kp1, des1 = self.detect_and_compute_sift(gray1)
            kp2, des2 = self.detect_and_compute_sift(gray2)

            if des1 is None or des2 is None:
                logging.error("디스크립터를 추출할 수 없습니다.")
                return None

            # 특징점 매칭
            matches = self.match_features(des1, des2)

            # 호모그래피 계산 및 이동 거리 추정
            movement_distance_mm = self.calculate_homography(kp1, kp2, matches, zoom_factor)

            return movement_distance_mm

        except Exception as e:
            logging.error(f"SIFT 기반 이동 거리 계산 실패: {e}")
            return None

    def calculate_camera_movement(self, image1, image2, background_img):
        """
        배경 차감 및 SIFT 기반 방법을 사용하여 카메라의 수직 이동 거리를 계산합니다.
        """
        try:
            # 1. 배경 차감 적용 (마스크 반환)
            binary_mask1, binary_mask2 = self.apply_background_subtraction(image1, image2, background_img)
            if binary_mask1 is None or binary_mask2 is None:
                logging.error("배경 차감된 마스크를 생성할 수 없습니다.")
                return None
            
            # 3. 줌 배율 보정 값
            if self.config.get("PreProcess", {}).get("Zoom", {}).get("Use", False):
                zoom_factor = self.config["PreProcess"]["Zoom"].get("Zoom_Factor", 1.0)
            else:
                zoom_factor = 1.0
                
            # 4. 배경 차감된 마스크를 사용하여 전경 이미지 생성
            foreground1 = cv2.bitwise_and(image1, image1, mask=binary_mask1)
            foreground2 = cv2.bitwise_and(image2, image2, mask=binary_mask2)

            # 5. SIFT 기반 이동 거리 계산
            movement_distance_mm = self.calculate_movement_sift(foreground1, foreground2, zoom_factor)

            self.calculate_camera_movement_phase_difference(image1, image2)
            return movement_distance_mm

        except Exception as e:
            logging.error(f"카메라 이동 거리 계산에 실패했습니다: {e}")
            return None
    
    def calculate_camera_movement_phase_difference(self, image1, image2):
        """
        위상차 기반으로 카메라의 수직 이동 거리를 계산합니다.

        """
        try:
            # 그레이스케일 변환
            if len(image1.shape) == 3:
                image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            else:
                image1_gray = image1

            if len(image2.shape) == 3:
                image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            else:
                image2_gray = image2

            # 푸리에 변환을 통해 주파수 도메인에서 이미지 분석
            f1 = np.fft.fft2(image1_gray)
            f2 = np.fft.fft2(image2_gray)

            # 위상 차이 계산
            phase_diff = np.angle(f2) - np.angle(f1)

            # 위상차의 평균 또는 특정 값으로 이동 거리를 계산
            
            phase_shift = np.mean(phase_diff)
            
            # 이동 거리 계산
            movement_distance_mm = phase_shift * self.pixel_size_mm / (2 * np.pi * self.lens_magnification)
            
            logging.info(f"위상차 기반 이동 거리: {movement_distance_mm:.2f} mm")
            return movement_distance_mm

        except Exception as e:
            logging.error(f"위상차 기반 카메라 이동 거리 계산 실패: {e}")
            return None
        
def main():
    """프로그램의 메인 실행 함수"""
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

        # Background Image Capture
        if config["Test_Img_Capture"]:
            input("캡처할 위치로 이동하여 Enter 키를 눌러 Test 이미지를 캡처하세요...")
            for j in range(num_cameras):
                set_i2c_channel(j)
                camera_controller = CameraController(config, j)
                background_img = camera_controller.capture_image("background")
                camera_controller.close()
        else:
            # 각 카메라의 배경 이미지가 존재하는지 확인
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
        
        # Image Capture for First and Second Time
        for capture_index, capture_list in enumerate([images1, images2], start=1):
            for j in range(num_cameras):
                logging.info(f"{j}번 카메라 초기화 중...")
                set_i2c_channel(j)
                camera_controller = CameraController(config, j)

                # 이미지 경로 정의
                first_image_path = config.get(f"capture_img{j}_path", "")
                second_image_path = config.get(f"img2_path", "")

                # 첫 번째 시간대의 이미지 존재 확인 및 로드 또는 캡처
                if capture_index == 1:
                    if first_image_path and os.path.exists(first_image_path):
                        image = cv2.imread(first_image_path)
                    else:
                        input(f"객체를 움직인 후 Enter 키를 눌러 첫 번째 {j}번 카메라 이미지를 캡처하세요...")
                        image = camera_controller.capture_image("first")

                # 두 번째 시간대의 이미지 존재 확인 및 로드 또는 캡처
                elif capture_index == 2:
                    if second_image_path and os.path.exists(second_image_path):
                        image = cv2.imread(second_image_path)
                    else:
                        # input(f"객체를 움직인 후 Enter 키를 눌러 두 번째 {j}번 카메라 이미지를 캡처하세요...")
                        image = camera_controller.capture_image("second")

                # 각 시간대 리스트에 이미지 추가
                capture_list.append(image)
                camera_controller.close()
        
        # # Calculate Camera Movement Between Two Captures
        # for k in range(num_cameras):
        #     image1 = images1[k]
        #     image2 = images2[k]
        #     camera_movement = measurement_tool.calculate_camera_movement(image1, image2, background_img)
        #     if camera_movement is not None:
        #         logging.info(f"{k}번 카메라의 수직 이동 거리: {camera_movement:.6f} mm")
    
    except Exception as e:
        logging.error(f"오류가 발생했습니다: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
    
