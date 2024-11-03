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
from scipy.spatial.transform import Rotation as R

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

class CameraController():
    def __init__(self, config, camera_index):
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

class MeasurementTool():
    def __init__(self, config):
        self.config = config
        self.pixel_size_mm = config["Camera"]['Pixel_Size']
        self.lens_magnification = config["Camera"]["Lens_Magnification"]
        self.focal_length = config["Camera"]['LensPosition']
        self.baseline = config["Camera"]["baseline"]
        self.mtx = np.array(config["Camera"]["Calibration"]["Mtx"])
        self.dist = np.array(config["Camera"]["Calibration"]["Dist"])
        self.rvecs = [np.array(rvec) for rvec in config["Camera"]["Calibration"]["Rvecs"]]
        self.tvecs = [np.array(tvec) for tvec in config["Camera"]["Calibration"]["Tvecs"]]
        
        
        logging.info("MeasurementTool 생성 완료.")
        
    def undistort_image(self, image):
        """
        캘리브레이션 데이터로 이미지를 왜곡 보정합니다.
        """
        try:
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                # 보정된 이미지 반환
                undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
                return undistorted_image
            else:
                logging.warning("캘리브레이션 데이터가 설정되지 않았습니다.")
                return image
        except Exception as e:
            logging.error(f"이미지 보정에 실패했습니다: {e}")
            return image
               
    def calculate_3d_movement(self, images):
        """카메라 쌍마다 이동을 계산하여 카메라와 객체의 이동 거리와 회전 각도를 계산합니다."""
        if self.mtx is not None or self.dist is not None:
            # 이미지 왜곡 보정 적용
            images = [self.undistort_image(img) for img in images]
        
        cy, cx = images[0].shape[:2]
        cx /= 2
        cy /= 2
        
        Q = np.float32([[1, 0, 0, -cx],
                        [0, 1, 0, -cy],
                        [0, 0, 0, self.focal_length],
                        [0, 0, -1/self.baseline, 0]])
        
        depth_maps = []
        all_points_3d = []
        
        for i in range(0, 4, 2):  # 0-1, 2-3 두 쌍
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
            disparity = stereo.compute(images[i], images[i+1])
            depth_map = cv2.reprojectImageTo3D(disparity, Q)
            depth_maps.append(depth_map)
            
            # SIFT 특징점 추출 및 매칭
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(images[i], None)
            keypoints2, descriptors2 = sift.detectAndCompute(images[i+1], None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            # 3D 포인트 추출
            points_3d = np.array([depth_map[int(keypoints1[m.queryIdx].pt[1]), int(keypoints1[m.queryIdx].pt[0])]
                                    for m in good_matches])
            if self.rvecs and self.tvecs:
                points_3d_transformed = cv2.transform(points_3d.reshape(-1, 1, 3), cv2.Rodrigues(self.rvecs[i//2])[0]) + self.tvecs[i//2]
                all_points_3d.append(points_3d_transformed.reshape(-1, 3))
            else:
                all_points_3d.append(points_3d)
        
        # 객체 이동 및 회전 계산
        total_translation = np.array([0.0, 0.0, 0.0])
        total_rotation = np.eye(3)
        
        # 카메라 이동 계산
        camera_movement = np.array([0.0, 0.0, 0.0])
        
        for i in range(1, len(all_points_3d)):
            pts_prev = all_points_3d[i - 1]
            pts_current = all_points_3d[i]
            centroid_prev = np.mean(pts_prev, axis=0)
            centroid_current = np.mean(pts_current, axis=0)
            pts_prev_centered = pts_prev - centroid_prev
            pts_current_centered = pts_current - centroid_current

            # SVD로 회전 행렬과 이동 벡터 추정
            H = np.dot(pts_prev_centered.T, pts_current_centered)
            U, _, Vt = np.linalg.svd(H)
            R_i = np.dot(Vt.T, U.T)
            t_i = centroid_current - np.dot(R_i, centroid_prev)
            
            # 카메라 이동 거리와 객체 이동 거리 분리
            camera_movement += np.abs(t_i)  # 카메라의 실제 이동 거리
            
            # 객체 이동 및 회전 누적
            total_translation += t_i
            total_rotation = np.dot(R_i, total_rotation)

        # 평균 이동 및 회전 계산
        avg_translation = total_translation / (len(all_points_3d) - 1)
        avg_camera_movement = camera_movement / (len(all_points_3d) - 1)
        rotation_obj = R.from_matrix(total_rotation)
        avg_rotation_angle = rotation_obj.as_euler('xyz', degrees=True)
        
        logging.info("\n==== 카메라 이동 거리 및 객체 전체 이동 및 회전 ====")
        logging.info(f"평균 카메라 이동 거리: x={avg_camera_movement[0]:.2f} mm, y={avg_camera_movement[1]:.2f} mm, z={avg_camera_movement[2]:.2f} mm")
        logging.info(f"객체 이동량: x={avg_translation[0]:.2f} mm, y={avg_translation[1]:.2f} mm, z={avg_translation[2]:.2f} mm")
        logging.info(f"객체 회전 각도 (x, y, z): {avg_rotation_angle[0]:.2f}도, {avg_rotation_angle[1]:.2f}도, {avg_rotation_angle[2]:.2f}도")

def main():
    json_file = "/home/hims/TestCam/etc/config3.json"
    config = load_config(json_file)
    if not config:
        return

    folder_path = setup_folders(os.path.expanduser(config["Path"]))
    if not folder_path:
        return
    config["Image_Save_Path"] = folder_path
    logging.info("설정 값: %s", json.dumps(config, indent=4, ensure_ascii=False))
    
    num_cameras = config["Num_Cameras"]

    measurement_tool = MeasurementTool(config)
    images = []
    for i in range(num_cameras):
        logging.info(f"{i}번 카메라 초기화 중...")
        set_i2c_channel(i)
        camera_controller = CameraController(config, i)
        
        input("객체를 움직인 후 Enter 키를 눌러 첫 번째 이미지를 캡처하세요...")
        image1 = camera_controller.capture_image("First")
        
        input("객체를 움직인 후 Enter 키를 눌러 두 번째 이미지를 캡처하세요...")
        image2 = camera_controller.capture_image("Second")
        
        images.append(image1)
        images.append(image2)
        
        camera_controller.close()
        
    # 이동 거리 및 회전 각도 계산
    measurement_tool.calculate_3d_movement(images)
                    
if __name__ == "__main__":
    main()
