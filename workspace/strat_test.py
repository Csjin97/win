

import os
import json
import time
import logging
import glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

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

class MeasurementTool:
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
        self.zoom_factor = self.config.get("PreProcess", {}).get("Zoom", {}).get("Zoom_Factor", 1.0)
        logging.info("MeasurementTool 생성 완료.")

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

    def calculate_movement(self, image1, image2, background_img=None):
        try:
            sift_movement = None
            phase_diff_movement = None

            # 배경 차감 기반 이동 거리 계산
            if background_img is not None:
                binary_mask1, binary_mask2 = self.apply_background_subtraction(image1, image2, background_img)
                if binary_mask1 is not None and binary_mask2 is not None:
                    foreground1 = cv2.bitwise_and(image1, image1, mask=binary_mask1)
                    foreground2 = cv2.bitwise_and(image2, image2, mask=binary_mask2)
                    sift_movement = self.calculate_movement_sift(foreground1, foreground2)

            # 위상차 기반 이동 거리 계산
            phase_diff_movement = self.calculate_camera_movement_phase_difference(image1, image2)
            
            return sift_movement, phase_diff_movement
        except Exception as e:
            logging.error(f"이동 거리 계산에 실패했습니다: {e}")
            return None, None

    def calculate_movement_sift(self, image1, image2):
        gray1, gray2 = self.to_gray(image1), self.to_gray(image2)
        kp1, des1 = self.detect_and_compute_sift(gray1)
        kp2, des2 = self.detect_and_compute_sift(gray2)
        matches = self.match_features(des1, des2)
        if len(matches) < 4:  # 매칭이 충분하지 않은 경우
            logging.error("SIFT 이동 계산에 충분한 매칭이 없습니다.")
            return None
        return self.calculate_homography(kp1, kp2, matches, self.zoom_factor)

    def to_gray(self, image):
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def apply_background_subtraction(self, image1, image2, background_img):
        try:
            gray_bg, gray1, gray2 = self.to_gray(background_img), self.to_gray(image1), self.to_gray(image2)
            mask1 = cv2.absdiff(gray_bg, gray1)
            mask2 = cv2.absdiff(gray_bg, gray2)
            _, mask1 = cv2.threshold(mask1, 30, 255, cv2.THRESH_BINARY)
            _, mask2 = cv2.threshold(mask2, 30, 255, cv2.THRESH_BINARY)
            return mask1, mask2
        except Exception as e:
            logging.error(f"배경 차감 실패: {e}")
            return None, None

    def detect_and_compute_sift(self, image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        return [m for m, n in matches if m.distance < 0.7 * n.distance]

    def calculate_homography(self, kp1, kp2, matches, zoom_factor):
        if len(matches) < 4:
            logging.error("호모그래피를 계산하기 위한 충분한 매칭이 없습니다.")
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            logging.error("호모그래피 계산 실패: 유효한 H 행렬을 얻을 수 없습니다.")
            return None
        shift_y = H[1, 2]
        return (shift_y * self.pixel_size_mm) / (zoom_factor * self.lens_magnification)

    def calculate_camera_movement_phase_difference(self, image1, image2):
        try:
            f1 = np.fft.fft2(self.to_gray(image1))
            f2 = np.fft.fft2(self.to_gray(image2))
            if f1.shape != f2.shape:
                logging.error("이미지 크기가 다릅니다. 위상차 계산을 위해 크기를 조정하세요.")
                return None
            phase_diff = np.mean(np.angle(f2) - np.angle(f1))
            return phase_diff * self.pixel_size_mm / (2 * np.pi * self.lens_magnification)
        except Exception as e:
            logging.error(f"위상차 기반 이동 거리 계산 실패: {e}")
            return None

    def calculate_3d_movement(self, images):
        try:
            images = [self.undistort_image(img) for img in images]
            cx, cy = images[0].shape[1] / 2, images[0].shape[0] / 2
            Q = np.float32([[1, 0, 0, -cx],
                            [0, 1, 0, -cy],
                            [0, 0, 0, self.focal_length],
                            [0, 0, -1/self.baseline, 0]])
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
            disparity = stereo.compute(images[0], images[1])
            if disparity is None or not np.any(disparity > 0):
                logging.error("불일치 맵 계산 실패: 유효한 disparity 맵을 얻을 수 없습니다.")
                return None
            depth_map = cv2.reprojectImageTo3D(disparity, Q)
            return depth_map
        except Exception as e:
            logging.error(f"3D 이동 거리 계산 실패: {e}")
            return None

    def strat(self, images, background_img=None):
        logging.info("==== 측정 시작 ====")
        
        sift_movement, phase_diff_movement = self.calculate_movement(images[0], images[1], background_img)
        
        if sift_movement is not None or phase_diff_movement is not None:
            if sift_movement is not None:
                logging.info(f"배경 차감 기반 이동 거리: {sift_movement} mm")
            if phase_diff_movement is not None:
                logging.info(f"위상차 기반 이동 거리: {phase_diff_movement} mm")
            
            if sift_movement is not None and phase_diff_movement is not None:
                avg_movement = (sift_movement + phase_diff_movement) / 2
                logging.info(f"평균 이동 거리: {avg_movement:.2f} mm")
        else:
            logging.error("이동 거리 계산에 실패했습니다: 두 계산 결과가 모두 None입니다.")

        if len(images) >= 4:
            logging.info("3D 이동 계산 시작")
            depth_map = self.calculate_3d_movement(images[:2]) 
            logging.info(f"3D 이동 계산 결과: {depth_map}")

def main():
    json_file = "C:\\Users\\310tk\\Desktop\\workspace\\etc\\config3.json"
    config = load_config(json_file)
    if not config:
        return

    folder_path = setup_folders(os.path.expanduser(config["Path"]))
    if not folder_path:
        return

    config["Image_Save_Path"] = folder_path
    logging.info("설정 값: %s", json.dumps(config, indent=4, ensure_ascii=False))
    
    num_cameras = config["Num_Cameras"]
    measurementTool = MeasurementTool(config)

    background_image = []
    images1 = []
    images2 = []
    background_path = "C:\\Users\\310tk\\Desktop\\workspace\\lens_x_af_x\\background_img"
    img_path = "C:\\Users\\310tk\\Desktop\\workspace\\lens_x_af_x\\0"

    for i in range(num_cameras):
        background_image_pattern = os.path.join(background_path, f"{i}.jpg")
        first_image_pattern = os.path.join(img_path, f"first_{i:06}.jpg")
        second_image_pattern = os.path.join(img_path, f"second_{i:06}.jpg")

        background_images = glob.glob(background_image_pattern)
        first_images = glob.glob(first_image_pattern)
        second_images = glob.glob(second_image_pattern)

        if background_images:
            background_image.append(cv2.imread(background_images[0]))
            logging.info(f"Background image for camera {i} loaded: {background_images[0]}")

        if first_images:
            images1.append(cv2.imread(first_images[0]))
            logging.info(f"First image for camera {i} loaded: {first_images[0]}")
        else:
            logging.warning(f"First image for camera {i} not found.")
        
        if second_images:
            images2.append(cv2.imread(second_images[0]))
            logging.info(f"Second image for camera {i} loaded: {second_images[0]}")
        else:
            logging.warning(f"Second image for camera {i} not found.")
    
    logging.info(f"{len(background_image)} background images loaded.")
    logging.info(f"{len(images1)} first images and {len(images2)} second images loaded.")

    # 각 카메라에 대해 strat 메서드를 호출하여 이동 계산 수행
    for i in range(num_cameras):
        if i < len(images1) and i < len(images2):
            measurementTool.strat([images1[i], images2[i]], background_image[i] if i < len(background_image) else None)
        else:
            logging.warning(f"Camera {i}: 필요한 이미지가 부족하여 이동 계산을 건너뜁니다.")

    logging.info("프로그램 종료.")

if __name__ == "__main__":
    main()
