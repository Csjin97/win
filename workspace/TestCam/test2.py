# Stereo Vision with Multiple Cameras using Picamera2
# 중단

import cv2
import numpy as np
import os
import json
import time
import logging
import traceback
from picamera2 import Picamera2
from collections import deque
import threading

def load_config(json_path):
    """
    JSON 파일에서 카메라 설정 값을 로드하는 함수.
    설정 파일을 불러와서 반환합니다.
    """
    try:
        with open(json_path, 'r') as file:
            settings = json.load(file)
        logging.info(f"설정 파일 로드 성공: {json_path}")
        return settings
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {json_path}")
    except json.JSONDecodeError as e:
        logging.error(f"설정 파일의 JSON 형식이 잘못되었습니다: {e}")
    except Exception as e:
        logging.error(f"설정 파일을 로드하는 도중 알 수 없는 오류 발생: {e}")
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
        logging.info("ImageProcessor 생성 완료.")  
    
    def apply_background_subtraction(self, image1, image2, background_img):
        """배경 이미지와 현재 이미지의 차이를 계산"""
        try:
            # 1. 그레이스케일 변환
            background_gray = self.to_grayscale(background_img)
            image1_gray = self.to_grayscale(image1)
            image2_gray = self.to_grayscale(image2)

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
            binary_mask1 = self.remove_small_objects(binary_mask1, min_area=500)
            binary_mask2 = self.remove_small_objects(binary_mask2, min_area=500)

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
            logging.error(f"배경 차감 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return None, None

    def to_grayscale(self, img):
        """이미지를 그레이스케일로 변환"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    def image_mask_check(self, image, mask):
        """이미지와 마스크의 타입과 형태를 확인하고 필요한 변환을 수행"""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if len(mask.shape) != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return image, mask

    def remove_small_objects(self, binary_mask, min_area=500):
        """이진 마스크에서 작은 객체를 제거합니다."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(binary_mask, [contour], -1, 0, -1)
        return binary_mask

    def process_image(self, image):
        """이미지를 전처리"""
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
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
        except Exception as e:
            logging.error(f"이미지 전처리 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return None

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
                height, width = image.shape[:2]
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

class MeasurementTool(ImageProcessor):
    def __init__(self, config):
        super().__init__(config)
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
            # 배경 차감 적용
            bg_Mask1, bg_Mask2 = self.apply_background_subtraction(image1, image2, background_img)
            if bg_Mask1 is None or bg_Mask2 is None:
                logging.error("배경 차감 결과가 유효하지 않습니다.")
                return None, None, None

            image1_gray, bg_Mask1 = self.image_mask_check(image1, bg_Mask1)
            image2_gray, bg_Mask2 = self.image_mask_check(image2, bg_Mask2)

            p0 = cv2.goodFeaturesToTrack(image1_gray, mask=bg_Mask1, **self.feature_params)

            if p0 is None:
                logging.error("첫 번째 이미지에서 특징점을 찾을 수 없습니다.")
                return None, None, None

            # 두 번째 이미지에서 특징점 추적
            p1, st, err = cv2.calcOpticalFlowPyrLK(image1_gray, image2_gray, p0, None, **self.lk_params)

            # 좋은 특징점 선택
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 10:
                logging.error("충분한 좋은 특징점 매칭을 찾을 수 없습니다.")

            # movement 파라미터가 제공되면 포인트를 보정
            if movement is not None:
                movement_pixels = movement * self.lens_magnification / self.pixel_size_mm
            else:
                movement_pixels = 0

            # 어파인 변환 추정을 위한 데이터 준비
            pts_old = np.float32(good_old).reshape(-1, 2)
            pts_new = np.float32(good_new).reshape(-1, 2)
            pts_new[:, 1] -= movement_pixels

            # ROI 보정 적용
            if self.config["PreProcess"]["ROI"]["Use"]:
                x_offset = self.config["PreProcess"]["ROI"]["Coordinates"].get("x", 0)
                y_offset = self.config["PreProcess"]["ROI"]["Coordinates"].get("y", 0)
                pts_old += np.array([x_offset, y_offset])
                pts_new += np.array([x_offset, y_offset])

            # 어파인 변환 추정 (회전 및 평행 이동 포함)
            M, inliers = cv2.estimateAffinePartial2D(
                pts_old, pts_new, method=cv2.RANSAC,
                ransacReprojThreshold=2.0, maxIters=5000, confidence=0.99
            )

            if M is None or inliers is None or np.sum(inliers) < 10:
                logging.error("어파인 변환 추정에 실패했습니다.")
                logging.error(traceback.format_exc())
                return None, None, None

            # 변환 행렬에서 스케일, 이동량, 회전 각도 추출
            sx = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
            sy = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
            scale = (sx + sy) / 2

            # 줌 배율로 스케일 보정
            if self.config["PreProcess"]["Zoom"]["Use"]:
                scale /= self.config["PreProcess"]["Zoom"]["Zoom_Factor"]

            # 이동량 계산
            tx_pixels = M[0, 2]
            ty_pixels = M[1, 2]
            tx_mm = (tx_pixels / scale) * self.pixel_size_mm / self.lens_magnification
            ty_mm = (ty_pixels / scale) * self.pixel_size_mm / self.lens_magnification

            # 회전 각도 계산
            angle_rad = np.arctan2(M[1, 0], M[0, 0])
            theta_deg = np.degrees(angle_rad)

            # 특징점 좌표를 배열 형태로 변환
            good_old_pts = pts_old[inliers.flatten() == 1]
            good_new_pts = pts_new[inliers.flatten() == 1]
            
            # Optical Flow 시각화
            # 두 이미지를 가로로 합쳐 하나의 이미지로 만듭니다.
            height, width = image1_gray.shape
            combined_img = np.zeros((height, width * 2), dtype=np.uint8)
            combined_img[:, :width] = image1_gray
            combined_img[:, width:] = image2_gray

            # 컬러 이미지로 변환하여 시각화에 사용합니다.
            combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)

            # 두 번째 이미지의 특징점 좌표에 너비만큼 오프셋 추가
            good_new_pts_offset = good_new_pts.copy()
            good_new_pts_offset[:, 0] += width

            # 특징점 이동을 선과 원으로 시각화
            for i in range(len(good_old_pts)):
                old_pt = tuple(good_old_pts[i].astype(int))
                new_pt = tuple(good_new_pts_offset[i].astype(int))
                # 첫 번째 이미지의 특징점에 빨간색 원 그리기
                cv2.circle(combined_img_color, old_pt, 5, (0, 0, 255), -1)
                # 두 번째 이미지의 특징점에 초록색 원 그리기
                cv2.circle(combined_img_color, new_pt, 5, (0, 255, 0), -1)
                # 두 특징점을 연결하는 파란색 선 그리기
                cv2.line(combined_img_color, old_pt, new_pt, (255, 0, 0), 2)

            # 시각화 이미지 저장
            match_img_path = os.path.join(
                self.config["Image_Save_Path"],
                f"{prefix}_optical_flow_{camera_index}.jpg"
            )
            cv2.imwrite(match_img_path, combined_img_color)
            logging.info(f"Optical flow 시각화 이미지 저장: {match_img_path}")

            return tx_mm, ty_mm, theta_deg
        except Exception as e:
            logging.error(f"이동 계산 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return None, None, None

    def calculate_camera_movement(self, depth_map1, depth_map2):
        """두 깊이 맵 사이의 수직 이동 거리를 계산합니다."""
        try:
            # 수직 이동 거리 계산
            diff = depth_map2 - depth_map1

            # 유효한 깊이 값만 사용
            valid_mask = (depth_map1 > 0) & (depth_map2 > 0)
            if not np.any(valid_mask):
                logging.error("유효한 깊이 값이 없습니다.")
                return None

            # 수직 이동 거리 추정 (미디언 사용)
            vertical_movement = np.nanmedian(diff[valid_mask])
            logging.info(f"추정된 수직 이동 거리: {vertical_movement} mm")

            return vertical_movement
        except Exception as e:
            logging.error(f"수직 이동 거리 계산에 실패했습니다: {e}")
            logging.error(traceback.format_exc())
            return None

    def compute_depth_map(self, img_left, img_right):
        """스테레오 매칭을 통해 깊이 맵을 계산합니다."""
        try:
            # 스테레오 매칭 객체 생성
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16*5,  # 80 (must be divisible by 16)
                blockSize=5,
                P1=8*3*5**2,
                P2=32*3*5**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )

            # 그레이스케일 변환
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # 시차 맵 계산
            disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

            # 깊이 맵 계산
            # Q 매트릭스를 사용하여 3D 포인트 클라우드 생성
            Q = self.calibrator.calibration_data.get("Q", None)
            if Q is None:
                logging.error("Q 매트릭스가 캘리브레이션 데이터에 없습니다.")
                return None

            # 재투영하여 3D 포인트 클라우드 생성
            points_3D = cv2.reprojectImageTo3D(disparity, Q)
            depth_map = points_3D[:, :, 2]

            return depth_map
        except Exception as e:
            logging.error(f"깊이 맵 계산에 실패했습니다: {e}")
            logging.error(traceback.format_exc())
            return None

class StereoCalibrator:
    def __init__(self, config):
        self.config = config
        self.calibration_data = {}
        logging.info("StereoCalibrator 생성 완료.")

    def calibrate_cameras(self, images_left, images_right, calibration_file):
        """스테레오 캘리브레이션을 수행하고 결과를 저장합니다."""
        try:
            # 체스보드 패턴 크기 및 사각형 크기 설정
            pattern_size = (self.config["Calibration"]["Pattern_Columns"],
                            self.config["Calibration"]["Pattern_Rows"])
            square_size = self.config["Calibration"]["Square_Size"]

            # 객체 포인트 생성
            objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0],
                                   0:pattern_size[1]].T.reshape(-1, 2)
            objp *= square_size

            objpoints = []  # 3D 점들
            imgpoints_left = []  # 왼쪽 이미지의 2D 점들
            imgpoints_right = []  # 오른쪽 이미지의 2D 점들

            # 각 이미지에 대해 코너 찾기
            for img_left, img_right in zip(images_left, images_right):
                gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

                ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
                ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

                if ret_left and ret_right:
                    objpoints.append(objp)
                    imgpoints_left.append(corners_left)
                    imgpoints_right.append(corners_right)
                else:
                    logging.warning("체스보드 코너를 찾을 수 없습니다.")

            if not objpoints:
                logging.error("체스보드 코너를 충분히 찾지 못했습니다.")
                return False

            # 개별 카메라 캘리브레이션
            ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
            ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
                objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

            # 스테레오 캘리브레이션
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                        cv2.TERM_CRITERIA_EPS, 100, 1e-5)

            ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], criteria=criteria, flags=flags)

            # 스테레오 정렬을 위한 Q 매트릭스 계산
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T, alpha=0)

            self.calibration_data = {
                "mtx_left": mtx_left,
                "dist_left": dist_left,
                "mtx_right": mtx_right,
                "dist_right": dist_right,
                "R": R,
                "T": T,
                "E": E,
                "F": F,
                "R1": R1,
                "R2": R2,
                "P1": P1,
                "P2": P2,
                "Q": Q
            }

            # 캘리브레이션 결과 저장
            os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
            np.savez(calibration_file, **self.calibration_data)
            logging.info(f"스테레오 캘리브레이션 완료. 데이터 저장: {calibration_file}")

            return True
        except Exception as e:
            logging.error(f"스테레오 캘리브레이션에 실패했습니다: {e}")
            logging.error(traceback.format_exc())
            return False

    def load_calibration_data(self, calibration_file):
        """캘리브레이션 데이터 파일을 로드합니다."""
        try:
            data = np.load(calibration_file)
            self.calibration_data = {key: data[key] for key in data.files}
            logging.info(f"캘리브레이션 데이터 로드 완료: {calibration_file}")
            return True
        except FileNotFoundError:
            logging.error(f"캘리브레이션 데이터 파일을 찾을 수 없습니다: {calibration_file}")
        except Exception as e:
            logging.error(f"캘리브레이션 데이터 로드에 실패했습니다: {e}")
            logging.error(traceback.format_exc())
        return False

    def rectify_images(self, img_left, img_right):
        """스테레오 이미지를 정렬합니다."""
        try:
            if not self.calibration_data:
                logging.error("캘리브레이션 데이터가 없습니다.")
                return None, None

            mtx_left = self.calibration_data["mtx_left"]
            dist_left = self.calibration_data["dist_left"]
            mtx_right = self.calibration_data["mtx_right"]
            dist_right = self.calibration_data["dist_right"]
            R1 = self.calibration_data["R1"]
            R2 = self.calibration_data["R2"]
            P1 = self.calibration_data["P1"]
            P2 = self.calibration_data["P2"]
            Q = self.calibration_data["Q"]

            image_size = img_left.shape[:2][::-1]

            # 리매핑 매트릭스 계산
            map1_left, map2_left = cv2.initUndistortRectifyMap(
                mtx_left, dist_left, R1, P1, image_size, cv2.CV_16SC2)
            map1_right, map2_right = cv2.initUndistortRectifyMap(
                mtx_right, dist_right, R2, P2, image_size, cv2.CV_16SC2)

            # 이미지 리매핑
            rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

            return rectified_left, rectified_right
        except Exception as e:
            logging.error(f"이미지 정렬에 실패했습니다: {e}")
            logging.error(traceback.format_exc())
            return None, None

class StereoPair:
    def __init__(self, left_camera_num, right_camera_num, config, stereo_id):
        self.left_camera_num = left_camera_num
        self.right_camera_num = right_camera_num
        self.config = config
        self.stereo_id = stereo_id
        self.picam2_left = None
        self.picam2_right = None
        self.calibrator = StereoCalibrator(config)
        self.measurement_tool = MeasurementTool(config)
        self.measurement_tool.calibrator = self.calibrator
        self.setup_cameras()
        logging.info(f"StereoPair {self.stereo_id} 생성 완료.")

    def setup_cameras(self):
        """왼쪽 및 오른쪽 카메라를 초기화하고 구성합니다."""
        try:
            # 왼쪽 카메라 설정
            self.picam2_left = Picamera2(camera_num=self.left_camera_num)
            config_left = self.picam2_left.create_still_configuration()
            self.picam2_left.configure(config_left)
            self.picam2_left.start()
            logging.info(f"StereoPair {self.stereo_id} - 왼쪽 카메라 시작 완료.")

            # 오른쪽 카메라 설정
            self.picam2_right = Picamera2(camera_num=self.right_camera_num)
            config_right = self.picam2_right.create_still_configuration()
            self.picam2_right.configure(config_right)
            self.picam2_right.start()
            logging.info(f"StereoPair {self.stereo_id} - 오른쪽 카메라 시작 완료.")
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 카메라 설정 중 오류 발생: {e}")
            logging.error(traceback.format_exc())

    def capture_images_threading(self, prefix):
        """왼쪽 및 오른쪽 카메라에서 이미지를 스레드를 사용해 캡처합니다."""
        try:
            img_left_path = os.path.join(self.config["Image_Save_Path"], f"{prefix}_stereo{self.stereo_id}_left.jpg")
            img_right_path = os.path.join(self.config["Image_Save_Path"], f"{prefix}_stereo{self.stereo_id}_right.jpg")

            # 재시도 로직 추가 (최대 3회 시도)
            def capture_with_retry(picam2, filename, side, retries=3):
                for attempt in range(retries):
                    try:
                        time.sleep(0.5)  # 카메라 안정화 시간
                        picam2.capture_file(filename)
                        logging.info(f"{side} 카메라 - 이미지 저장 완료: {filename}")
                        return True
                    except Exception as e:
                        logging.error(f"{side} 카메라 - 이미지 저장 실패 (시도 {attempt+1}/{retries}): {e}")
                        time.sleep(1)  # 잠시 대기 후 재시도
                logging.error(f"{side} 카메라 - 모든 재시도 실패.")
                return False

            # 스레드를 사용해 동시 캡처
            thread_left = threading.Thread(target=capture_with_retry, args=(self.picam2_left, img_left_path, 'left'))
            thread_right = threading.Thread(target=capture_with_retry, args=(self.picam2_right, img_right_path, 'right'))
            thread_left.start()
            thread_right.start()
            thread_left.join()
            thread_right.join()

            logging.info(f"StereoPair {self.stereo_id} - 이미지 캡처 완료: {img_left_path}, {img_right_path}")
            
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_path)

            if img_left is not None and img_right is not None:
                return {'left': img_left, 'right': img_right}
            else:
                logging.error("이미지 로드 실패.")
                return None
        except Exception as e:
            logging.error(f"이미지 캡처 오류: {e}")
            logging.error(traceback.format_exc())
            return None

    def calibrate(self, images_left, images_right):
        """스테레오 페어 캘리브레이션을 수행합니다."""
        try:
            calibration_file = os.path.join(self.config["Calibration_Data_Path"], f"stereo_calibration_data_stereo{self.stereo_id}.npz")
            success = self.calibrator.calibrate_cameras(images_left, images_right, calibration_file)
            if success:
                logging.info(f"StereoPair {self.stereo_id} - 캘리브레이션 성공.")
            else:
                logging.error(f"StereoPair {self.stereo_id} - 캘리브레이션 실패.")
            return success
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 캘리브레이션 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return False

    def load_calibration_data(self):
        """스테레오 캘리브레이션 데이터를 로드합니다."""
        try:
            calibration_file = os.path.join(self.config["Calibration_Data_Path"], f"stereo_calibration_data_stereo{self.stereo_id}.npz")
            success = self.calibrator.load_calibration_data(calibration_file)
            if success:
                logging.info(f"StereoPair {self.stereo_id} - 캘리브레이션 데이터 로드 완료.")
            else:
                logging.error(f"StereoPair {self.stereo_id} - 캘리브레이션 데이터 로드 실패.")
            return success
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 캘리브레이션 데이터 로드 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return False

    def rectify(self, img_left, img_right):
        """스테레오 이미지를 정렬합니다."""
        try:
            rectified_left, rectified_right = self.calibrator.rectify_images(img_left, img_right)
            return rectified_left, rectified_right
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 이미지 정렬 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return None, None

    def compute_depth(self, rectified_left, rectified_right):
        """정렬된 이미지를 기반으로 깊이 맵을 계산합니다."""
        try:
            depth_map = self.measurement_tool.compute_depth_map(rectified_left, rectified_right)
            return depth_map
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 깊이 맵 계산 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return None

    def calculate_movement(self, depth_map1, depth_map2):
        """두 깊이 맵 사이의 수직 이동 거리를 계산합니다."""
        try:
            vertical_movement = self.measurement_tool.calculate_camera_movement(depth_map1, depth_map2)
            return vertical_movement
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 수직 이동 거리 계산 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            return None

    def close(self):
        """왼쪽 및 오른쪽 카메라를 정지하고 종료합니다."""
        try:
            if self.picam2_left:
                self.picam2_left.stop()
            if self.picam2_right:
                self.picam2_right.stop()
            logging.info(f"StereoPair {self.stereo_id} - 카메라 종료 완료.")
        except Exception as e:
            logging.error(f"StereoPair {self.stereo_id} - 카메라 종료 중 오류 발생: {e}")

def main():
    try:
        json_path = "/home/hims/TestCam/etc/config2.json"
        config = load_config(json_path)
        if not config:
            logging.error("설정 파일을 로드하지 못했습니다. 프로그램을 종료합니다.")
            return

        folder_path = setup_folders(os.path.expanduser(config["Path"]))
        if not folder_path:
            logging.error("폴더 설정에 실패했습니다. 프로그램을 종료합니다.")
            return

        config["Image_Save_Path"] = folder_path
        logging.info("설정 값: %s", json.dumps(config, indent=4, ensure_ascii=False))

        # 스테레오 페어 정의 (예: 0-1, 2-3)
        camera_indices = config.get("Camera_Indices", [])
        if len(camera_indices) < 2:
            logging.error("최소 2개의 카메라 인덱스가 필요합니다.")
            return

        # 스테레오 페어 정의
        num_pairs = len(camera_indices) // 2
        stereo_pairs = []
        for i in range(num_pairs):
            left = camera_indices[2*i]
            right = camera_indices[2*i + 1]
            stereo_pairs.append({"left": left, "right": right, "id": i+1})

        # 스테레오 페어 객체 생성
        stereo_objects = []
        for pair in stereo_pairs:
            stereo = StereoPair(pair["left"], pair["right"], config, pair["id"])
            stereo_objects.append(stereo)

        # 캘리브레이션 수행 또는 기존 데이터 로드
        for stereo in stereo_objects:
            if config.get("Perform_Calibration", False):
                # 캘리브레이션 이미지 캡처
                num_calibration_images = config.get("Calibration", {}).get("Num_Images", 10)
                images_left = []
                images_right = []
                for i in range(num_calibration_images):
                    input(f"StereoPair {stereo.stereo_id}: 캘리브레이션을 위해 체스보드 패턴을 배치하고 Enter 키를 누르세요 ({i+1}/{num_calibration_images})...")
                    images = stereo.capture_images_threading(f"calibration_{i+1}")
                    if images:
                        images_left.append(images['left'])
                        images_right.append(images['right'])
                    else:
                        logging.error(f"StereoPair {stereo.stereo_id}: 캘리브레이션 이미지 캡처 실패.")
                        break

                if len(images_left) >= 2:  # 최소 두 이미지 필요
                    success = stereo.calibrate(images_left, images_right)
                    if not success:
                        logging.error(f"StereoPair {stereo.stereo_id}: 캘리브레이션 실패.")
                else:
                    logging.error(f"StereoPair {stereo.stereo_id}: 충분한 캘리브레이션 이미지를 캡처하지 못했습니다.")
            else:
                # 기존 캘리브레이션 데이터 로드
                success = stereo.load_calibration_data()
                if not success:
                    logging.error(f"StereoPair {stereo.stereo_id}: 캘리브레이션 데이터 로드 실패. 캘리브레이션을 수행하세요.")
                    continue

        # 측정 수행
        for stereo in stereo_objects:
            if not config.get("Perform_Calibration", False) and not stereo.calibrator.calibration_data:
                logging.error(f"StereoPair {stereo.stereo_id}: 캘리브레이션 데이터가 없어 측정을 수행할 수 없습니다.")
                continue

            # 첫 번째 이미지 캡처
            input(f"StereoPair {stereo.stereo_id}: 첫 번째 이미지를 캡처하려면 Enter 키를 누르세요...")
            images1 = stereo.capture_images_threading("first")
            if not images1:
                continue

            # 두 번째 이미지 캡처
            input(f"StereoPair {stereo.stereo_id}: 두 번째 이미지를 캡처하려면 Enter 키를 누르세요...")
            images2 = stereo.capture_images_threading("second")
            if not images2:
                continue

            # 이미지 정렬
            rectified_left1, rectified_right1 = stereo.rectify(images1['left'], images1['right'])
            rectified_left2, rectified_right2 = stereo.rectify(images2['left'], images2['right'])

            if rectified_left1 is None or rectified_right1 is None or rectified_left2 is None or rectified_right2 is None:
                logging.error(f"StereoPair {stereo.stereo_id}: 이미지 정렬에 실패했습니다.")
                continue

            # 깊이 맵 계산
            depth_map1 = stereo.compute_depth(rectified_left1, rectified_right1)
            depth_map2 = stereo.compute_depth(rectified_left2, rectified_right2)

            if depth_map1 is None or depth_map2 is None:
                logging.error(f"StereoPair {stereo.stereo_id}: 깊이 맵 계산에 실패했습니다.")
                continue

            # 수직 이동 거리 계산
            vertical_movement = stereo.calculate_movement(depth_map1, depth_map2)
            if vertical_movement is not None:
                logging.info(f"StereoPair {stereo.stereo_id}: 수직 이동 거리: {vertical_movement:.6f} mm")
            else:
                logging.error(f"StereoPair {stereo.stereo_id}: 수직 이동 거리를 계산할 수 없습니다.")

        # 모든 스테레오 페어 종료
        for stereo in stereo_objects:
            stereo.close()

    except Exception as e:
        logging.error(f"오류 발생: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
