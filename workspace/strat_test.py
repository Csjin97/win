

import os
import json
import time
import logging
import glob
import cv2

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









import glob

def main():
    json_file = "C:\\Users\\310tk\\Desktop\\workspace\\etc\\config.json"
    config = load_config(json_file)
    if not config:
        return

    folder_path = setup_folders(os.path.expanduser(config["Path"]))
    if not folder_path:
        return

    config["Image_Save_Path"] = folder_path
    logging.info("설정 값: %s", json.dumps(config, indent=4, ensure_ascii=False))
    
    num_cameras = config["Num_Cameras"]
    
    # 지정된 경로에서 background 및 first 및 second 이미지 읽기
    background_image = []
    images1 = []
    images2 = []
    background_path = "C:\Users\310tk\Desktop\workspace\lens_x_af_x\background_img"
    img_path = "C:\Users\310tk\Desktop\workspace\lens_x_af_x\0"

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
    
    # images1과 images2 리스트에는 각 카메라에 이미지가 저장됨
    logging.info(f"{len(background_image)} background images loaded.")
    logging.info(f"{len(images1)} first images and {len(images2)} second images loaded.")



    logging.info("프로그램 종료.")

if __name__ == "__main__":
    main()
