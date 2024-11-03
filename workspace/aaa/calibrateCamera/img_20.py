import os
import time
import cv2
from picamera2 import Picamera2
from libcamera import controls
from smbus2 import SMBus

"""
기본 설정의 카메라로 20개의 이미지를 캡쳐합니다.
20개의 이미지는 calibrateCamera 폴더에 저장됩니다.
"""
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
        print(f"유효하지 않은 채널입니다: {channel}")
        return
    try:
        with SMBus(10) as bus:  # SMBus 10번 버스 사용
            channel_value = channels[channel]
            bus.write_byte_data(I2C_MUX_ADDRESS, CONTROL_REGISTER, channel_value)
            print(f"I2C 채널 {channel} 설정 성공.")
    except Exception as e:
        print(f"I2C 채널 {channel} 설정에 실패했습니다: {e}")
        
def setup_camera():
    """카메라를 초기화하고 기본 설정을 구성합니다."""
    picam2 = Picamera2()
    configuration = picam2.create_still_configuration()
    picam2.configure(configuration)
    controls_dict = {
        "AfMode": controls.AfModeEnum.Manual,
    }
    picam2.set_controls(controls_dict)
    time.sleep(1)
    return picam2

def capture_images(picam2, save_path, num_images=20):
    """카메라로 이미지를 캡처하고 저장합니다."""
    for i in range(num_images):
        input(f"{i+1}/{num_images}번째 이미지를 캡처하려면 Enter 키를 누르세요...")
        picam2.start()
        image = picam2.capture_array()
        picam2.stop()

        if image is None:
            print(f"{i+1}번째 이미지 캡처에 실패했습니다.")
            continue

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_path = os.path.join(save_path, f"image_{i+1}.jpg")
        cv2.imwrite(img_path, image)
        print(f"이미지 저장: {img_path}")

def main():
    """메인 실행 함수"""
    save_path = os.path.expanduser("/home/hims/TestCam/calibrateCamera")
    os.makedirs(save_path, exist_ok=True)
    set_i2c_channel(0)
    picam2 = setup_camera()
    capture_images(picam2, save_path)
    picam2.close()

if __name__ == "__main__":
    main()