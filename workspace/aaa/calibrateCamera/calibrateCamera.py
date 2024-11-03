import cv2
import numpy as np
import glob
import os
# 체커보드 패턴의 내부 코너 수
CHECKERBOARD = (9, 6)

# 3D 점과 2D 점 저장을 위한 배열
objpoints = []
imgpoints = []

# 체커보드 패턴의 3D 점 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 이미지 목록 가져오기
images = glob.glob('/home/hims/TestCam/calibrateCamera/*.jpg')
gray = None  # 초기 gray 변수를 None으로 설정

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"코너를 찾았습니다: {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # 코너 시각화
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        img_fname = os.path.basename(fname)
        cv2.imwrite(f"/home/hims/TestCam/calibrateCamera/lens_x_af_x/calibrateCamera_img/{img_fname}.jpg",img)
    else:
        print(f"코너를 찾지 못했습니다: {fname}")

cv2.destroyAllWindows()

# 유효한 objpoints와 imgpoints가 있는지 확인
if objpoints and imgpoints and gray is not None:
    # 카메라 매트릭스와 왜곡 계수 계산
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 계산된 값 출력
    print("카메라 매트릭스 (mtx):")
    print(mtx)
    print("\n왜곡 계수 (dist):")
    print(dist)
    print("\n회전 벡터 (rvecs):")
    print(rvecs)
    print("\n이동 벡터 (tvecs):")
    print(tvecs)
    
    output_path = "/home/hims/TestCam/calibrateCamera/lens_x_af_x/calibration_results.txt"
    with open(output_path, "w") as f:
        f.write("카메라 매트릭스 (mtx):\n")
        f.write(f"{mtx}\n\n")
        
        f.write("왜곡 계수 (dist):\n")
        f.write(f"{dist}\n\n")
        
        f.write("회전 벡터 (rvecs):\n")
        for rvec in rvecs:
            f.write(f"{rvec}\n")
        f.write("\n")
        
        f.write("이동 벡터 (tvecs):\n")
        for tvec in tvecs:
            f.write(f"{tvec}\n")
else:
    print("유효한 체커보드 코너를 찾지 못해 캘리브레이션을 수행할 수 없습니다.")
