import cv2
import os

root_path = "/media/ms-neo2/ms-ssd11/1.dataset/VFI/adobe240fps"

# 비디오 폴더 경로
video_dir = os.path.join(root_path, './videos')

# 이미지 저장 폴더 경로
image_dir = os.path.join(root_path, './images')

# 비디오 파일 확장자 목록
video_extensions = ['.mp4', '.m4v', '.MOV', '.mov']

for filename in os.listdir(video_dir):
    # 비디오 파일인지 확인
    ext = os.path.splitext(filename)[1].lower()
    if ext in video_extensions:
        video_path = os.path.join(video_dir, filename)

        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)

        # 비디오 파일 이름에서 확장자 제거
        base_name = os.path.splitext(filename)[0]

        # 이미지 저장 폴더 생성
        image_folder = os.path.join(image_dir, base_name)
        os.makedirs(image_folder, exist_ok=True)

        frame_num = 0
        while True:
            # 프레임 읽기
            ret, frame = cap.read()

            if not ret:
                # 더 이상 프레임이 없으면 종료
                break

            # 프레임 저장
            image_path = os.path.join(image_folder, f'{frame_num:05d}.png')
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            frame_num += 1

        # 캡처 객체 해제
        cap.release()

        print(f'{filename} 처리 완료')
