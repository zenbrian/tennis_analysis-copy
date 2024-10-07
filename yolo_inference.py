from ultralytics import YOLO 

model = YOLO('yolov8x')

result = model.predict('input_videos/input_video.mp4',conf=0.2, save=True)
#哈哈屁眼
#大屁眼
#我是edison
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)