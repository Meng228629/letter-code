from ultralytics import YOLO
 
def main():
    model = YOLO('model/yolov8m-obb.yaml').load('model/yolov8m-obb.pt')  # build from YAML and transfer weights
    model.train(data='datasets/data.yaml', epochs=400, imgsz=1280, batch=16, workers=8)
if __name__ == '__main__':
    main()