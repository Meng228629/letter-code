from ultralytics import YOLO

def main():
    model = YOLO(r'runs/obb/train11/weights/best.pt')
    model.val(data='datasets/data.yaml', imgsz=1280, batch=16, workers=8)
if __name__ == '__main__':
    main()