from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict(
        source="/path/to/images",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        save=True,
        show_labels=True,
        show_conf=True,
        visualize=False,
    )
