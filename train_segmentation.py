from ultralytics import YOLO

def train_model():
    # 1. Инициализация модели
    # yolov8n-seg - нано (быстрая), yolov8x-seg - экстра большая (точная)
    # Для спутников лучше брать M или L, если позволяет видеокарта, т.к. детали мелкие
    model = YOLO('yolov9c-seg.pt')  

    # 2. Запуск обучения
    results = model.train(
        data='./datasets/satellite_seg/data.yaml', # Путь к yaml из прошлого скрипта
        epochs=100,             # Кол-во эпох
        imgsz=640,              # Размер входа (должен совпадать с нарезкой)
        batch=16,               # Подберите под память GPU
        # device=0,               # GPU ID (или 'cpu')
        patience=20,            # Ранняя остановка
        # workers=8,              # Потоки загрузки данных
        
        # --- Аугментации (критически важны для спутников) ---
        degrees=90,             # Повороты (у спутника нет "верха")
        flipud=0.5,             # Вертикальное отражение
        fliplr=0.5,             # Горизонтальное отражение
        mosaic=1.0,             # Mosaic помогает учить мелкие объекты
        mixup=0.1,              # Немного mixup для робастности
        copy_paste=0.1,         # Полезно для сегментации
        
        project='satellite_project',
        name='run_v1',
        exist_ok=True
    )

    # 3. Валидация
    metrics = model.val()
    print(f"mAP50-95 (Seg): {metrics.seg.map:.3f}")

    # 4. Пример инференса на полном снимке (опционально - логика тайлинга нужна и тут)
    # model.predict('test_image.tif', save=True, imgsz=640)

if __name__ == '__main__':
    train_model()