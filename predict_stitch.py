import os
import cv2
import json
import torch
import rasterio
import rasterio.features
import numpy as np
import pandas as pd
import geopandas as gpd
from ultralytics import YOLO
from rasterio.windows import Window
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm
from pathlib import Path

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    'image_path': './raw_data/test_image.tif',  # Путь к снимку для прогноза
    'model_path': './runs/segment/run_v1/weights/best.pt', # Путь к обученной модели
    'output_dir': './inference_results',
    'tile_size': 640,
    'overlap': 0.2,          # Перекрытие тайлов при прогнозе
    'conf_thres': 0.4,       # Порог уверенности
    'iou_thres': 0.5,        # Порог IoU для удаления дубликатов при сшивке
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def pixel_to_map(coords, window, transform):
    """
    Переводит локальные пиксели тайла (x, y) в глобальные координаты карты.
    coords: список точек [[x,y], [x,y]...]
    window: текущее окно чтения Rasterio
    transform: аффинная трансформация всего растра
    """
    global_coords = []
    # Смещение окна
    off_x, off_y = window.col_off, window.row_off
    
    for x, y in coords:
        # Глобальный пиксель
        gx = off_x + x
        gy = off_y + y
        # Перевод в map coords (например, метры или градусы)
        # transform * (col, row) -> (x_geo, y_geo)
        mx, my = transform * (gx, gy)
        global_coords.append((mx, my))
    return global_coords

def geometric_nms(gdf, iou_threshold=0.5):
    """
    Удаляет дубликаты геометрий (Non-Maximum Suppression).
    Если два полигона пересекаются сильнее, чем iou_threshold, оставляем тот, у которого выше score.
    """
    if gdf.empty:
        return gdf

    # Сортируем по уверенности (сначала самые уверенные)
    gdf = gdf.sort_values(by='conf', ascending=False).reset_index(drop=True)
    keep_indices = []
    
    # Создаем пространственный индекс для ускорения
    sindex = gdf.sindex
    
    active_indices = set(gdf.index)
    
    pbar = tqdm(total=len(gdf), desc="Фильтрация дубликатов (NMS)")
    
    while active_indices:
        curr_idx = min(active_indices) # Берем самый уверенный из оставшихся
        keep_indices.append(curr_idx)
        active_indices.remove(curr_idx)
        pbar.update(1)
        
        curr_geom = gdf.geometry.iloc[curr_idx]
        
        # Ищем кандидатов на пересечение через R-tree индекс
        possible_matches_idx = list(sindex.intersection(curr_geom.bounds))
        
        for match_idx in possible_matches_idx:
            if match_idx not in active_indices:
                continue
            
            match_geom = gdf.geometry.iloc[match_idx]
            
            # Считаем IoU
            intersection = curr_geom.intersection(match_geom).area
            union = curr_geom.union(match_geom).area
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                active_indices.remove(match_idx)
                pbar.update(1)
                
    pbar.close()
    return gdf.iloc[keep_indices]

def run_inference(cfg):
    save_path = Path(cfg['output_dir'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка модели: {cfg['model_path']}")
    model = YOLO(cfg['model_path'])
    
    all_detections = [] # Список словарей для DataFrame
    
    print(f"Открытие изображения: {cfg['image_path']}")
    with rasterio.open(cfg['image_path']) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        
        # Параметры скользящего окна
        stride = int(cfg['tile_size'] * (1 - cfg['overlap']))
        
        # --- 1. ПРОГНОЗ ПО ТАЙЛАМ ---
        total_tiles = ((height // stride) + 1) * ((width // stride) + 1)
        pbar = tqdm(total=total_tiles, desc="Прогноз тайлов")
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Определяем окно
                window = Window(x, y, cfg['tile_size'], cfg['tile_size'])
                
                # Корректируем чтение на краях (чтобы не выйти за пределы)
                # Важно: YOLO требует фиксированный размер, поэтому если край меньше, 
                # мы читаем чуть назад или паддим.
                # Для простоты: Rasterio позволяет читать за пределами (вернет 0),
                # но лучше ограничить чтение реальными данными.
                actual_w = min(cfg['tile_size'], width - x)
                actual_h = min(cfg['tile_size'], height - y)
                read_window = Window(x, y, actual_w, actual_h)
                
                img = src.read([1, 2, 3], window=read_window)
                
                # Если тайл меньше 640x640, паддим его нулями до 640x640
                if img.shape[1] != cfg['tile_size'] or img.shape[2] != cfg['tile_size']:
                    padded_img = np.zeros((3, cfg['tile_size'], cfg['tile_size']), dtype=img.dtype)
                    padded_img[:, :actual_h, :actual_w] = img
                    img = padded_img
                
                # (C, H, W) -> (H, W, C) -> BGR
                img = np.moveaxis(img, 0, -1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # INFERENCE
                results = model.predict(img, conf=cfg['conf_thres'], verbose=False, device=cfg['device'])
                
                if results[0].masks is not None:
                    masks = results[0].masks.xy # Координаты полигонов (список массивов)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for i, poly_coords in enumerate(masks):
                        if len(poly_coords) < 3: continue # Игнорируем точки/линии
                        
                        # Перевод локальных координат в глобальные
                        global_poly = pixel_to_map(poly_coords, window, transform)
                        shapely_poly = Polygon(global_poly)
                        
                        # Проверка на валидность геометрии
                        if not shapely_poly.is_valid:
                            shapely_poly = shapely_poly.buffer(0)
                        
                        all_detections.append({
                            'geometry': shapely_poly,
                            'conf': float(confs[i]),
                            'class': int(classes[i]),
                            'bbox_str': str(list(boxes[i])) # Для CSV
                        })
                
                pbar.update(1)
        pbar.close()

    # --- 2. ПОСТОБРАБОТКА И СОХРАНЕНИЕ ---
    if not all_detections:
        print("Объекты не найдены.")
        return

    print("Создание GeoDataFrame...")
    gdf = gpd.GeoDataFrame(all_detections, crs=crs)
    
    print(f"Найдено сырых объектов: {len(gdf)}")
    # Удаление дубликатов (NMS)
    clean_gdf = geometric_nms(gdf, iou_threshold=cfg['iou_thres'])
    print(f"Объектов после очистки: {len(clean_gdf)}")

    # 1. Сохранение ВЕКТОРОВ (Shapefile / GeoJSON)
    vector_out = save_path / 'predictions.shp'
    # Shapefile ограничивает длину имен полей, bbox_str может обрезаться
    clean_gdf.drop(columns=['bbox_str']).to_file(vector_out, encoding='utf-8')
    print(f"Вектора сохранены: {vector_out}")
    
    # 2. Сохранение ТЕКСТОВЫХ ДАННЫХ (CSV)
    csv_out = save_path / 'predictions_metadata.csv'
    # Добавляем WKT (Well Known Text) представление геометрии
    df_export = pd.DataFrame(clean_gdf.drop(columns='geometry'))
    df_export['wkt'] = clean_gdf.geometry.apply(lambda x: x.wkt)
    df_export.to_csv(csv_out, index=False)
    print(f"Метаданные сохранены: {csv_out}")

    # 3. Сохранение РАСТРА С РАЗМЕТКОЙ (GeoTIFF Mask)
    # Создаем растр, где значение пикселя = ID класса (или 255 для маски)
    print("Генерация растровой маски...")
    mask_out = save_path / 'predictions_mask.tif'
    
    # Обновляем профиль: один канал, uint8, сжатие
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
    
    with rasterio.open(mask_out, 'w', **profile) as dst:
        # Растеризуем геометрии. 
        # shapes принимает пары (geometry, value)
        shapes = ((geom, 255) for geom in clean_gdf.geometry)
        
        # rasterize прожигает геометрии в массив размера изображения
        # ВАЖНО: transform должен быть от исходного растра
        burned = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )
        dst.write(burned, 1)
        
    print(f"Растровая маска сохранена: {mask_out}")
    
    # 4. (Опционально) Визуализация превью (JPG) с bbox
    # Генерируем уменьшенную копию для быстрого просмотра, так как оригинал огромный
    print("Генерация превью (Visual JPG)...")
    preview_scale = 0.1 # 10% от размера
    small_w, small_h = int(width * preview_scale), int(height * preview_scale)
    
    with rasterio.open(cfg['image_path']) as src:
        # Читаем уменьшенную версию всего снимка
        img_small = src.read([1, 2, 3], out_shape=(3, small_h, small_w))
        img_small = np.moveaxis(img_small, 0, -1)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR) # BGR for cv2
        
        # Рисуем контуры
        # Нам нужно пересчитать глобальные координаты в координаты маленькой картинки
        # Формула: (MapCoord - TopLeftMap) / PixelSizeMap * Scale
        min_x, max_y = transform.c, transform.f
        pixel_w, pixel_h = transform.a, transform.e # pixel_h обычно отрицательный
        
        for idx, row in clean_gdf.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            
            # Получаем внешние координаты
            if geom.geom_type == 'Polygon':
                parts = [geom]
            elif geom.geom_type == 'MultiPolygon':
                parts = geom.geoms
            else:
                parts = []
                
            for poly in parts:
                pts = np.array(poly.exterior.coords)
                # Map -> Pixel (Full Res) -> Pixel (Small Res)
                pts[:, 0] = (pts[:, 0] - min_x) / pixel_w * preview_scale
                pts[:, 1] = (pts[:, 1] - max_y) / pixel_h * preview_scale
                
                pts = pts.astype(np.int32)
                cv2.polylines(img_small, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imwrite(str(save_path / 'visualization_preview.jpg'), img_small)
        
    print("Готово!")

if __name__ == "__main__":
    run_inference(CONFIG)