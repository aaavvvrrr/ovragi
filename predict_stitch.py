import os
import cv2
import torch
import rasterio
import rasterio.features
import numpy as np
import pandas as pd
import geopandas as gpd
from ultralytics import YOLO
from rasterio.windows import Window
from shapely.geometry import Polygon
from tqdm import tqdm
from pathlib import Path

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    # Пути к файлам
    'image_path': './raw_data/test_image.tif',       # Входной снимок
    'model_path': './runs/segment/run_v1/weights/best.pt', # Обученная модель
    'output_dir': './inference_results',             # Куда сохранять
    
    # Параметры инференса
    'tile_size': 640,
    'overlap': 0.20,          # 20% перекрытия
    'conf_thres': 0.40,       # Порог уверенности (отсекать всё, что ниже 40%)
    'iou_thres': 0.50,        # Порог IoU для удаления дубликатов (NMS)
    
    # Железо
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def pixel_to_map(coords, window, transform):
    """
    Переводит локальные пиксели тайла (x, y) в глобальные координаты карты.
    """
    global_coords = []
    off_x, off_y = window.col_off, window.row_off
    
    for x, y in coords:
        # Глобальный пиксель
        gx = off_x + x
        gy = off_y + y
        # Перевод в координаты карты
        mx, my = transform * (gx, gy)
        global_coords.append((mx, my))
    return global_coords

def geometric_nms(gdf, iou_threshold=0.5):
    """
    Удаляет дубликаты геометрий, возникающие из-за перекрытия тайлов.
    Использует пространственный индекс для ускорения.
    """
    if gdf.empty:
        return gdf

    print("Запуск NMS (удаление дубликатов)...")
    # Сортируем по уверенности (сначала самые уверенные)
    gdf = gdf.sort_values(by='conf', ascending=False).reset_index(drop=True)
    keep_indices = []
    
    sindex = gdf.sindex
    active_indices = set(gdf.index)
    
    pbar = tqdm(total=len(gdf), desc="NMS Processing")
    
    while active_indices:
        curr_idx = min(active_indices) # Берем самый уверенный из оставшихся
        keep_indices.append(curr_idx)
        active_indices.remove(curr_idx)
        pbar.update(1)
        
        curr_geom = gdf.geometry.iloc[curr_idx]
        
        # Быстрый поиск пересечений через R-tree
        possible_matches_idx = list(sindex.intersection(curr_geom.bounds))
        
        for match_idx in possible_matches_idx:
            if match_idx not in active_indices:
                continue
            
            match_geom = gdf.geometry.iloc[match_idx]
            
            # Точный расчет IoU
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
    
    print(f"Загрузка модели: {cfg['model_path']} на устройстве {cfg['device']}")
    model = YOLO(cfg['model_path'])
    
    all_detections = []
    
    print(f"Обработка изображения: {cfg['image_path']}")
    with rasterio.open(cfg['image_path']) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        
        stride = int(cfg['tile_size'] * (1 - cfg['overlap']))
        
        # --- ЭТАП 1: Нарезка и Предсказание ---
        # Вычисляем сетку тайлов
        y_steps = range(0, height, stride)
        x_steps = range(0, width, stride)
        total_tiles = len(y_steps) * len(x_steps)
        
        pbar = tqdm(total=total_tiles, desc="Inference")
        
        for y in y_steps:
            for x in x_steps:
                # Читаем реальный кусок данных
                actual_w = min(cfg['tile_size'], width - x)
                actual_h = min(cfg['tile_size'], height - y)
                window = Window(x, y, actual_w, actual_h)
                
                img = src.read([1, 2, 3], window=window)
                
                # Паддинг до квадратного размера 640x640 (если мы на краю)
                # YOLO ожидает фиксированный размер для корректной работы
                if img.shape[1] != cfg['tile_size'] or img.shape[2] != cfg['tile_size']:
                    padded_img = np.zeros((3, cfg['tile_size'], cfg['tile_size']), dtype=img.dtype)
                    padded_img[:, :actual_h, :actual_w] = img
                    img = padded_img
                
                # Подготовка для OpenCV/YOLO (C,H,W -> H,W,C -> BGR)
                img = np.moveaxis(img, 0, -1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Прогноз
                results = model.predict(img, conf=cfg['conf_thres'], verbose=False, device=cfg['device'])
                
                if results[0].masks is not None:
                    masks = results[0].masks.xy
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    cls_ids = results[0].boxes.cls.cpu().numpy()
                    
                    for i, poly_coords in enumerate(masks):
                        if len(poly_coords) < 3: continue
                        
                        # Перевод в глобальные координаты
                        global_poly_coords = pixel_to_map(poly_coords, window, transform)
                        shapely_poly = Polygon(global_poly_coords)
                        
                        # Исправление самопересечений, если есть
                        if not shapely_poly.is_valid:
                            shapely_poly = shapely_poly.buffer(0)
                            
                        all_detections.append({
                            'geometry': shapely_poly,
                            'conf': float(confs[i]),
                            'class_id': int(cls_ids[i]),
                            'bbox': list(boxes[i]) # Локальный bbox, для справки
                        })
                pbar.update(1)
        pbar.close()

    # --- ЭТАП 2: Постобработка ---
    if not all_detections:
        print("Объекты не найдены!")
        return

    print(f"Всего найдено кандидатов: {len(all_detections)}")
    gdf = gpd.GeoDataFrame(all_detections, crs=crs)
    
    # Удаление дубликатов
    clean_gdf = geometric_nms(gdf, iou_threshold=cfg['iou_thres'])
    print(f"Объектов после очистки: {len(clean_gdf)}")

    # --- ЭТАП 3: Сохранение результатов ---

    # 1. SHAPEFILE (Вектор)
    shp_out = save_path / 'predictions.shp'
    # Shapefile не поддерживает списки в полях, удаляем bbox перед сохранением
    clean_gdf.drop(columns=['bbox']).to_file(shp_out, encoding='utf-8')
    print(f"Вектора сохранены: {shp_out}")

    # 2. CSV (Метаданные + WKT)
    csv_out = save_path / 'predictions.csv'
    df_export = pd.DataFrame(clean_gdf.drop(columns='geometry'))
    df_export['wkt'] = clean_gdf.geometry.apply(lambda g: g.wkt) # Геометрия текстом
    df_export['bbox'] = clean_gdf['bbox'].apply(lambda b: str(b))
    df_export.to_csv(csv_out, index=False)
    print(f"Таблица сохранена: {csv_out}")

    # 3. PROBABILITY RASTER (Тепловая карта уверенности)
    # Значения пикселей от 0.0 до 1.0 (float32)
    prob_out = save_path / 'predictions_probability.tif'
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=0.0)
    
    print("Генерация растра вероятностей...")
    # Сортируем по возрастанию: более уверенные объекты рисуются последними (поверх)
    sorted_gdf = clean_gdf.sort_values(by='conf', ascending=True)
    
    with rasterio.open(prob_out, 'w', **profile) as dst:
        shapes = ((row.geometry, row.conf) for _, row in sorted_gdf.iterrows())
        burned = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            dtype=rasterio.float32
        )
        dst.write(burned, 1)
    print(f"Растр вероятностей сохранен: {prob_out}")

    # 4. VISUALIZATION PREVIEW (JPG)
    # Создаем уменьшенную копию для быстрого просмотра
    print("Генерация превью...")
    scale = 0.1 # 10% от оригинала
    small_w, small_h = int(width * scale), int(height * scale)
    
    with rasterio.open(cfg['image_path']) as src:
        # Читаем уменьшенную версию всего снимка
        img_small = src.read([1, 2, 3], out_shape=(3, small_h, small_w))
        img_small = np.moveaxis(img_small, 0, -1)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
        
        # Матрица для перевода Map -> SmallPixel
        # x_small = (x_map - x_origin) / pixel_width * scale
        origin_x, origin_y = transform.c, transform.f
        px_w, px_h = transform.a, transform.e
        
        for _, row in clean_gdf.iterrows():
            if row.geometry.is_empty: continue
            
            geoms = [row.geometry] if row.geometry.geom_type == 'Polygon' else row.geometry.geoms
            
            for poly in geoms:
                pts = np.array(poly.exterior.coords)
                # Трансформация координат
                pts[:, 0] = (pts[:, 0] - origin_x) / px_w * scale
                pts[:, 1] = (pts[:, 1] - origin_y) / px_h * scale
                pts = pts.astype(np.int32)
                
                # Цвет зависит от уверенности (зеленый -> ярко-зеленый)
                color_intensity = int(row.conf * 255)
                cv2.polylines(img_small, [pts], True, (0, color_intensity, 0), 1)
                
        preview_out = save_path / 'preview.jpg'
        cv2.imwrite(str(preview_out), img_small)
        print(f"Превью сохранено: {preview_out}")

    print("Инференс завершен успешно.")

if __name__ == "__main__":
    run_inference(CONFIG)