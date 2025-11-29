import os
import cv2
import yaml
import random
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box, Point, MultiPolygon
from shapely.ops import transform
from pathlib import Path
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    'raster_dir': './raw_data/images',
    'vector_path': './raw_data/labels.shp',
    'output_dir': './datasets/satellite_seg',
    
    'tile_size': 640,
    'overlap': 0.2,
    'val_split': 0.2,
    'grid_cells': 10,
    'bg_ratio': 0.1,      # Доля пустых тайлов (фон)
    
    # --- НАСТРОЙКИ КЛАССОВ ---
    'class_column': 'class_id',  # Имя поля в .shp, где лежит ID класса (int)
    'default_class_id': 0,       # ID по умолчанию, если поля нет
    
    # Имена классов для data.yaml (важно для визуализации и метрик)
    'class_names': {
        0: 'ravine',    # Овраг
        1: 'hard_negative',      # Дорога 
    }
}

def create_spatial_split_grid(raster_files, grid_cells, val_ratio):
    """Создает гео-сетку для разбиения на train/val без утечек."""
    print("Генерация пространственной сетки...")
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    
    # Считаем общие границы
    for r_path in tqdm(raster_files, desc="Анализ границ"):
        with rasterio.open(r_path) as src:
            b = src.bounds
            min_x, min_y = min(min_x, b.left), min(min_y, b.bottom)
            max_x, max_y = max(max_x, b.right), max(max_y, b.top)

    width, height = max_x - min_x, max_y - min_y
    step_x, step_y = width / grid_cells, height / grid_cells
    
    polygons, splits = [], []
    for i in range(grid_cells):
        for j in range(grid_cells):
            x1 = min_x + i * step_x
            y1 = min_y + j * step_y
            polygons.append(box(x1, y1, x1 + step_x, y1 + step_y))
            splits.append('val' if random.random() < val_ratio else 'train')
                
    grid_gdf = gpd.GeoDataFrame({'split': splits, 'geometry': polygons})
    # Присваиваем CRS из первого файла
    if raster_files:
        with rasterio.open(raster_files[0]) as src:
            grid_gdf.set_crs(src.crs, inplace=True)
            
    return grid_gdf

def normalize_polygon(polygon, tile_w, tile_h):
    """Нормализация координат для YOLO (0-1)."""
    if polygon.is_empty: return []
    poly_simplified = polygon.simplify(1.0, preserve_topology=False)
    
    geoms = poly_simplified.geoms if isinstance(poly_simplified, MultiPolygon) else [poly_simplified]
    normalized_polys = []
    
    for geom in geoms:
        if geom.is_empty: continue
        pts = np.array(geom.exterior.coords)
        pts[:, 0] /= tile_w
        pts[:, 1] /= tile_h
        pts = np.clip(pts, 0.0, 1.0)
        normalized_polys.append(pts.flatten().tolist())
    return normalized_polys

def process_dataset(cfg):
    save_path = Path(cfg['output_dir'])
    for split in ['train', 'val']:
        (save_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (save_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("Загрузка векторных данных...")
    gdf = gpd.read_file(cfg['vector_path'])
    
    # Проверка наличия колонки класса
    has_class_col = cfg['class_column'] in gdf.columns
    if has_class_col:
        print(f"Найден атрибут классов: '{cfg['class_column']}'")
    else:
        print(f"Атрибут '{cfg['class_column']}' не найден. Используем ID={cfg['default_class_id']} для всех объектов.")

    raster_files = list(Path(cfg['raster_dir']).glob("*.tif"))
    
    # 1. Spatial Grid
    split_grid = create_spatial_split_grid(raster_files, cfg['grid_cells'], cfg['val_split'])
    split_sindex = split_grid.sindex
    stride = int(cfg['tile_size'] * (1 - cfg['overlap']))

    for raster_path in tqdm(raster_files, desc="Обработка снимков"):
        with rasterio.open(raster_path) as src:
            if gdf.crs != src.crs:
                gdf_local = gdf.to_crs(src.crs)
            else:
                gdf_local = gdf
            
            # Предварительная фильтрация векторов по границам снимка
            img_bounds = box(*src.bounds)
            relevant_gdf = gdf_local[gdf_local.geometry.intersects(img_bounds)]
            if relevant_gdf.empty: continue

            width, height = src.width, src.height
            
            # --- Итерация по тайлам ---
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    x_off = min(x, width - cfg['tile_size'])
                    y_off = min(y, height - cfg['tile_size'])
                    
                    window = Window(x_off, y_off, cfg['tile_size'], cfg['tile_size'])
                    win_transform = src.window_transform(window)
                    
                    # Границы окна в координатах карты
                    minx, maxy = win_transform.c, win_transform.f
                    maxx = win_transform.c + win_transform.a * cfg['tile_size']
                    miny = win_transform.f + win_transform.e * cfg['tile_size']
                    win_bounds = box(minx, miny, maxx, maxy)
                    
                    # Определение SPLIT (Train/Val)
                    center_point = Point((minx + maxx)/2, (miny + maxy)/2)
                    possible_idx = list(split_sindex.intersection(center_point.bounds))
                    current_split = 'train'
                    if possible_idx:
                        current_split = split_grid.iloc[possible_idx[0]]['split']
                    
                    # Поиск объектов в тайле
                    precise_matches = relevant_gdf[relevant_gdf.intersects(win_bounds)]

                    yolo_labels = []
                    if not precise_matches.empty:
                        for idx, row in precise_matches.iterrows():
                            # --- ОПРЕДЕЛЕНИЕ КЛАССА ---
                            obj_class = cfg['default_class_id']
                            if has_class_col:
                                try:
                                    val = row[cfg['class_column']]
                                    # Проверка на NaN и приведение к int
                                    if pd.notna(val):
                                        obj_class = int(val)
                                except (ValueError, TypeError):
                                    pass # Оставляем дефолтный если мусор в данных

                            # Обработка геометрии
                            intersection = row.geometry.intersection(win_bounds)
                            
                            def transform_coords(xc, yc):
                                px = (xc - win_transform.c) / win_transform.a
                                py = (yc - win_transform.f) / win_transform.e
                                return px, py
                            
                            poly_pixel = transform(transform_coords, intersection)
                            norm_coords_list = normalize_polygon(poly_pixel, cfg['tile_size'], cfg['tile_size'])
                            
                            for coords in norm_coords_list:
                                label_str = f"{obj_class} " + " ".join(map(str, coords))
                                yolo_labels.append(label_str)

                    # Логика сохранения
                    has_objects = len(yolo_labels) > 0
                    should_save = has_objects or (random.random() < cfg['bg_ratio'])

                    if should_save:
                        img = src.read([1, 2, 3], window=window)
                        img = np.moveaxis(img, 0, -1)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        base_name = f"{raster_path.stem}_x{x_off}_y{y_off}"
                        img_out = save_path / 'images' / current_split / f"{base_name}.jpg"
                        lbl_out = save_path / 'labels' / current_split / f"{base_name}.txt"

                        cv2.imwrite(str(img_out), img)
                        with open(lbl_out, 'w') as f:
                            f.write("\n".join(yolo_labels))

    # Создаем yaml с учетом имен классов из конфига
    yaml_content = {
        'path': str(save_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': cfg['class_names']
    }
    with open(save_path / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

    print("Подготовка завершена. Data.yaml обновлен.")

if __name__ == "__main__":
    import pandas as pd # Нужен для проверки pd.notna
    process_dataset(CONFIG)