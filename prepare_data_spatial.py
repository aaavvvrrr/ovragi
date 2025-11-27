import os
import cv2
import yaml
import random
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.ops import transform
from pathlib import Path
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    'raster_dir': './raw_data/images',
    'vector_path': './raw_data/gully shapes.shp',
    'output_dir': './datasets/satellite_seg',
    'tile_size': 640,
    'overlap': 0.2,
    'val_split': 0.4,     # (val_split*100)% площади уйдет на валидацию
    'grid_cells': 10,     # Разбиваем область на 10x10 крупных блоков
    'class_id': 0,
    'bg_ratio': 0.001,
}

def create_spatial_split_grid(raster_files, grid_cells, val_ratio):
    """
    Создает гео-сетку (Grid) на основе объединения границ всех растров.
    Возвращает GeoDataFrame, где каждой ячейке назначен split='train' или 'val'.
    """
    print("Генерация пространственной сетки для разбиения...")
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    
    # 1. Считаем общие границы всех снимков
    for r_path in tqdm(raster_files, desc="Анализ границ"):
        with rasterio.open(r_path) as src:
            b = src.bounds
            min_x = min(min_x, b.left)
            min_y = min(min_y, b.bottom)
            max_x = max(max_x, b.right)
            max_y = max(max_y, b.top)

    # 2. Создаем сетку
    width = max_x - min_x
    height = max_y - min_y
    step_x = width / grid_cells
    step_y = height / grid_cells
    
    polygons = []
    splits = []
    
    for i in range(grid_cells):
        for j in range(grid_cells):
            # Координаты ячейки
            x1 = min_x + i * step_x
            y1 = min_y + j * step_y
            x2 = x1 + step_x
            y2 = y1 + step_y
            polygons.append(box(x1, y1, x2, y2))
            
            # Случайное назначение с учетом вероятности
            if random.random() < val_ratio:
                splits.append('val')
            else:
                splits.append('train')
                
    grid_gdf = gpd.GeoDataFrame({'split': splits, 'geometry': polygons})
    # Задаем CRS (берем из первого растра, предполагая, что все в одной системе)
    with rasterio.open(raster_files[0]) as src:
        grid_gdf.set_crs(src.crs, inplace=True)
        
    print(f"Сетка создана: {len(grid_gdf[grid_gdf['split']=='train'])} train зон, {len(grid_gdf[grid_gdf['split']=='val'])} val зон.")
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
    raster_files = list(Path(cfg['raster_dir']).glob("*.tif"))
    
    # --- ЭТАП 1: Создаем Spatial Grid для разделения ---
    split_grid = create_spatial_split_grid(raster_files, cfg['grid_cells'], cfg['val_split'])
    # Создаем пространственный индекс для быстрого поиска зоны
    split_sindex = split_grid.sindex

    stride = int(cfg['tile_size'] * (1 - cfg['overlap']))

    for raster_path in tqdm(raster_files, desc="Обработка снимков"):
        with rasterio.open(raster_path) as src:
            # Приводим вектора к CRS растра
            if gdf.crs != src.crs:
                gdf_local = gdf.to_crs(src.crs)
            else:
                gdf_local = gdf
            
            # Оптимизация: работаем только с объектами в пределах снимка
            img_bounds = box(*src.bounds)
            relevant_gdf = gdf_local[gdf_local.geometry.intersects(img_bounds)]
            if relevant_gdf.empty: continue # Пропускаем пустые снимки, если нет векторов совсем

            width, height = src.width, src.height
            
            # --- Итерация по тайлам ---
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    x_off = min(x, width - cfg['tile_size'])
                    y_off = min(y, height - cfg['tile_size'])
                    
                    window = Window(x_off, y_off, cfg['tile_size'], cfg['tile_size'])
                    win_transform = src.window_transform(window)
                    
                    # Геометрические границы тайла (в координатах карты)
                    minx_win = win_transform.c
                    maxy_win = win_transform.f
                    maxx_win = win_transform.c + win_transform.a * cfg['tile_size']
                    miny_win = win_transform.f + win_transform.e * cfg['tile_size']
                    win_bounds = box(minx_win, miny_win, maxx_win, maxy_win)
                    
                    # --- ОПРЕДЕЛЕНИЕ SPLIT (Train/Val) ---
                    # Берем центр тайла
                    center_point = Point((minx_win + maxx_win)/2, (miny_win + maxy_win)/2)
                    
                    # Ищем, в какую ячейку сетки попадает центр тайла
                    # query returns indices of intersecting geometries
                    possible_idx = list(split_sindex.intersection(center_point.bounds))
                    current_split = 'train' # fallback
                    
                    if possible_idx:
                        # Берем первую попавшуюся зону (обычно она одна)
                        grid_row = split_grid.iloc[possible_idx[0]]
                        current_split = grid_row['split']
                    
                    # --- ПОИСК ОБЪЕКТОВ В ТАЙЛЕ ---
                    possible_matches_index = list(relevant_gdf.sindex.intersection(win_bounds.bounds))
                    possible_matches = relevant_gdf.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(win_bounds)]

                    yolo_labels = []
                    if not precise_matches.empty:
                        for _, row in precise_matches.iterrows():
                            intersection = row.geometry.intersection(win_bounds)
                            
                            # Transform map -> pixel
                            def transform_coords(xc, yc):
                                px = (xc - win_transform.c) / win_transform.a
                                py = (yc - win_transform.f) / win_transform.e
                                return px, py
                            
                            poly_pixel = transform(transform_coords, intersection)
                            norm_coords_list = normalize_polygon(poly_pixel, cfg['tile_size'], cfg['tile_size'])
                            
                            for coords in norm_coords_list:
                                label_str = f"{cfg['class_id']} " + " ".join(map(str, coords))
                                yolo_labels.append(label_str)

                    # --- СОХРАНЕНИЕ ---
                    has_objects = len(yolo_labels) > 0
                    # Сохраняем, если есть объекты, ИЛИ рандомно фон
                    should_save = has_objects or (random.random() < cfg['bg_ratio'])

                    if should_save:
                        # Читаем данные только сейчас, чтобы не грузить диск зря
                        img = src.read([1, 2, 3], window=window)
                        img = np.moveaxis(img, 0, -1)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # Имя файла включает координаты, чтобы избежать коллизий
                        # Имя папки (train/val) берем из current_split
                        base_name = f"{raster_path.stem}_x{x_off}_y{y_off}"
                        img_out = save_path / 'images' / current_split / f"{base_name}.jpg"
                        lbl_out = save_path / 'labels' / current_split / f"{base_name}.txt"

                        cv2.imwrite(str(img_out), img)
                        with open(lbl_out, 'w') as f:
                            f.write("\n".join(yolo_labels))

    # Создаем yaml
    yaml_content = {
        'path': str(save_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'object'}
    }
    with open(save_path / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

    print("Подготовка завершена с пространственным разделением!")

if __name__ == "__main__":
    process_dataset(CONFIG)