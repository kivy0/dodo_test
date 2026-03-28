import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import cv2
import numpy as np
import pandas as pd
import pytesseract
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo26s.pt")


class TableStatus(StrEnum):
    FREE = "free"
    OCCUPIED = "occupied"
    PASSING = "passing"


STATUS_COLORS = {
    TableStatus.FREE: sv.Color.GREEN,
    TableStatus.OCCUPIED: sv.Color.RED,
    TableStatus.PASSING: sv.Color.YELLOW,
}

STATUS_LABELS = {
    TableStatus.FREE: "FREE",
    TableStatus.OCCUPIED: "OCCUPIED",
    TableStatus.PASSING: "PASSING",
}

# ============ OCR ============


def preprocess_text_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    threshold = 200  # порог для белого
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel)
    binary = cv2.dilate(binary, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.bitwise_not(binary)  # Делаем текст черным

    border = 20
    binary = cv2.copyMakeBorder(
        binary, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )

    # cv2.imshow("Binary", binary)
    # cv2.waitKey(0)

    # cv2.imwrite("img2.png", crop)

    return binary


def detect_text(image, roi):
    """OCR детекция текста в участке изображения."""
    x, y, w, h = roi
    text_crop = image[y : y + h, x : x + w]

    processed_crop = preprocess_text_crop(text_crop)
    custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:-"
    text = pytesseract.image_to_string(processed_crop, config=custom_config)
    return text.strip()


def clean_text_for_date(text):
    """
    Очищает и нормализует текст в формат даты YYYY-MM-DD.
    Обрабатывает случаи с пропущенными дефисами.
    """

    # Если уже в правильном формате YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    # Убираем всё кроме цифр
    cleaned = re.sub(r"[^0-9]", "", text)

    # Если у нас 8 цифр YYYYMMDD
    if len(cleaned) == 8:
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"

    # Возвращаем как есть, если не удалось распознать
    return text


def clean_text_for_time(text):
    """Оставляем только цифры и двоеточия для времени формата HH:MM:SS."""
    # TODO добавить аналогичный date process
    return re.sub(r"[^0-9:]", "", text)


def detect_date(image, roi):
    """
    OCR детекция даты в участке изображения.
    Возращает date или None при неуспешном парсинге.
    """
    text = detect_text(image, roi)
    text = clean_text_for_date(text)
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except:
        print(f"Ошибка при парсинге даты. {text}")

    return None


def detect_time(image, roi):
    """
    OCR детекция времени в участке изображения.
    Возращает time или None при неуспешном парсинге.
    """

    text = detect_text(image, roi)
    text = clean_text_for_time(text)

    try:
        return datetime.strptime(text, "%H:%M:%S").time()
    except:
        print(f"Ошибка при парсинге времени. {text}")

    return None


# ============ Tracker logic ============


@dataclass
class TableOccupancyTracker:
    """Трекер занятости стола по пространственной тепловой карте."""

    table_roi: tuple[int, int, int, int]  # x, y, w, h

    grid_cols: int = 8
    grid_rows: int = 6

    overlap_threshold: float = 0.5

    heat_decay: float = 0.92
    heat_increment: float = 0.15

    occupied_threshold: float = 0.7

    _heatmap: np.ndarray = field(default=None, init=False)
    _prev_status: TableStatus = field(default=TableStatus.FREE, init=False)
    _last_free_time: datetime = field(default=None, init=False)
    events: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._heatmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

    def _bbox_overlap_ratio(self, xyxy) -> float:
        """Какая доля bbox находится внутри ROI стола."""
        x, y, w, h = self.table_roi
        tx1, ty1, tx2, ty2 = x, y, x + w, y + h

        xi1 = max(xyxy[0], tx1)
        yi1 = max(xyxy[1], ty1)
        xi2 = min(xyxy[2], tx2)
        yi2 = min(xyxy[3], ty2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        bbox_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        return intersection / bbox_area if bbox_area > 0 else 0.0

    def _bbox_to_grid_cells(self, xyxy) -> tuple[int, int, int, int] | None:
        """Конвертирует bbox в индексы ячеек сетки."""
        roi_x, roi_y, roi_w, roi_h = self.table_roi

        x1 = max(xyxy[0], roi_x) - roi_x
        y1 = max(xyxy[1], roi_y) - roi_y
        x2 = min(xyxy[2], roi_x + roi_w) - roi_x
        y2 = min(xyxy[3], roi_y + roi_h) - roi_y

        if x2 <= x1 or y2 <= y1:
            return None

        cell_w = roi_w / self.grid_cols
        cell_h = roi_h / self.grid_rows

        col1 = max(0, int(x1 / cell_w))
        row1 = max(0, int(y1 / cell_h))
        col2 = min(self.grid_cols, int(np.ceil(x2 / cell_w)))
        row2 = min(self.grid_rows, int(np.ceil(y2 / cell_h)))

        return col1, row1, col2, row2

    def _update_heatmap(self, detections) -> bool:
        """Обновляет тепловую карту. Возвращает True если кто-то в зоне."""
        self._heatmap *= self.heat_decay

        has_person = False

        for xyxy in detections.xyxy:
            if self._bbox_overlap_ratio(xyxy) < self.overlap_threshold:
                continue

            has_person = True
            cells = self._bbox_to_grid_cells(xyxy)
            if cells is None:
                continue

            col1, row1, col2, row2 = cells

            self._heatmap[row1:row2, col1:col2] += self.heat_increment

        np.clip(self._heatmap, 0, 1.0, out=self._heatmap)
        return has_person

    def _determine_status(self, has_person: bool) -> TableStatus:
        """Статус по максимуму тепловой карты."""
        if self._heatmap.max() >= self.occupied_threshold:
            return TableStatus.OCCUPIED

        if has_person:
            return TableStatus.PASSING

        return TableStatus.FREE

    def update(self, detections, frame_datetime: datetime = None) -> TableStatus:
        has_person = self._update_heatmap(detections)
        status = self._determine_status(has_person)

        if frame_datetime and status != self._prev_status:
            self._record_event(status, frame_datetime)

        self._prev_status = status
        return status

    def _record_event(self, new_status: TableStatus, dt: datetime):
        if new_status == TableStatus.FREE:
            self._last_free_time = dt
            self.events.append(
                {"timestamp": dt, "event": "became_free", "wait_seconds": None}
            )
        elif self._prev_status == TableStatus.FREE and self._last_free_time:
            wait = (dt - self._last_free_time).total_seconds()
            self.events.append(
                {"timestamp": dt, "event": "person_arrived", "wait_seconds": wait}
            )
            self._last_free_time = None

    def get_heatmap(self) -> np.ndarray:
        return self._heatmap.copy()

    def get_events_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def get_stats(self) -> dict:
        df = self.get_events_df()
        waits = df[df["event"] == "person_arrived"]["wait_seconds"].dropna()
        if waits.empty:
            return {"mean_wait": None, "count": 0}
        return {
            "mean_wait": waits.mean(),
            "median_wait": waits.median(),
            "min_wait": waits.min(),
            "max_wait": waits.max(),
            "count": len(waits),
        }


# ============ Video processing ============


def select_roi(image, description):
    """Создание окна для выбора области вручную."""
    roi = cv2.selectROI(description, image, fromCenter=False)
    cv2.destroyWindow(description)
    return roi


def make_datetime_detections(
    date_roi: tuple[int, int, int, int],
    time_roi: tuple[int, int, int, int],
    date_text: str,
    time_text: str,
) -> tuple[sv.Detections, list[str]]:
    """Создаёт detections и labels для ROI даты/времени."""
    rois = [date_roi, time_roi]
    xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in rois])
    labels = [str(date_text), str(time_text)]
    return sv.Detections(xyxy=xyxy), labels


def annotate_scene(
    frame: np.ndarray,
    layers: list[
        tuple[
            sv.Detections, list[str] | None, sv.BoxAnnotator, sv.LabelAnnotator | None
        ]
    ],
) -> np.ndarray:
    """Универсальная отрисовка нескольких слоёв аннотаций."""
    scene = frame.copy()
    for detections, labels, box_ann, label_ann in layers:
        scene = box_ann.annotate(scene=scene, detections=detections)
        if label_ann and labels:
            scene = label_ann.annotate(
                scene=scene, detections=detections, labels=labels
            )
    return scene


def create_table_annotators(
    status: TableStatus,
) -> tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    """Создание аннотаторов с цветом по статусу."""
    color = STATUS_COLORS[status]

    box_annotator = sv.BoxAnnotator(color=color, thickness=3)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_scale=1.0,
        text_thickness=2,
        text_position=sv.Position.TOP_CENTER,
    )

    return box_annotator, label_annotator


def annotate_table_with_status(
    frame: np.ndarray,
    table_roi: tuple[int, int, int, int],
    status: TableStatus,
) -> np.ndarray:
    """Аннотация стола со статусом."""
    x, y, w, h = table_roi

    table_dets = sv.Detections(
        xyxy=np.array([[x, y, x + w, y + h]]),
        class_id=np.array([0]),
    )

    label = f"{STATUS_LABELS[status]}"

    box_ann, label_ann = create_table_annotators(status)

    frame = box_ann.annotate(frame, table_dets)
    frame = label_ann.annotate(frame, table_dets, labels=[label])

    return frame


def draw_heatmap_overlay(
    frame: np.ndarray,
    table_roi: tuple[int, int, int, int],
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Накладывает тепловую карту на область стола.

    frame: исходный кадр
    table_roi: (x, y, w, h) области стола
    heatmap: 2D массив тепловой карты от трекера
    alpha: прозрачность наложения (0-1)
    """
    x, y, w, h = table_roi
    frame = frame.copy()

    # Нормализуем к 0-255
    heatmap_norm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Масштабируем до размера ROI
    heatmap_resized = cv2.resize(
        heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Накладываем
    roi = frame[y : y + h, x : x + w]
    frame[y : y + h, x : x + w] = cv2.addWeighted(
        roi, 1 - alpha, heatmap_resized, alpha, 0
    )

    return frame


def process_video(video_path: str, output_path="res.mp4"):
    cap = cv2.VideoCapture(video_path)

    # Получаем параметры исходного видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_num = 0

    # Считываем первый кадр для выбора ROI
    ret, frame = cap.read()

    table_roi = select_roi(frame, "Select table area.")
    date_roi = select_roi(frame, "Select date area.")
    time_roi = select_roi(frame, "Select time area.")

    date = detect_date(frame, date_roi)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    dt_box = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
    dt_label = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=1.5,
        text_thickness=3,
        text_position=sv.Position.BOTTOM_CENTER,
    )

    person_box = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
    person_label = sv.LabelAnnotator(
        color=sv.Color.RED,
        text_scale=0.8,
        text_thickness=2,
    )

    table_tracker = TableOccupancyTracker(
        table_roi=table_roi,
        grid_cols=10,
        grid_rows=10,
        occupied_threshold=0.8,
        heat_decay=0.98,
        heat_increment=0.02,
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = detect_time(frame, time_roi)

        if date and frame_time:
            frame_datetime = datetime.combine(date, frame_time)

        dt_dets, dt_labels = make_datetime_detections(
            date_roi, time_roi, date, frame_time
        )

        detection_result = model.predict(frame, classes=[0], nms=True)[0]
        person_dets = sv.Detections.from_ultralytics(detection_result)

        table_status = table_tracker.update(person_dets, frame_datetime)

        # ===== Аннотация =====
        # Базовые слои
        layers = [
            (dt_dets, dt_labels, dt_box, dt_label),
            (person_dets, None, person_box, person_label),
        ]

        annotated_frame = annotate_scene(frame, layers)

        annotated_frame = draw_heatmap_overlay(
            annotated_frame,
            table_roi,
            table_tracker.get_heatmap(),
            alpha=0.4,
        )

        # Стол со статусом
        annotated_frame = annotate_table_with_status(
            annotated_frame,
            table_roi,
            table_status,
        )

        out.write(annotated_frame)
        frame_num += 1

    cap.release()
    out.release()
    print(f"Обработаное видео сохранено: {output_path}")

    events_df = table_tracker.get_events_df()
    print(events_df)
    events_df.to_csv("events.csv")
    stats = table_tracker.get_stats()
    print(f"Среднее время ожидания: {stats['mean_wait']:.1f} сек")


def main():
    # Создаем парсер для аргументов командной строки
    parser = argparse.ArgumentParser(description="Путь к видео.")
    parser.add_argument("--video", help="Путь к видео.")

    args = parser.parse_args()

    # Запускаем обработку
    video_path = args.video
    process_video(video_path)


if __name__ == "__main__":
    main()
