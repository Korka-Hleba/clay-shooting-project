import pandas as pd
import tempfile
import json


def export_to_excel(detections, filename="detections.xlsx"):
    data = []

    for d in detections:
        bbox = d.get("bbox", {})
        if isinstance(bbox, str):
            bbox = json.loads(bbox)

        row = {
            "ID": d.get("id"),
            "Видео файл": d.get("video_file"),
            "Время (сек)": d.get("timestamp"),
            "Состояние": d.get("target_state"),
            "Уверенность": d.get("confidence")
        }

        if bbox:
            row.update({
                "X1": bbox.get("x1"),
                "Y1": bbox.get("y1"),
                "X2": bbox.get("x2"),
                "Y2": bbox.get("y2")
            })

        data.append(row)

    if not data:
        data = [{"Сообщение": "Нет данных"}]

    df = pd.DataFrame(data)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    df.to_excel(temp_file.name, index=False)

    return temp_file.name


def export_to_csv(detections, filename="detections.csv"):
    data = []

    for d in detections:
        bbox = d.get("bbox", {})
        if isinstance(bbox, str):
            bbox = json.loads(bbox)

        x1 = bbox.get("x1") if bbox else None
        y1 = bbox.get("y1") if bbox else None
        x2 = bbox.get("x2") if bbox else None
        y2 = bbox.get("y2") if bbox else None

        width = x2 - x1 if x1 and x2 else None
        height = y2 - y1 if y1 and y2 else None

        data.append({
            "video_file": d.get("video_file"),
            "timestamp": d.get("timestamp"),
            "target_state": d.get("target_state"),
            "pos_x": x1,
            "pos_y": y1,
            "width": width,
            "height": height,
            "confidence": d.get("confidence")
        })

    if not data:
        data = [{"message": "No data"}]

    df = pd.DataFrame(data)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False, encoding='utf-8-sig')

    return temp_file.name