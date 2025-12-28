from ultralytics import YOLO
from pathlib import Path
import shutil


def train_clay_target_model():
    data_yaml_path = Path(__file__).parent.parent / "dataset" / "data.yaml"

    print(f"Использую data.yaml: {data_yaml_path}")

    dataset_dir = data_yaml_path.parent
    train_path = dataset_dir / "train" / "images"
    val_path = dataset_dir / "val" / "images"

    print(f"Папка train: {train_path}")
    print(f"Папка val: {val_path}")

    train_files = list(train_path.glob("*"))
    val_files = list(val_path.glob("*"))

    print(f"Изображений в train: {len(train_files)}")
    print(f"Изображений в val: {len(val_files)}")

    if len(train_files) == 0:
        print("Ошибка: Нет изображений в train/images")
        return

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model = YOLO('yolov8n.pt')

    print("=" * 50)

    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=50,
            imgsz=640,
            batch=8,
            name='clay_target',
            project=str(models_dir),
            exist_ok=True,
            patience=10,
            save=True,
            save_period=10,
            verbose=True,
            workers=0,
            plots=True
        )

        print("=" * 50)
        print("Обучение завершено!")

        train_dir = models_dir / "clay_target"
        if train_dir.exists():
            best_model = train_dir / "weights" / "best.pt"
            if best_model.exists():
                shutil.copy(best_model, models_dir / "best.pt")
                print(f"Лучшая модель сохранена: models/best.pt")

                import pandas as pd
                results_csv = train_dir / "results.csv"
                if results_csv.exists():
                    try:
                        df = pd.read_csv(results_csv)
                        if 'metrics/mAP50(B)' in df.columns:
                            best_map = df['metrics/mAP50(B)'].max()
                            print(f"Лучший mAP50: {best_map:.4f}")
                    except:
                        pass

        print(f"Результаты в: {train_dir}")

    except Exception as e:
        print(f"Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_clay_target_model()