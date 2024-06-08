import argparse
from pathlib import Path
import subprocess

from tqdm import tqdm

from app.task.classify_error_type import BoxErrorTypeAnalyzer
from app.task.study_in_statistics import recorder, Statistician, PolygonAnnotator
from app.utils.parse import TruthPredictionFetcher, yaml_load


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", help="the parameters of the dataset", type=str)
    parser.add_argument("--data_subset", help="train, val or test set", type=str)
    parser.add_argument("--predict_folder_path", help="path of the prediction outputs", type=str)
    parser.add_argument("--analysis_folder_path", help="path of the analyzed outputs", type=str)
    parser.add_argument("--is_obb", help="whether the dataset is obb", action="store_true")
    return parser.parse_args()


def analyze_errors(args: argparse.Namespace, analysis_folder_path: Path):
    fetcher = TruthPredictionFetcher(
        args.yaml_path, args.data_subset, args.predict_folder_path, is_obb=args.is_obb)
    data = []
    for (fact, guess) in tqdm(fetcher, total=len(fetcher), desc="Analyzing"):
        analyzer = BoxErrorTypeAnalyzer(fact, guess, is_obb=args.is_obb)
        analyzer.analyze()
        data.extend(analyzer.parse_analysis_results())
    recorder.write_csv(analysis_folder_path / f"{args.data_subset}.csv", data)
    print(f"Analyzed results are saved in {analysis_folder_path / f'{args.data_subset}.csv'}")


def annotate_errors(args: argparse.Namespace, analysis_folder_path: Path, yaml_data: dict):
    src_img_folder = str(yaml_data[args.data_subset])
    annotator = PolygonAnnotator(
        src_img_folder=src_img_folder,
        prediction_folder=args.predict_folder_path,
        dst_img_folder=str(analysis_folder_path / 'images'),
        class_names=yaml_data['names'],
        analyzed=recorder.read_csv(analysis_folder_path / f"{args.data_subset}.csv"))
    annotator.run()
    print(f"Annotated images are saved in {analysis_folder_path / 'images'}")


def make_statistic(args: argparse.Namespace, analysis_folder_path: Path, yaml_data: dict):
    stat = Statistician(output_folder_path=analysis_folder_path,
                        df=recorder.read_csv(analysis_folder_path / f"{args.data_subset}.csv"))
    stat.error_proportion()
    stat.sort_by_errors_per_image()
    stat.delta_map(target_classes=None, num_classes=len(yaml_data['names']))
    print(f"Statistics are saved in {analysis_folder_path}")


def backend(args: argparse.Namespace):
    analysis_folder_path = Path(args.analysis_folder_path)
    analyze_errors(args, analysis_folder_path)

    yaml_data = yaml_load(args.yaml_path)
    annotate_errors(args, analysis_folder_path, yaml_data)
    make_statistic(args, analysis_folder_path, yaml_data)


def main():
    args = get_args()
    backend(args)
    cmd = (f"streamlit run gui.py --client.showSidebarNavigation=False -- --folder {args.analysis_folder_path}"
           f" --subset {args.data_subset} --yaml_path {args.yaml_path}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

"""
* Script
$ python main.py --yaml_path "datasets/cfg/coco128.yaml" --data_subset train --predict_folder_path "datasets/predictions/set_train" --analysis_folder_path "datasets/predictions/analyzed_train"
"""
