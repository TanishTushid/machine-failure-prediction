"""Generate global and local SHAP explanations for the selected pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_dataset
from src.predict import get_model_bundle
from src.preprocessing import FEATURE_COLUMNS


def save_shap_explanations(
    data_path: str | Path | None = None,
    output_dir: str | Path = Path("artifacts") / "shap",
    *,
    background_size: int = 100,
    sample_size: int = 250,
) -> list[Path]:
    """Save model-aligned SHAP summary and one individual explanation."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is required for explanations. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    dataframe = load_dataset(data_path)
    features = dataframe.loc[:, FEATURE_COLUMNS].sample(
        min(sample_size, len(dataframe)), random_state=42
    )
    background = features.sample(min(background_size, len(features)), random_state=43)
    bundle = get_model_bundle()
    engineering = bundle.pipeline.named_steps["feature_engineering"]
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    estimator = bundle.pipeline.named_steps["model"]

    background_transformed = preprocessor.transform(engineering.transform(background))
    sample_transformed = preprocessor.transform(engineering.transform(features))
    names = preprocessor.get_feature_names_out()
    background_frame = pd.DataFrame(
        background_transformed.toarray()
        if hasattr(background_transformed, "toarray")
        else background_transformed,
        columns=names,
    )
    sample_frame = pd.DataFrame(
        sample_transformed.toarray()
        if hasattr(sample_transformed, "toarray")
        else sample_transformed,
        columns=names,
    )

    failure_score = lambda values: estimator.predict_proba(values)[:, 1]
    explainer = shap.Explainer(
        failure_score,
        background_frame,
        algorithm="permutation",
        output_names=["failure score"],
    )
    explanation = explainer(sample_frame, max_evals=2 * len(names) + 1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "shap_global_importance.png", output / "shap_example.png"]

    plt.figure()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(paths[0], dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(paths[1], dpi=160, bbox_inches="tight")
    plt.close()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "shap")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in save_shap_explanations(args.data, args.output):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
