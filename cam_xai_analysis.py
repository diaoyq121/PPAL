import torch
from src.explainability.cam.utils import save_cams
from src.explainability.cam.utils import save_mean_cams_folds
from src.explainability.cam.utils import entire_dataset_cams
from src.explainability.cam.utils import compute_xai_metrics


def main():
    model_name = "Axial3DVGG16"
    # Get and save the CAMs for each fold
    print(f"Getting and saving CAMs for {model_name}")

    save_cams(model_name=model_name, cam_method="gradcam++", num_workers=10, cuda_idx=[0])
    # Save the mean CAMs for each fold
    print(f"Saving mean CAMs for {model_name}")
    save_mean_cams_folds(model_name=model_name)
    # Save the entire dataset CAMs
    print(f"Saving entire dataset CAMs for {model_name}")
    entire_dataset_cams(model_name=model_name)
    # Compute the XAI metrics
    print(f"Computing XAI metrics for {model_name}")
    compute_xai_metrics(model_name=model_name, fold_num="entire_dataset", cam_method="gradcam++")


if __name__ == "__main__":
    # Print the PyTorch version used
    print(f"PyTorch version: {torch.__version__}")
    # Run the main function
    main()
