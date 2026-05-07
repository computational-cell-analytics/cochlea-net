import os
import urllib.request

from micro_sam.training.training import export_instance_segmentation_model


OWNCLOUD_URL = "https://owncloud.gwdg.de/index.php/s/v8QBny7BOtwiHoM/download"
MODEL_NAME = "cochlea_micro-sam_sgn-v2_2d"
MODEL_TYPE = "vit_b_lm"


def download_checkpoint(save_path):
    if os.path.exists(save_path):
        print(f"Checkpoint already at '{save_path}', skipping download.")
        return
    print("Downloading checkpoint ...")
    urllib.request.urlretrieve(OWNCLOUD_URL, save_path)
    print(f"Downloaded to '{save_path}'.")


def main():
    checkpoint_path = f"{MODEL_NAME}_checkpoint.pt"
    output_path = f"{MODEL_NAME}.pt"

    download_checkpoint(checkpoint_path)
    export_instance_segmentation_model(checkpoint_path, output_path, model_type=MODEL_TYPE)

    from micro_sam.instance_segmentation import get_predictor_and_decoder
    print("Verifying model loads correctly ...")
    get_predictor_and_decoder(model_type="vit_b", checkpoint_path=output_path)
    print("Verification passed.")


if __name__ == "__main__":
    main()
