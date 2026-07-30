import os

from micro_sam.training.training import export_instance_segmentation_model

MODEL_NAME = "cochlea_micro_sam_2026-07-28_micro-sam_ihc-v11_2d"
MODEL_TYPE = "vit_b_lm"
MODEL_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC"


def main():
    checkpoint_path = os.path.join(MODEL_DIR, MODEL_NAME, "best.pt")
    output_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")

    export_instance_segmentation_model(checkpoint_path, output_path, model_type=MODEL_TYPE)

    from micro_sam.instance_segmentation import get_predictor_and_decoder
    print("Verifying model loads correctly ...")
    get_predictor_and_decoder(model_type="vit_b", checkpoint_path=output_path)
    print("Verification passed.")


if __name__ == "__main__":
    main()

