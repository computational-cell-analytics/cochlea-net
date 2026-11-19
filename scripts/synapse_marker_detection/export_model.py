import argparse
import sys

import flamingo_tools.synapse_detection.detection_dataset as _dd
from flamingo_tools.model_utils import export_model

# Allow loading checkpoints that were saved when detection_dataset was a local script.
sys.modules["detection_dataset"] = _dd

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/schilling40/u15000/czii-protein-challenge")


def main():
    parser = argparse.ArgumentParser(
        description="Export a synapse detection model to a file with only its architecture and its weights."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="A torch_em checkpoint folder, a trainer checkpoint file (e.g. latest.pt), or a model file."
    )
    parser.add_argument("-o", "--output", required=True, help="The path of the exported file.")
    args = parser.parse_args()
    export_model(args.input, args.output)


if __name__ == "__main__":
    main()
