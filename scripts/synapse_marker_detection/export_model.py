import argparse

from flamingo_tools.model_utils import export_model


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
