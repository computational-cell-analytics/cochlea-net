import argparse
import os
from pathlib import Path
from shutil import rmtree

import pybdv.metadata as bdv_metadata
import torch
import z5py

from flamingo_tools.segmentation import run_unet_prediction, filter_isolated_objects
from flamingo_tools.mobie import add_raw_to_mobie, add_segmentation_to_mobie


def postprocess_seg(output_folder):
    print("Run segmentation postprocessing ...")
    seg_path = os.path.join(output_folder, "segmentation.zarr")
    seg_key = "segmentation"

    with z5py.File(seg_path, "r") as f:
        segmentation = f[seg_key][:]

    seg_filtered, n_pre, n_post = filter_isolated_objects(segmentation)

    with z5py.File(seg_path, "a") as f:
        chunks = f[seg_key].chunks
        f.create_dataset(
            "segmentation_postprocessed", data=seg_filtered, compression="gzip",
            chunks=chunks, dtype=seg_filtered.dtype
        )


def export_to_mobie(xml_path, segmentation_folder, output_folder, scale, mobie_dataset, chunks, channel_names):
    # Add to mobie: All the channels.
    setup_ids = bdv_metadata.get_setup_ids(xml_path)
    if channel_names is None:
        channel_names = [f"channel-{i}" for i in range(len(setup_ids))]
    else:
        assert len(channel_names) == len(setup_ids)
    for i, setup_id in enumerate(setup_ids):
        add_raw_to_mobie(
            mobie_project=output_folder,
            mobie_dataset=mobie_dataset,
            source_name=channel_names[i],
            xml_path=xml_path,
            setup_id=setup_id
        )

    # The segmentation and post-processed segmentation results.
    seg_path = os.path.join(segmentation_folder, "segmentation.zarr")
    seg_resolution = bdv_metadata.get_resolution(xml_path, setup_id=0)
    if scale == 1:
        seg_resolution = [2 * res for res in seg_resolution]
    unit = bdv_metadata.get_unit(xml_path, setup_id=0)

    seg_key = "segmentation"
    seg_name = "nuclei_fullscale" if scale == 0 else "nuclei_downscaled"
    add_segmentation_to_mobie(
        mobie_project=output_folder,
        mobie_dataset=mobie_dataset,
        source_name=seg_name,
        segmentation_path=seg_path,
        segmentation_key=seg_key,
        resolution=seg_resolution,
        unit=unit,
        scale_factors=4 * [[2, 2, 2]],
        chunks=chunks,
    )

    seg_key = "segmentation_postprocessed"
    seg_name += "_postprocessed"
    add_segmentation_to_mobie(
        mobie_project=output_folder,
        mobie_dataset=mobie_dataset,
        source_name=seg_name,
        segmentation_path=seg_path,
        segmentation_key=seg_key,
        resolution=seg_resolution,
        unit=unit,
        scale_factors=4 * [[2, 2, 2]],
        chunks=chunks,
    )


def main():
    # Argument parser so that this script can be used from the command line.
    parser = argparse.ArgumentParser(
        description="Run segmentation and export the segmentation result for a lightsheet volume."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input volume. This should be the path to the xml file obtained after stitching."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output folder. This is where the MoBIE project, with image data and segmentation result, will be stored."  # noqa
    )
    parser.add_argument(
        "-s", "--segmentation_folder", required=True,
        help="Path to a folder where intermediate results for the segmentation will be stored. "
        "The results will be removed after the export to MoBIE."
    )
    parser.add_argument(
        "--mobie_dataset",
        help="Internal name of the dataset in MoBIE. If not given this will be derived from the name of the input volume.",  # noqa
    )
    parser.add_argument(
        "--setup_id", default=0, type=int,
        help="The setup id to use for the segmentation. Choose the setup-id for the channel that contains the data to be used for segmentation."  # noqa
        "This should be the PV channel for SGN segmentation."
    )
    parser.add_argument(
        "--scale", default=0, type=int,
        help="The scale to use for segmentation. By default this will run at the lowest scale (= full resolution)."
    )
    parser.add_argument("--model")
    parser.add_argument("--channel_names", nargs="+", default=None, help="The names of channels in the dataset, in the same order as the setup-ids.")  # noqa
    args = parser.parse_args()

    # This is just some preparation logic to get a good size for filtering
    # the nuclei depending on which scale we use for running the segmentation.
    scale = args.scale
    if scale == 0:
        min_size = 1000
    elif scale == 1:
        min_size = 250
    else:
        raise ValueError

    # Here we read the path to the data from the xml file and we construct the
    # input key (= internal file path in the n5 file with the data),
    # that points to the correct setup-id and scale.
    xml_path = args.input
    assert os.path.splitext(xml_path)[1] == ".xml"
    input_path = bdv_metadata.get_data_path(xml_path, return_absolute_path=True)
    input_key = f"setup{args.setup_id}/timepoint0/s{scale}"

    # This is just some preparation to choose the correct block sizes for running prediction
    # depending on having a GPU or not available.
    # (You will need a GPU to run this for any larger volume, CPU support is just for testing purposes.)
    have_cuda = torch.cuda.is_available()
    chunks = z5py.File(input_path, "r")[input_key].chunks
    block_shape = tuple([2 * ch for ch in chunks]) if have_cuda else tuple(chunks)
    halo = (16, 64, 64) if have_cuda else (8, 32, 32)

    # Here we find the path to the model for segmentation.
    # If the path it given it should point to the ".pt" file.
    # Otherwise, we try to load the model from where the checkpoint was stored on my system.
    if args.model is not None:
        model = args.model
    else:
        if scale == 0:
            model = "../training/checkpoints/cochlea_distance_unet"
        else:
            model = "../training/checkpoints/cochlea_distance_unet-train-downsampled"

    # These functions run the actual segmentation and the segmentation postprocessing.
    run_unet_prediction(
        input_path, input_key, args.segmentation_folder, model,
        scale=None, min_size=min_size,
        block_shape=block_shape, halo=halo,
    )
    postprocess_seg(args.segmentation_folder)

    # This function exports the segmentation and the corresponding channel to MoBIE.
    if args.mobie_dataset is None:
        mobie_dataset = Path(xml_path).stem
    else:
        mobie_dataset = args.mobie_dataset
    export_to_mobie(
        xml_path, args.segmentation_folder, args.output_folder, scale, mobie_dataset, chunks,
        channel_names=args.channel_names
    )

    # Finally, we clean up the intermediate segmentation results, that are not needed anymore
    # because everything was exported to MoBIE.
    print("Cleaning up intermediate segmentation results")
    print("This may take a while, but everything else is done.")
    print("You can check the results in the MoBIE project already at:")
    print(f"{args.output_folder}:{mobie_dataset}")
    rmtree(args.segmentation_folder)


if __name__ == "__main__":
    main()
