import json
import os

import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import create_s3_target, BUCKET_NAME, get_s3_path


# Note: downsampling with anisotropic scale in the beginning would make sense for better visualization.
def analyze_sgn(visualize=False):
    s3 = create_s3_target()
    datasets = ["LaVision-M04", "LaVision-Mar05"]

    # Use this to select the compoents for analysis.
    sgn_components = {
        "LaVision-M04": [1],
        "LaVision-Mar05": [1],
    }
    seg_name = "SGN_LOWRES-v2"

    for cochlea in datasets:
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the segmentation table.
        seg_source = sources[seg_name]
        table_folder = os.path.join(
            BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
        )
        table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        if visualize:
            import napari
            import zarr
            from bioimage_cpp.utils import take_dict

            key = "s2"
            img_s3 = f"{cochlea}/images/ome-zarr/PV.ome.zarr"
            seg_s3 = os.path.join(cochlea, seg_source["segmentation"]["imageData"]["ome.zarr"]["relativePath"])
            img_path, _ = get_s3_path(img_s3)
            seg_path, _ = get_s3_path(seg_s3)

            print("Loading image data")
            f = zarr.open(seg_path, mode="r")
            seg = f[key][:]

            seg_ids = np.unique(seg)
            component_dict = {int(label_id): int(component_id)
                              for label_id, component_id in zip(table.label_id, table.component_labels)}
            missing_ids = np.setdiff1d(seg_ids, table.label_id.values)
            component_dict.update({miss: 0 for miss in missing_ids})
            components = take_dict(component_dict, seg)

            f = zarr.open(img_path, mode="r")
            data = f[key][:]

            v = napari.Viewer()
            v.add_image(data)
            v.add_labels(seg)
            v.add_labels(components)
            napari.run()

        table = table[table.component_labels.isin(sgn_components[cochlea])]
        n_sgns = len(table)
        print(cochlea, ":", n_sgns)


def main():
    analyze_sgn(visualize=True)


if __name__ == "__main__":
    main()
