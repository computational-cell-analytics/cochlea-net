"""Export the CochleaNet SGN, IHC, synapse, and low-resolution networks as BioImage Model Zoo packages.

Run in the cochlea-net environment from the repository root:
    python scripts/bioimageio/export_models.py [KEY ...]
KEY is a key of MODELS. Without a key, the script builds all packages.
The packages go to models/bioimageio/<key>.

The packages are published as version v0 on BioImage.IO (see MODELS in flamingo_tools/model_utils.py):
sgn = knowledgeable-mouse, ihc = content-fish, synapses = loyal-ant, sgn-lowres = nervous-shark,
ihc-lowres = confused-cow. They were built with torch 2.12.1, torch_em 0.10.1 (source of unet.py) and
bioimageio.spec 0.5.10.2. The training checkpoints in models/ are not public.
A rebuild with these versions reproduces the published files byte for byte.
"""
import os
import shutil
import sys
import tempfile

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch_em.model.unet
from bioimageio.spec import save_bioimageio_package_as_folder

from flamingo_tools.segmentation.unet_prediction import _load_model
from flamingo_tools.test_data import _sample_registry

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUTPUT_ROOT = os.path.join(ROOT, "models", "bioimageio")
README_ROOT = os.path.dirname(__file__)

VOXEL_SIZE = 0.38  # um, isotropic
LAVISION_VOXEL_SIZE = (3.0, 1.887779, 1.887779)  # um, z, y, x

DISTANCE_TRAINING = {
    "training_preprocessing": "Per-patch standardization. Random flips along all three axes.",
    "training_batch_size": 8,
    "initial_learning_rate": 1e-4,
    "learning_rate_schedule": "ReduceLROnPlateau",
    "loss_function": "torch_em.loss.distance_based.DiceBasedDistanceLoss",
    "loss_function_kwargs": {"mask_distances_in_bg": True},
    "optimizer": "torch.optim.AdamW",
}

# The BioEngine GPU (NVIDIA A40) reproduced the low-res test outputs only within 1.4e-2 (0.6% of the elements
# above the default tolerance of 1e-3); the 0.38 um models stay within it. The outputs are in [0, 1].
GPU_TOLERANCE = [{"relative_tolerance": 0.01, "absolute_tolerance": 0.01}]

# The checkpoints differ from the zoo version, so the internal names only go to config.
MODELS = {
    "sgn": {
        "checkpoint": "v2_cochlea_distance_unet_SGN_supervised_2025-05-27",
        "timestamp": "2025-05-27T22:06:49",
        "name": "CochleaNet SGN Segmentation",
        "description": (
            "3D U-Net for the instance segmentation of spiral ganglion neurons (SGNs) in light-sheet "
            "microscopy of cleared cochleae, stained for parvalbumin (PV) or calretinin (CR). "
            "It predicts the foreground probability, the distance to the object center, and the distance "
            "to the object boundary. A seeded watershed turns these maps into SGN instances. "
            "Trained on annotated PV-stained mouse cochleae at 0.38 um isotropic voxel size."
        ),
        "tags": ["mouse", "instance-segmentation", "neurons", "spiral-ganglion-neurons", "parvalbumin", "calretinin"],
        "target_structure": ["spiral ganglion neurons"],
        "task": "instance segmentation",
        "sample": "PV",
        "channel": "PV",
        "outputs": ("distances", ["foreground", "center_distance", "boundary_distance"]),
        "step": (16, 16, 16), "min": (48, 80, 80), "halo": (16, 32, 32),
        "training": DISTANCE_TRAINING,
    },
    "ihc": {
        "checkpoint": "v11_cochlea_distance_unet_IHC_supervised_2026-07-20",
        "timestamp": "2026-07-21T14:17:19",
        "name": "CochleaNet IHC Segmentation",
        "description": (
            "3D U-Net for the instance segmentation of inner hair cells (IHCs) in light-sheet microscopy of "
            "cleared cochleae, stained for VGlut3 or Myosin VIIa. "
            "It predicts the foreground probability, the distance to the object center, and the distance "
            "to the object boundary. A seeded watershed turns these maps into IHC instances. "
            "Trained on annotated VGlut3-stained mouse and gerbil cochleae at 0.38 um isotropic voxel size."
        ),
        "tags": ["mouse", "gerbil", "instance-segmentation", "hair-cells", "inner-hair-cells", "vglut3", "myosin-viia"],
        "target_structure": ["inner hair cells"],
        "task": "instance segmentation",
        "sample": "VGlut3",
        "channel": "VGlut3",
        "outputs": ("distances", ["foreground", "center_distance", "boundary_distance"]),
        "step": (16, 16, 16), "min": (48, 80, 80), "halo": (16, 32, 32),
        "training": DISTANCE_TRAINING,
    },
    "synapses": {
        "checkpoint": "synapse_detection_model_v3.pt",
        "timestamp": "2025-06-15T00:00:00",
        "name": "CochleaNet Ribbon Synapse Detection",
        "description": (
            "3D U-Net for the detection of ribbon synapses in light-sheet microscopy of cleared cochleae, "
            "stained for CtBP2. It predicts a heatmap with a Gaussian peak at each synapse. "
            "Local maxima of the heatmap give the synapse positions. "
            "Trained on annotated CtBP2-stained cochleae at 0.38 um isotropic voxel size."
        ),
        "tags": ["detection", "synapses", "ribbon-synapses", "ctbp2"],
        "target_structure": ["ribbon synapses"],
        "task": "object detection",
        "sample": "CTBP2",
        "channel": "CtBP2",
        "outputs": ("heatmap", ["synapse_heatmap"]),
        "step": (8, 16, 16), "min": (40, 144, 144), "halo": (16, 64, 64),
        "training": {
            "training_preprocessing": "None. The network starts with an instance normalization.",
            "training_batch_size": 32,
            "initial_learning_rate": 1e-4,
            "loss_function": "torch.nn.MSELoss",
        },
    },
    "sgn-lowres": {
        # latest.pt and not best.pt (iteration 1,250): the same weights as flamingo_tools.get_model("SGN-lowres").
        "checkpoint": "cochlea_distance_unet_sgn-low-res-v5/latest.pt",
        "timestamp": "2025-10-01T15:42:15",
        "name": "CochleaNet Low-Resolution SGN Segmentation",
        "description": (
            "Anisotropic 3D U-Net for the instance segmentation of spiral ganglion neurons (SGNs) in "
            "lower-resolution light-sheet microscopy of cleared cochleae, stained for parvalbumin (PV). "
            "It predicts the foreground probability, the distance to the object center, and the distance "
            "to the object boundary. A seeded watershed turns these maps into SGN instances. "
            "Trained on annotated PV-stained mouse and marmoset cochleae from a LaVision UltraMicroscope at "
            "1.887779 x 1.887779 x 3.0 um voxel size (x, y, z)."
        ),
        "tags": ["mouse", "marmoset", "instance-segmentation", "neurons", "spiral-ganglion-neurons", "parvalbumin",
                 "anisotropic"],
        "target_structure": ["spiral ganglion neurons"],
        "task": "instance segmentation",
        "sample": "PV-lowres",
        "channel": "PV",
        "voxel_size": LAVISION_VOXEL_SIZE,
        "outputs": ("distances", ["foreground", "center_distance", "boundary_distance"]),
        "step": (8, 16, 16), "min": (24, 80, 80), "halo": (8, 32, 32),
        "training": {**DISTANCE_TRAINING, "training_batch_size": 16},
        "tolerance": GPU_TOLERANCE,
    },
    "ihc-lowres": {
        "checkpoint": "cochlea_distance_unet_ihc-lowres-v3",
        "timestamp": "2025-09-18T23:37:33",
        "name": "CochleaNet Low-Resolution IHC Segmentation",
        "description": (
            "3D U-Net for the instance segmentation of inner hair cells (IHCs) in lower-resolution light-sheet "
            "microscopy of cleared cochleae, stained for Myosin VIIa, calretinin (CR) or VGlut3. "
            "It predicts the foreground probability, the distance to the object center, and the distance "
            "to the object boundary. A seeded watershed turns these maps into IHC instances. "
            "Trained on annotated mouse and marmoset cochleae at about 1.89 x 1.89 x 3.0 um voxel size (x, y, z)."
        ),
        "tags": ["mouse", "marmoset", "instance-segmentation", "hair-cells", "inner-hair-cells", "myosin-viia",
                 "calretinin", "vglut3", "anisotropic"],
        "target_structure": ["inner hair cells"],
        "task": "instance segmentation",
        "sample": "MYO-lowres",
        "channel": "MyosinVIIa",
        "voxel_size": LAVISION_VOXEL_SIZE,
        "outputs": ("distances", ["foreground", "center_distance", "boundary_distance"]),
        "step": (16, 16, 16), "min": (32, 80, 80), "halo": (8, 32, 32),
        "training": {**DISTANCE_TRAINING, "training_batch_size": 16},
        "tolerance": GPU_TOLERANCE,
    },
}

# The numbers follow the affiliation list of the CochleaNet preprint.
AFFILIATIONS = {
    1: "Institute for Auditory Neuroscience and InnerEarLab, University Medical Center Göttingen, Göttingen, Germany",
    2: "Department of Otolaryngology, University Medical Center Göttingen, Göttingen, Germany",
    3: 'Cluster of Excellence "Multiscale Bioimaging: from Molecular Machines to Networks of Excitable Cells" '
       "(MBExC), University of Göttingen, Göttingen, Germany",
    4: "Else-Kröner-Fresenius Center for Optogenetic Therapies, University Medical Center Göttingen, Göttingen, "
       "Germany",
    5: "Auditory Neuroscience & Synaptic Nanophysiology Group, Max-Planck-Institute for Multidisciplinary Sciences, "
       "Göttingen, Germany",
    6: "Göttingen Graduate School for Neurosciences and Molecular Biosciences, University of Göttingen, Göttingen, "
       "Germany",
    7: "Institute of Computer Science, University of Göttingen, Göttingen, Germany",
    8: 'Collaborative Research Center CRC 1690 "Disease Mechanisms and Functional Restoration of Sensory and Motor '
       'Systems", University of Göttingen, 37075, Göttingen, Germany',
    9: "Functional Auditory Genomics Group, Auditory Neuroscience and Optogenetics laboratory, German Primate Center, "
       "Göttingen, Germany",
    10: "Multiscale Biology, Department of Biology and Psychology, University of Göttingen, Göttingen, Germany",
    11: "CAIMed - Lower Saxony Center for Artificial Intelligence and Causal Methods in Medicine, Göttingen",
    12: "German Center for Child and Adolescent Health (DZKJ), partner site Göttingen, Göttingen, Germany",
    13: "Machine Intelligence in the Life Sciences Group, Max-Planck-Institute for Multidisciplinary Sciences, "
        "Göttingen, Germany",
}
AUTHORS = [
    {"name": name, "orcid": orcid, "affiliation": "; ".join(AFFILIATIONS[i] for i in affiliations), **extra}
    for name, orcid, affiliations, extra in [
        ("Lennart Roos", "0009-0007-8051-7964", (1, 2, 3, 4), {}),
        ("Aleyna Miraç Diniz", "0009-0002-5372-8639", (1, 3, 5, 6, 8), {}),
        ("Elisabeth Koert", "0000-0001-8040-9444", (1, 3, 6), {}),
        ("Martin Schilling", "0000-0003-3639-3187", (3, 7, 8),
         {"email": "martin.schilling@med.uni-goettingen.de", "github_user": "schilling40"}),
        ("Mara Uhl", "0009-0009-4340-1422", (1, 4, 6, 9), {}),
        ("Anupriya Thirumalai", "0000-0002-3624-0343", (1, 3), {}),
        ("Mostafa Aakhte", "0000-0003-4685-5576", (10,), {}),
        ("Kathrin Kusch", "0000-0002-5079-502X", (1, 4, 8, 9), {}),
        ("Jan Huisken", "0000-0001-7250-3756", (3, 10), {}),
        ("Tobias Moser", "0000-0001-7145-0533", (1, 2, 3, 4, 5, 8, 11, 12), {}),
        ("Constantin Pape", "0000-0001-6562-7187", (3, 7, 8, 11, 13),
         {"email": "constantin.pape@informatik.uni-goettingen.de", "github_user": "constantinpape"}),
    ]
]
MAINTAINERS = [author for author in AUTHORS if author["name"] == "Martin Schilling"]
CITE = [
    {
        "text": "Roos et al. CochleaNet: deep learning-based image analysis for cochlear connectomics and gene "
                "therapy. bioRxiv 2025.",
        "doi": "10.1101/2025.11.16.688700",
    },
    {
        "text": "Roos et al. CochleaNet training data for SGN segmentation, IHC segmentation and ribbon "
                "synapse detection. Zenodo 2026.",
        "doi": "10.5281/zenodo.23478267",
    },
    {
        "text": "Cicek et al. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI 2016.",
        "doi": "10.1007/978-3-319-46723-8_49",
    },
]


def _space_axes(sizes, voxel_size, halos=None):
    axes = []
    for i, (ax, size, scale) in enumerate(zip("zyx", sizes, voxel_size)):
        axis = {"type": "space", "id": ax, "size": size, "unit": "micrometer", "scale": scale}
        if halos is not None:
            axis["halo"] = halos[i]
        axes.append(axis)
    return axes


def _to_rgb(image, cmap):
    lo, hi = np.percentile(image, [1, 99.5])
    return plt.get_cmap(cmap)(np.clip((image - lo) / (hi - lo + 1e-6), 0, 1))[..., :3]


def _write_cover(raw, prediction, path, max_projection):
    if max_projection:
        raw, prediction = raw.max(axis=0), prediction.max(axis=0)
    else:
        raw, prediction = raw[raw.shape[0] // 2], prediction[prediction.shape[0] // 2]
    cover = np.concatenate([_to_rgb(raw, "gray"), _to_rgb(prediction, "magma")], axis=1)
    plt.imsave(path, cover)


def export_model(key, cfg, tmp):
    model = _load_model(os.path.join(ROOT, "models", cfg["checkpoint"]), device="cpu").eval()
    # Class-valued kwargs (conv_block_impl) are the defaults and cannot go into YAML.
    kwargs = {k: v for k, v in model.init_kwargs.items() if not isinstance(v, type)}

    weights_path, arch_path = os.path.join(tmp, "weights.pt"), os.path.join(tmp, "unet.py")
    torch.save(model.state_dict(), weights_path)
    shutil.copyfile(torch_em.model.unet.__file__, arch_path)

    raw = imageio.imread(_sample_registry().fetch(cfg["sample"]), extension=".tif").astype("float32")
    test_input = raw[None, None]
    # Same as the declared preprocessing: zero_mean_unit_variance over z, y, x with eps=1e-6.
    normalized = (test_input - test_input.mean()) / (test_input.std() + 1e-6)
    with torch.no_grad():
        test_output = model(torch.from_numpy(normalized)).numpy()
    input_path, output_path = os.path.join(tmp, "test_input.npy"), os.path.join(tmp, "test_output.npy")
    np.save(input_path, test_input)
    np.save(output_path, test_output)
    sample_input_path = os.path.join(tmp, "sample_input.tif")
    sample_output_path = os.path.join(tmp, "sample_output.tif")
    tifffile.imwrite(sample_input_path, test_input, compression="zlib")
    tifffile.imwrite(sample_output_path, test_output, compression="zlib")

    cover_path = os.path.join(tmp, "cover.png")
    _write_cover(raw, test_output[0, 0], cover_path, max_projection=key == "synapses")
    readme_path = os.path.join(tmp, "README.md")
    shutil.copyfile(os.path.join(README_ROOT, f"{key}_README.md"), readme_path)

    output_id, channel_names = cfg["outputs"]
    output_range = [0.0, 1.0] if kwargs["final_activation"] == "Sigmoid" else [None, None]
    voxel_size = cfg.get("voxel_size", (VOXEL_SIZE,) * 3)
    z, y, x = voxel_size
    voxel_text = f"{x} um" if len(set(voxel_size)) == 1 else f"{x} x {y} x {z} um (x, y, z)"
    rdf = {
        "type": "model",
        "format_version": "0.5.10",
        "name": cfg["name"],
        "description": cfg["description"],
        "version": "1",
        "authors": AUTHORS,
        "maintainers": MAINTAINERS,
        "cite": CITE,
        "license": "MIT",
        "git_repo": "https://github.com/computational-cell-analytics/flamingo-tools",
        "tags": ["3d", "light-sheet", "fluorescence-light-microscopy", "cochlea", "unet", "pytorch",
                 "cochlea-net"] + cfg["tags"],
        "documentation": readme_path,
        "covers": [cover_path],
        "timestamp": cfg["timestamp"],
        "inputs": [{
            "id": "raw",
            "description": f"Light-sheet volume of the {cfg['channel']} channel at {voxel_text} voxel size.",
            "axes": [{"type": "batch"}, {"type": "channel", "channel_names": [cfg["channel"]]}] + _space_axes(
                [{"min": mn, "step": st} for mn, st in zip(cfg["min"], cfg["step"])], voxel_size
            ),
            "test_tensor": {"source": input_path},
            "sample_tensor": {"source": sample_input_path},
            "preprocessing": [
                {"id": "ensure_dtype", "kwargs": {"dtype": "float32"}},
                {"id": "zero_mean_unit_variance", "kwargs": {"axes": ["z", "y", "x"]}},
            ],
        }],
        "outputs": [{
            "id": output_id,
            "axes": [{"type": "batch"}, {"type": "channel", "channel_names": channel_names}] + _space_axes(
                [{"tensor_id": "raw", "axis_id": ax} for ax in "zyx"], voxel_size, halos=cfg["halo"]
            ),
            "test_tensor": {"source": output_path},
            "sample_tensor": {"source": sample_output_path},
            "data": {"type": "float32", "range": output_range},
        }],
        "weights": {"pytorch_state_dict": {
            "source": weights_path,
            "architecture": {"source": arch_path, "callable": type(model).__name__, "kwargs": kwargs},
            "pytorch_version": torch.__version__.split("+")[0],
        }},
        "config": {
            "bioimageio": {
                "architecture_type": "3D U-Net",
                "modality": "light-sheet fluorescence microscopy of cleared tissue",
                "target_structure": cfg["target_structure"],
                "task": cfg["task"],
                "model_parameter_count": sum(p.numel() for p in model.parameters()),
                "training": cfg["training"],
                **({"reproducibility_tolerance": cfg["tolerance"]} if "tolerance" in cfg else {}),
            },
            "cochlea_net": {"source_checkpoint": cfg["checkpoint"]},
        },
    }

    out = os.path.join(OUTPUT_ROOT, key)
    shutil.rmtree(out, ignore_errors=True)
    save_bioimageio_package_as_folder(rdf, output_path=out)
    print("Exported", key, "to", out)


def main():
    # A rebuild changes the files of an uploaded package, so build only the requested keys.
    for key in sys.argv[1:] or MODELS:
        with tempfile.TemporaryDirectory() as tmp:
            export_model(key, MODELS[key], tmp)


if __name__ == "__main__":
    main()
