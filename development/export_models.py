import torch
from torch_em.util import load_model


def export_sgn():
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/v2_cochlea_distance_unet_SGN_supervised_2025-05-27"  # noqa
    model = load_model(path, device="cpu")
    torch.save(model, "SGN.pt")


def export_ihc():
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v4_cochlea_distance_unet_IHC_supervised_2025-07-14"  # noqa
    model = load_model(path, device="cpu")
    torch.save(model, "IHC.pt")


def export_synapses():
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v3.pt"  # noqa
    model = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(model, "Synapses.pt")


def export_sgn_lowres():
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/cochlea_distance_unet_sgn-low-res-v4"  # noqa
    model = load_model(path, device="cpu")
    torch.save(model, "SGN-lowres.pt")


def export_ihc_lowres():
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/cochlea_distance_unet_ihc-lowres-v3"  # noqa
    model = load_model(path, device="cpu")
    torch.save(model, "IHC-lowres.pt")


def main():
    # export_sgn()
    # export_ihc()
    # export_synapses()
    export_sgn_lowres()
    # export_ihc_lowres()


if __name__ == "__main__":
    main()
