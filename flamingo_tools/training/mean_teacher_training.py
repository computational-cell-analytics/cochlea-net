from typing import Optional, Tuple

import torch
import torch_em
import torch_em.self_training as self_training
from torchvision import transforms

from ..model_utils import load_model_from_path
from .util import get_supervised_loader, get_3d_model


def weak_augmentations(p: float = 0.75) -> callable:
    """The weak augmentations used in the unsupervised data loader.

    Args:
        p: The probability for applying one of the augmentations.

    Returns:
        The transformation function applying the augmentation.
    """
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)


def get_unsupervised_loader(
    data_paths: Tuple[str],
    raw_key: Optional[str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
) -> torch.utils.data.DataLoader:
    """Get a dataloader for unsupervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.

    Returns:
        The PyTorch dataloader.
    """
    raw_transform = torch_em.transform.get_raw_transform()
    transform = torch_em.transform.get_augmentations(ndim=3)

    if n_samples is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples / len(data_paths))

    augmentations = (weak_augmentations(), weak_augmentations())
    datasets = [
        torch_em.data.RawDataset(path, raw_key, patch_shape, raw_transform, transform,
                                 augmentations=augmentations, ndim=3, n_samples=n_samples_per_ds)
        for path in data_paths
    ]
    ds = torch.utils.data.ConcatDataset(datasets)

    # num_workers = 4 * batch_size
    num_workers = batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader


def mean_teacher_training(
    name: str,
    unsupervised_train_paths: Tuple[str],
    unsupervised_val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    save_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
    supervised_train_image_paths: Optional[Tuple[str]] = None,
    supervised_val_image_paths: Optional[Tuple[str]] = None,
    supervised_train_label_paths: Optional[Tuple[str]] = None,
    supervised_val_label_paths: Optional[Tuple[str]] = None,
    confidence_threshold: float = 0.9,
    raw_key: Optional[str] = None,
    raw_key_supervised: Optional[str] = None,
    label_key: Optional[str] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    sampler: Optional[callable] = None,
) -> None:
    """This function implements network training with a mean teacher approach.

    It can be used for semi-supervised learning, unsupervised domain adaptation and supervised domain adaptation.
    These different training modes can be used as this:
    - semi-supervised learning: pass 'unsupervised_train/val_paths' and 'supervised_train/val_paths'.
    - unsupervised domain adaptation: pass 'unsupervised_train/val_paths' and 'source_checkpoint'.
    - supervised domain adaptation: pass 'unsupervised_train/val_paths', 'supervised_train/val_paths', 'source_checkpoint'.

    Args:
        name: The name for the checkpoint to be trained.
        unsupervsied_train_paths: Filepaths to the hdf5 files or similar file formats
            for the training data in the target domain.
            This training data is used for unsupervised learning, so it does not require labels.
        unsupervised_val_paths: Filepaths to the hdf5 files or similar file formats
            for the validation data in the target domain.
            This validation data is used for unsupervised learning, so it does not require labels.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        source_checkpoint: Checkpoint to the initial model trained on the source domain.
            This is used to initialize the teacher model.
            If the checkpoint is not given, then both student and teacher model are initialized
            from scratch. In this case `supervised_train_paths` and `supervised_val_paths` have to
            be given in order to provide training data from the source domain.
        supervised_train_image_paths: Paths to the files for the supervised image data; training split.
            This training data is optional. If given, it also requires labels.
        supervised_val_image_paths: Ppaths to the files for the supervised image data; validation split.
            This validation data is optional. If given, it also requires labels.
        supervised_train_label_paths: Filepaths to the files for the supervised label masks; training split.
            This training data is optional.
        supervised_val_label_paths: Filepaths to the files for the supervised label masks; validation split.
            This tvalidation data is optional.
        confidence_threshold: The threshold for filtering data in the unsupervised loss.
            The label filtering is done based on the uncertainty of network predictions, and only
            the data with higher certainty than this threshold is used for training.
        raw_key: The key that holds the raw data inside of the hdf5 or similar files;
            for the unsupervised training data. Set to None for tifs.
        raw_key_supervised: The key that holds the raw data inside of the hdf5 or similar files;
            for the supervised training data. Set to None for tifs.
        label_key: The key that holds the labels inside of the hdf5 files for supervised learning.
            This is only required if `supervised_train_label_paths` and `supervised_val_label_paths` are given.
            Set to None for tifs.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
    """  # noqa
    assert (supervised_train_image_paths is None) == (supervised_val_image_paths is None)

    if source_checkpoint is None:
        # Training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert supervised_train_image_paths is not None
        model = get_3d_model(out_channels=3)
        reinit_teacher = True
    else:
        print("Mean teacehr training initialized from source model:", source_checkpoint)
        model = load_model_from_path(source_checkpoint)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # self training functionality
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold, mask_channel=0)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    unsupervised_train_loader = get_unsupervised_loader(
        unsupervised_train_paths, raw_key, patch_shape, batch_size, n_samples=n_samples_train
    )
    unsupervised_val_loader = get_unsupervised_loader(
        unsupervised_val_paths, raw_key, patch_shape, batch_size, n_samples=n_samples_val
    )

    if supervised_train_image_paths is not None:
        supervised_train_loader = get_supervised_loader(
            supervised_train_image_paths, supervised_train_label_paths,
            patch_shape=patch_shape, batch_size=batch_size, n_samples=n_samples_train,
            image_key=raw_key_supervised, label_key=label_key,
        )
        supervised_val_loader = get_supervised_loader(
            supervised_val_image_paths, supervised_val_label_paths,
            patch_shape=patch_shape, batch_size=batch_size, n_samples=n_samples_val,
            image_key=raw_key_supervised, label_key=label_key,
        )
    else:
        supervised_train_loader = None
        supervised_val_loader = None

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=save_root,
        sampler=sampler,
    )
    trainer.fit(n_iterations)
