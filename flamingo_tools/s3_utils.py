"""This file contains utility functions for processing data located on an S3 storage.
The upload of data to the storage system should be performed with 'rclone'.
"""
import json
import os
import warnings
from shutil import which
from subprocess import run
from typing import Optional, Tuple

import s3fs
import zarr

try:
    from zarr.abc.store import Store
except ImportError:
    from zarr._storage.store import BaseStore as Store


# Dedicated bucket for cochlea lightsheet project
MOBIE_FOLDER = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
SERVICE_ENDPOINT = "https://s3.fs.gwdg.de/"
BUCKET_NAME = "cochlea-lightsheet"

DEFAULT_CREDENTIALS = os.path.expanduser("~/.aws/credentials")


def check_s3_credentials(
    bucket_name: Optional[str], service_endpoint: Optional[str], credential_file: Optional[str]
) -> Tuple[str, str, str]:
    """Check if S3 parameter and credentials were set as input arguments, as environment variables, or as globals.

    Args:
        bucket_name: S3 bucket name
        service_endpoint: S3 service endpoint
        credential_file: Credential file containing access key and secret key

    Returns:
        bucket_name
        service_endpoint
        credential_file
    """
    if bucket_name is None:
        bucket_name = os.getenv("BUCKET_NAME")
        if bucket_name is None:
            if "BUCKET_NAME" in globals():
                bucket_name = BUCKET_NAME
            else:
                raise ValueError(
                    "Provide a bucket name for accessing S3 data.\n"
                    "Either by using an optional argument or exporting an environment variable:\n"
                    "--s3_bucket_name <bucket_name>\n"
                    "export BUCKET_NAME=<bucket_name>"
                )

    if service_endpoint is None:
        service_endpoint = os.getenv("SERVICE_ENDPOINT")
        if service_endpoint is None:
            if "SERVICE_ENDPOINT" in globals():
                service_endpoint = SERVICE_ENDPOINT
            else:
                raise ValueError(
                    "Provide a service endpoint for accessing S3 data.\n"
                    "Either by using an optional argument or exporting an environment variable:\n"
                    "--s3_service_endpoint <endpoint>\n"
                    "export SERVICE_ENDPOINT=<endpoint>")

    if credential_file is None:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        # check for default credentials if no credential_file is provided
        if access_key is None:
            if os.path.isfile(DEFAULT_CREDENTIALS):
                access_key, _ = read_s3_credentials(credential_file=DEFAULT_CREDENTIALS)
            else:
                raise ValueError(
                    "Either provide a credential file as an optional argument,"
                    f" have credentials at '{DEFAULT_CREDENTIALS}',"
                    " or export an access key as an environment variable:\n"
                    "export AWS_ACCESS_KEY_ID=<access_key>")
        if secret_key is None:
            # check for default credentials
            if os.path.isfile(DEFAULT_CREDENTIALS):
                _, secret_key = read_s3_credentials(credential_file=DEFAULT_CREDENTIALS)
            else:
                raise ValueError(
                    "Either provide a credential file as an optional argument,"
                    f" have credentials at '{DEFAULT_CREDENTIALS}',"
                    " or export a secret access key as an environment variable:\n"
                    "export AWS_SECRET_ACCESS_KEY=<secret_key>")

    else:
        # check validity of credential file
        _, _ = read_s3_credentials(credential_file=credential_file)

    return bucket_name, service_endpoint, credential_file


def get_s3_path(
    input_path: str,
    bucket_name: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    credential_file: Optional[str] = None,
) -> Tuple[Store, s3fs.core.S3FileSystem]:
    """Get S3 path for a file or folder and file system based on S3 parameters and credentials.

    Args:
        input_path: Inputh path in S3 bucket
        bucket_name: S3 bucket name
        service_endpoint: S3 service endpoint
        credential_file: Credential file containing access key and secret key

    Returns:
        s3_path
        s3_filesystem
    """
    bucket_name, service_endpoint, credential_file = check_s3_credentials(
        bucket_name, service_endpoint, credential_file
    )

    zarr_major_version = int(zarr.__version__.split(".")[0])
    s3_filesystem = create_s3_target(
        url=service_endpoint, anon=False, credential_file=credential_file, asynchronous=zarr_major_version == 3,
    )

    zarr_path = f"{bucket_name}/{input_path}"

    if zarr_major_version == 2 and not s3_filesystem.exists(zarr_path):
        print(f"Error: S3 path {zarr_path} does not exist!")

    # The approach for opening a dataset from S3 differs in zarr v2 and zarr v3.
    if zarr_major_version == 2:
        s3_path = zarr.storage.FSStore(zarr_path, fs=s3_filesystem)
    elif zarr_major_version == 3:
        s3_path = zarr.storage.FsspecStore(fs=s3_filesystem, path=zarr_path)
    else:
        raise RuntimeError(f"Unsupported zarr version {zarr_major_version}")

    return s3_path, s3_filesystem


def read_s3_credentials(credential_file: str) -> Tuple[str, str]:
    """Read access key amd secret key from credential file.

    Args:
        credential_file: File path to credentials

    Returns:
        access_key
        secret_key
    """
    access_key, secret_key = None, None
    with open(credential_file) as f:
        for line in f:
            if line.startswith("aws_access_key_id"):
                access_key = line.rstrip("\n").strip().split(" ")[-1]
            if line.startswith("aws_secret_access_key"):
                secret_key = line.rstrip("\n").strip().split(" ")[-1]
    if access_key is None or secret_key is None:
        raise ValueError(f"Invalid credential file {credential_file}")
    return access_key, secret_key


def create_s3_target(
    url: Optional[str] = None,
    anon: Optional[str] = False,
    credential_file: Optional[str] = None,
    asynchronous: bool = False,
) -> s3fs.core.S3FileSystem:
    """Create file system for S3 bucket based on a service endpoint and an optional credential file.
    If the credential file is not provided, the s3fs.S3FileSystem function checks the environment variables
    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.

    Args:
        url: Service endpoint for S3 bucket
        anon: Option for anon argument of S3FileSystem
        credential_file: File path to credentials
        asynchronous: Whether to open the file system in async mode.

    Returns:
        s3_filesystem
    """
    client_kwargs = {"endpoint_url": SERVICE_ENDPOINT if url is None else url}
    if credential_file is not None:
        key, secret = read_s3_credentials(credential_file)
        s3_filesystem = s3fs.S3FileSystem(
            key=key, secret=secret, client_kwargs=client_kwargs, asynchronous=asynchronous
        )
    else:
        s3_filesystem = s3fs.S3FileSystem(anon=anon, client_kwargs=client_kwargs, asynchronous=asynchronous)
    return s3_filesystem


def _sync_rclone(local_dir, target):
    # The rclone alias could also be exposed as parameter.
    rclone_alias = "cochlea-lightsheet"
    print("Sync", local_dir, "to", target)
    run(["rclone", "--progress", "copyto", local_dir, f"{rclone_alias}:{target}"])


def sync_dataset(
    mobie_root: str,
    dataset_name: str,
    bucket_name: Optional[str] = None,
    url: Optional[str] = None,
    anon: Optional[str] = False,
    credential_file: Optional[str] = None,
    force_segmentation_update: bool = False,
) -> None:
    """Sync a MoBIE dataset on the s3 bucket using rclone.

    Args:
        mobie_root: The directory with the local MoBIE project.
        dataset_name: The MoBIE dataset to sync.
        bucket_name: The name of the dataset's bucket on s3.
        url: Service endpoint for S3 bucket
        anon: Option for anon argument of S3FileSystem
        credential_file: File path to credentials
        force_segmentation_update: Whether to force segmentation updates.
    """
    from mobie.metadata import add_remote_project_metadata

    # Make sure that rclone is loaded.
    if which("rclone") is None:
        raise RuntimeError("rclone is required for synchronization. Try loading it via 'module load rclone'.")

    # Make sure the dataset is in the local version of the dataset.
    with open(os.path.join(mobie_root, "project.json")) as f:
        project_metadata = json.load(f)
    datasets = project_metadata["datasets"]
    assert dataset_name in datasets

    # Get s3 filsystem and bucket name.
    s3 = create_s3_target(url, anon, credential_file)
    if bucket_name is None:
        bucket_name = BUCKET_NAME
    if url is None:
        url = SERVICE_ENDPOINT

    # Add the required remote metadata to the project. Suppress warnings about missing local data.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        add_remote_project_metadata(mobie_root, bucket_name, url)

    # Get the metadata from the S3 bucket.
    project_metadata_path = os.path.join(bucket_name, "project.json")
    with s3.open(project_metadata_path, "r") as f:
        project_metadata = json.load(f)

    # Check if the dataset is part of the remote project already.
    local_ds_root = os.path.join(mobie_root, dataset_name)
    remote_ds_root = os.path.join(bucket_name, dataset_name)
    if dataset_name not in project_metadata["datasets"]:
        print("The dataset is not yet synced. Will copy it over.")
        _sync_rclone(os.path.join(mobie_root, "project.json"), project_metadata_path)
        _sync_rclone(local_ds_root, remote_ds_root)
        return

    # Otherwise, check which sources are new and add them.
    with open(os.path.join(mobie_root, dataset_name, "dataset.json")) as f:
        local_dataset_metadata = json.load(f)

    dataset_metadata_path = os.path.join(bucket_name, dataset_name, "dataset.json")
    with s3.open(dataset_metadata_path, "r") as f:
        remote_dataset_metadata = json.load(f)

    for source_name, source_data in local_dataset_metadata["sources"].items():
        source_type, source_data = next(iter(source_data.items()))
        is_segmentation = source_type == "segmentation"
        is_spots = source_type == "spots"
        data_path = source_data["imageData"]["ome.zarr"]["relativePath"]
        source_not_on_remote = (source_name not in remote_dataset_metadata["sources"])
        # Only update the image data if the source is not updated or if we force updates for segmentations.
        if source_not_on_remote or (is_segmentation and force_segmentation_update):
            _sync_rclone(os.path.join(local_ds_root, data_path), os.path.join(remote_ds_root, data_path))
        # We always sync the tables.
        if is_segmentation or is_spots:
            table_path = source_data["tableData"]["tsv"]["relativePath"]
            _sync_rclone(os.path.join(local_ds_root, table_path), os.path.join(remote_ds_root, table_path))

    # Sync the dataset metadata.
    _sync_rclone(
        os.path.join(mobie_root, dataset_name, "dataset.json"), os.path.join(remote_ds_root, "dataset.json")
    )
