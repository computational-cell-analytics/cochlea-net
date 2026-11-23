flamingo_tools.export_data lower_resolution \
    --cochlea M_AMD_OTOF27_L \
    --s3 \
    --scale 3 \
    --output_folder otof27l-editing \
    --channels Otof IHC_v11 \
    --filter_cochlea_channels IHC_v11 \
    --filter_ihc_components 1 \
    --filter_dilation_iterations 3
