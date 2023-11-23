import os
import numpy as np
import label_studio_converter.brush as brush
from label_studio_sdk import Client

LABEL_STUDIO_URL = 'http://localhost:9001'
LABEL_STUDIO_API_KEY = 'ea837125b3838452872f0c277ef9774ba3f6e050'
LABEL = 'peak'

ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
ls.check_connection()


drc_data = "label_studio"
exp           = 'cxic00318'
run           = 123
events = [5100, 5177, 16987, 18789, 25243, 29400, 29741, 66893, 69304, 241841]
sigma_cuts = [0.2, 4, 6]
basename = f"{exp}_{run:04d}"

project = ls.start_project(
    title='peakdiff',
    label_config=f"""
    <View>
    <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
    <BrushLabels name="brush_labels_tag" toName="image">
    <Label value="peak" background="#D4380D"/><Label value="artifact scattering" background="#0433ff"/><Label value="bad pixel" background="#00f900"/></BrushLabels>
    </View>
    """,
)



for event in events:
    # Export label to hdf5 (when using requests is not allowed in the network)...····
    input_label_file = f"{basename}_{event:06d}.npy"
    input_label_path = os.path.join(drc_data, input_label_file)
    label = np.load(input_label_path)

    label = (label > 0).astype(
        np.uint8
    ) * 255  # better to threshold, it reduces output annotation size
    rle = brush.mask2rle(label)  # mask image in RLE format

    # Save the image as JPEG
    for sig_idx, sigma_cut in enumerate(sigma_cuts):
        # Create an image object and save it
        input_file = f"{basename}_{event:06d}.sig{sig_idx:01d}.jpeg"
        # input_jpeg_path = os.path.join(drc_data, input_file)

        ids = project.import_tasks(
            [{'image': f'/data/local-files/?d=data_root/pf/label_studio/{input_file}'}]
        )

        project.create_annotation(
        task_id=ids[0],
        model_version=None,
        result=[
            {
                "from_name": "brush_labels_tag",
                "to_name": "image",
                "type": "brushlabels",
                'value': {"format": "rle", "rle": rle, "brushlabels": [LABEL]},
            }
        ],
    )

# project.create_prediction(
#     task_id=ids[0],
#     model_version=None,
#     result=[
#         {
#             "from_name": "brush_labels_tag",
#             "to_name": "image",
#             "type": "brushlabels",
#             'value': {"format": "rle", "rle": rle, "brushlabels": [LABEL]},
#         }
#     ],
# )


