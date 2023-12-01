import os
import uuid
import json
from label_studio_sdk import Client

def coco_to_label_studio(coco_json_path, images_dir, output_json_path):
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)
    
    image_width = coco_data["images"][0]["width"]
    image_height = coco_data["images"][0]["height"]

    # Mapping from COCO category ID to category name
    category_mapping = {category['id']: category['name'] for category in coco_data['categories']}

    # Organize annotations by image
    annotations_by_image = {image['id']: [] for image in coco_data['images']}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(ann)

    label_studio_tasks = []
    for image_info in coco_data['images']:
        task = {
            'id': image_info['id'],
            'data': {'image': os.path.join(images_dir, image_info['file_name'])},
            'annotations': [{'result': []}]
        }

        for ann in annotations_by_image[image_info['id']]:
            label_name = category_mapping[ann['category_id']]
            for segmentation in ann['segmentation']:
                # polygon_points = [[segmentation[i], segmentation[i + 1]] for i in range(0, len(segmentation), 2)]
                polygon_points = [[segmentation[i+1] / image_width * 100.0, segmentation[i] / image_height * 100.0] for i in range(0, len(segmentation), 2)]
                label_data = {
                    "id": uuid.uuid4().hex[0:10],
                    'from_name': 'label',
                    'to_name': 'image',
                    'type': 'polygonlabels',
                    'value': {
                        'points': polygon_points,
                        'polygonlabels': [label_name]
                    },
                    'closed' : True,
                    "original_width": image_width,
                    "original_height": image_height,
                }
                task['annotations'][0]['result'].append(label_data)
        label_studio_tasks.append(task)

    with open(output_json_path, 'w') as file:
        json.dump(label_studio_tasks, file, indent=4)
    
    return label_studio_tasks

# Example usage
coco_json_path = 'coco_annotations.json'
images_dir = '/data/local-files/?d='
output_json_path = 'output_label_studio_annotations.json'

label_studio_tasks = coco_to_label_studio(coco_json_path, images_dir, output_json_path)


LABEL_STUDIO_URL = 'http://localhost:5008'
LABEL_STUDIO_API_KEY = 'ea837125b3838452872f0c277ef9774ba3f6e050'
LABEL = 'peak'

ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
ls.check_connection()

project = ls.start_project(
    title='peakdiff_test2',
    label_config=f"""
<View>
    <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false" contrastControl="true" brightnessControl="true"/>
    <Header value="Brush labels"/>
    <BrushLabels name="brush_labels_tag" toName="image" choice="single" showInline="true">
        <Label value="peak" background="#D4380D"/>
        <Label value="artifact scattering" background="#0433ff"/>
        <Label value="bad pixel" background="#00f900"/>
    </BrushLabels>
  
    <Header value="Polygon labels"/>
    <PolygonLabels name="label" toName="image">
        <Label value="peak" background="#D4380D"/>
        <Label value="artifact scattering" background="#0433ff"/>
        <Label value="bad pixel" background="#00f900"/>
    </PolygonLabels>
    <Polygon name="polygon" toName="image" />
</View>
    """,
)

ids = project.import_tasks(
    label_studio_tasks
)