from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import json

class Annotation(object):
    region_attributes = "region_attributes"
    region_shape_attributes = "region_shape_attributes"
    region_count = "region_count"
    filename = "filename"
    file_attributes = "file_attributes"
    file_size = "file_size"
    region_id = "region_id"

class Attributes:
    filename: str
    skin_tone: int
    smiling: bool
    color: str
    frontal: bool

    def __init__(self, filename, skin_tone, smiling, color, frontal):
        self.filename = filename
        self.skin_tone = skin_tone
        self.smiling = smiling
        self.color = color
        self.frontal = frontal

    def as_tuple(self):
        return self.filename, self.skin_tone, self.smiling, self.color, self.frontal

    def __str__(self):
        return str(self.as_tuple())

ANNOTATIONS_FILE_PATH = Path("data/annotations/annotations.csv")
INPUT_FILE_BASE_PATH = Path("data/raw/1")
PROCESSED_PATH = Path("data/processed")
PROCESSED_IMG_BASE_PATH = PROCESSED_PATH / "images"
PROCESSED_CROP_SIZE = (224, 224)

def pad_to_square_dimensions(x, y, width, height):
    '''
    If image dimensions is not a square, add padding to make it so
    '''
    if width > height:
        diff = width - height
        y -= diff // 2
        height += diff
    elif height > width:
        diff = height - width
        x -= diff // 2
        width += diff

    return x, y, width, height

def crop_and_save_image(row):
    '''
    Get face coordinates and crop face from original image
    '''
    region_shape_attributes = row.loc[Annotation.region_shape_attributes]
    region_shape_attributes = json.loads(region_shape_attributes)

    if len(region_shape_attributes.keys()) == 0:
        return

    _, x, y, width, height = region_shape_attributes.values()

    x, y, width, height = pad_to_square_dimensions(x, y, width, height)

    path = INPUT_FILE_BASE_PATH / row.loc[Annotation.filename] 
    new_filename = path.stem + "_" + str(row.loc[Annotation.region_id]) + path.suffix

    with Image.open(path) as im:
        cropped = im.crop((x, y, x + width, y + height))

        if cropped.mode in ("RGBA", "P"):
            cropped = cropped.convert('RGB')

        cropped = cropped.resize(PROCESSED_CROP_SIZE)

        cropped.save(PROCESSED_IMG_BASE_PATH / new_filename, 'JPEG')
        
    print(f'Saved image crop: {new_filename}')
    return new_filename

def get_attributes(row):
    '''
    Get relevant information in a concise and easy to use manner, namely filename, skin_tone, 
    whether subject is smiling, color pixel value (if provided) and whether person is facing camera
    '''
    attr = row.loc[Annotation.region_attributes]
    attr = json.loads(attr)

    if len(attr.keys()) == 0:
        return None

    path = Path(row.loc[Annotation.filename])
    new_filename = path.stem + "_" + str(row.loc[Annotation.region_id]) + path.suffix

    skin_tone = int(attr["skin_tone"])
    expression = attr["expression"]
    smiling = expression["smiling"] == True if "smiling" in expression else False
    clarity = attr["clarity"]
    frontal = clarity["frontal"] == True if "frontal" in clarity else False
    color = attr["color"]

    return Attributes(new_filename, skin_tone, smiling, color, frontal)


def process_data(path):
    '''
    Full pipeline of looping through rows in the csv file, cropping faces and extracting labels and attributes
    '''
    annotations = pd.read_csv(path)
    
    data_gen = annotations.iterrows()
    
    attributes_list: list[Attributes] = []

    for i, row in data_gen:
        attr = get_attributes(row)
        if attr is not None:
            attributes_list.append(attr)

        crop_and_save_image(row)


    attributes_df = pd.DataFrame(
        map(
            lambda attr: attr.as_tuple(), attributes_list), 
            columns=("filename", "skin_tone", "smiling", "color", "frontal"))

    attributes_df.to_csv(PROCESSED_PATH / "attributes.csv", index=False)

if __name__ == "__main__":
    process_data(ANNOTATIONS_FILE_PATH)
