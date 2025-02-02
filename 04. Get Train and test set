# GET 1000 IMAGES IN TRAIN AND TEST SETS
import random
from sklearn.model_selection import train_test_split
import os
import requests
import csv
from pycocotools.coco import COCO
from google.colab import drive

# Define paths to annotation files in Google Drive
instances_train_ann_file = '/content/drive/MyDrive/abcd/coco_dataset/annotations/instances_train2017.json'
captions_train_ann_file = '/content/drive/MyDrive/abcd/coco_dataset/annotations/captions_train2017.json'

# Initialize COCO objects for instances and captions
coco_instances_train = COCO(instances_train_ann_file)
coco_captions_train = COCO(captions_train_ann_file)

# Get category IDs for 'bus' and 'sports ball'
categories = coco_instances_train.loadCats(coco_instances_train.getCatIds())
category_name_to_id = {cat['name']: cat['id'] for cat in categories}
bus_id = category_name_to_id.get('bus')
sports_ball_id = category_name_to_id.get('sports ball')

# Function to get image info and captions for specific categories
# Function to get image info and captions, keeping only the first sentence up to the first full stop
def get_image_info_and_captions(coco_instance, coco_caption, category_id, num_samples):
    img_ids = coco_instance.getImgIds(catIds=[category_id])
    random.shuffle(img_ids)

    images_info = []
    count = 0

    for img_id in img_ids:
        img_info = coco_instance.loadImgs(img_id)[0]
        ann_ids = coco_caption.getAnnIds(imgIds=img_id)
        anns = coco_caption.loadAnns(ann_ids)

        # Take only the first caption and trim it up to the first full stop
        if anns:
            first_caption = anns[0]['caption']
            trimmed_prompt = first_caption.split('.')[0] + '.'

            images_info.append({
                'id': img_info['id'],
                'file_name': img_info['file_name'],
                'url': img_info['coco_url'],
                'prompt': trimmed_prompt
            })
            count += 1

            # Debug: Print every collected image info to track the collection process
            print(f"Collected {count}/{num_samples} images: Category ID {category_id} - {img_info['file_name']} with prompt: {trimmed_prompt}")

        if len(images_info) >= num_samples:
            break

    return images_info



# Get image info and captions for 'bus' and 'sports ball'
train_images_info_bus = get_image_info_and_captions(coco_instances_train, coco_captions_train, bus_id, num_samples=500)
train_images_info_sports_ball = get_image_info_and_captions(coco_instances_train, coco_captions_train, sports_ball_id, num_samples=500)

# Combine train and validation info for splitting
all_images_info = train_images_info_bus + train_images_info_sports_ball

# Split into train and test sets
train_images_info, test_images_info = train_test_split(all_images_info, test_size=0.2, random_state=42)

# Define paths to save images and prompts in Google Drive

train_image_folder = '/content/drive/MyDrive/abcd/coco_images/train'
test_image_folder = '/content/drive/MyDrive/abcd/coco_images/test'
train_prompt_folder = '/content/drive/MyDrive/abcd/coco_prompts/train'
test_prompt_folder = '/content/drive/MyDrive/abcd/coco_prompts/test'

# Function to download images and write prompts
def download_images_and_prompts(image_info_list, image_folder, prompt_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(prompt_folder):
        os.makedirs(prompt_folder)

    for img_info in image_info_list:
        # Download image
        img_data = requests.get(img_info['url']).content
        img_path = os.path.join(image_folder, img_info['file_name'])
        with open(img_path, 'wb') as handler:
            handler.write(img_data)

        # Write prompt to a text file
        prompt_path = os.path.join(prompt_folder, f"{img_info['id']}.txt")
        with open(prompt_path, 'w') as handler:
            handler.write(img_info['prompt'])

# Download train and test images and prompts
download_images_and_prompts(train_images_info, train_image_folder, train_prompt_folder)
download_images_and_prompts(test_images_info, test_image_folder, test_prompt_folder)

# Function to write CSV files with slno column
def write_csv_with_slno(filename, data, fields):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['slno'] + fields)
        writer.writeheader()
        for idx, row in enumerate(data, start=1):
            row_with_slno = {'slno': idx, **row}
            writer.writerow(row_with_slno)

# Define paths to save CSV files in Google Drive
train_prompts_csv = '/content/drive/MyDrive/abcd/train_prompts.csv'
test_prompts_csv = '/content/drive/MyDrive/abcd/test_prompts.csv'
train_images_csv = '/content/drive/MyDrive/abcd/train_images.csv'
test_images_csv = '/content/drive/MyDrive/abcd/test_images.csv'

# Create train and test prompts CSV files with slno
train_prompts = [{'id': img_info['id'], 'prompt': img_info['prompt']} for img_info in train_images_info]
test_prompts = [{'id': img_info['id'], 'prompt': img_info['prompt']} for img_info in test_images_info]

# Debug: Print the lengths of train and test prompts to verify counts
print(f"Number of train prompts: {len(train_prompts)}")
print(f"Number of test prompts: {len(test_prompts)}")

write_csv_with_slno(train_prompts_csv, train_prompts, ['id', 'prompt'])
write_csv_with_slno(test_prompts_csv, test_prompts, ['id', 'prompt'])

# Create train and test images CSV files with slno
train_images = [{'id': img_info['id'], 'file_name': img_info['file_name']} for img_info in train_images_info]
test_images = [{'id': img_info['id'], 'file_name': img_info['file_name']} for img_info in test_images_info]

# Debug: Print the lengths of train and test images to verify counts
print(f"Number of train images: {len(train_images)}")
print(f"Number of test images: {len(test_images)}")

write_csv_with_slno(train_images_csv, train_images, ['id', 'file_name'])
write_csv_with_slno(test_images_csv, test_images, ['id', 'file_name'])
