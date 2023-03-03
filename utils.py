import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split


def read_dataset(images_dir, reports_dir):

    img = []
    img_finding = []
    finding = None

    for filename in tqdm(os.listdir(reports_dir)):
        if filename.endswith(".xml"):
            f = reports_dir + '/' + filename
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if child.tag == 'MedlineCitation':
                    for attr in child:
                        if attr.tag == 'Article':
                            for i in attr:
                                if i.tag == 'Abstract':
                                    for name in i:
                                        if name.get('Label') == 'FINDINGS':
                                            finding = name.text
            for p_image in root.findall('parentImage'):
                if finding is not None:
                    img.append(p_image.get('id'))
                    img_finding.append(finding)

    for i in range(len(img)):
        img[i] = os.path.join(images_dir, img[i] + '.png')

    train_images, test_images, train_reports, test_reports = train_test_split(
        img, img_finding, test_size=0.1, random_state=7
    )
    return train_images, train_reports, test_images, test_reports


def collate_fn(data, processor, max_length):
    images, reports = zip(*data)
    images = list(images)
    reports = list(reports)

    processed = processor(
        images, text=reports, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
    )
    return processed
