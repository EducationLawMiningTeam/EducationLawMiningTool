from .registry import DATASETS
from .xml_style import XMLDataset
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
from splits import get_unseen_class_ids, get_seen_class_ids


@DATASETS.register_module
class VOCDataset(XMLDataset):
    
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'cat', 'chair', 'cow', 'diningtable', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'tvmonitor',
               'car', 'dog', 'sofa', 'train')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
    
    def set_classes_split(self):
        self.unseen_classes = ['car', 'dog', 'sofa', 'train']
        self.seen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'cat', 'chair', 'cow', 'diningtable', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'tvmonitor']

    def load_annotations(self, ann_file,  classes_to_load=None, split=None):
        self.set_classes_split()
        img_infos = []
        classes_to_exclude = None

        unseen_class_ids, seen_class_ids = get_unseen_class_ids('voc')-1, get_seen_class_ids('voc')-1
        
        if 'unseen' in classes_to_load or classes_to_load=='zsd':
            classes_to_exclude = self.seen_classes
        elif 'seen' in classes_to_load:
            classes_to_exclude = self.unseen_classes
        elif classes_to_load == 'gzsd':
            self.cat_to_load = unseen_class_ids

        classes_loaded = []
        img_ids = mmcv.list_from_file(ann_file)
        if classes_to_load == 'gzsd':
            self.class_names_to_load = np.array(self.CLASSES)[self.cat_to_load]
            for img_id in img_ids:
                filename = 'JPEGImages/{}.jpg'.format(img_id)
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    '{}.xml'.format(img_id))
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                include_image = self.should_include_image(root)
                if include_image == True:
                    img_infos.append(
                        dict(id=img_id, filename=filename, width=width, height=height))
        else:
            for img_id in img_ids:
                filename = 'JPEGImages/{}.jpg'.format(img_id)
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    '{}.xml'.format(img_id))
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                include_image = True
                if classes_to_exclude is not None:
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        if name in classes_to_exclude:
                            include_image = False
                            break
                        classes_loaded.append(name)

                if include_image == True:
                    img_infos.append(
                        dict(id=img_id, filename=filename, width=width, height=height))
        
        # import pdb; pdb.set_trace()
        # files = ["VOC2007/"+filename['filename'] for filename in img_infos]
        print(f"classes loaded {np.unique(np.array(classes_loaded))}")

        return img_infos
    
    
    def should_include_image(self, root):
        """
        root: xml file parser
        while loading annotations checks whether to include image in the dataset
        checks for each obj name in the class_names_to_load list
        for seen classes we strictly exclude objects if an unseen object is present
        for unseen classes during validation we load the image if the unseen object is present and ignore the annotation for seen object
        """
        include_image = False
        # if self.classes_to_load == 'seen':
        #     # include stricktly only images with seen objects 
        #     for obj in root.findall('object'):
        #         name = obj.find('name').text
        #         if name in self.class_names_to_load:
        #             include_image = True
        #         else:
        #             include_image = False
        #             break
        # else:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_names_to_load:
                include_image = True
                break
        return include_image