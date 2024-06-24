from mmdetection.splits import get_unseen_class_ids

coco_unseen = get_unseen_class_ids('coco')
print(coco_unseen)