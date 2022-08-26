# coding=utf-8
"""
@Data: 2021/1/4
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
import os
import time
import json


def sel_cat(anno_file, sel_num):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(anno_file, 'r'))
    assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))

    # sort by cat_ids
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])

    # select first 40 cats
    sel_cats = dataset['categories'][:sel_num]
    # selected annotations
    sel_cats_ids = [cat['id'] for cat in sel_cats]
    sel_anno = []
    sel_image_ids = []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_image_ids.append(anno['image_id'])
    # selected images
    sel_images = []
    for image in dataset['images']:
        if image['id'] in sel_image_ids:
            sel_images.append(image)
    # selected dataset
    sel_dataset = dict()
    sel_dataset['categories'] = sel_cats
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images
    # writing results
    # fp = open(os.path.splitext(anno_file)[0] + '_sel_first_40_cats.json', 'w')
    # json.dump(sel_dataset, fp)

    
    # select last 40 cats
    sel_cats = dataset['categories'][sel_num:]
    # selected annotations
    sel_cats_ids = [cat['id'] for cat in sel_cats]
    sel_anno = []
    sel_image_ids = []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_image_ids.append(anno['image_id'])
    # selected images
    sel_images = []
    for image in dataset['images']:
        if image['id'] in sel_image_ids:
            sel_images.append(image)
    # selected dataset
    sel_dataset = dict()
    sel_dataset['categories'] = sel_cats
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images
    # writing results
    fp = open(os.path.splitext(anno_file)[0] + '_sel_last_40_cats.json', 'w')
    json.dump(sel_dataset, fp)


if __name__ == "__main__":
    anno_file = '/data-nas/sy/coco/annotations/instances_val2017.json'
    sel_num = 40
    sel_cat(anno_file, sel_num)
