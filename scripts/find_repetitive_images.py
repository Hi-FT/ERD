# coding=utf-8
"""
@Data: 2021/1/4
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
import time
import json


def find_repeat(first_cats_json, last_cats_json):
    # load first cats
    print('loading annotations into memory...')
    tic = time.time()
    first_cats_dataset = json.load(open(first_cats_json, 'r'))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))
    # load last cats
    print('loading annotations into memory...')
    tic = time.time()
    last_cats_dataset = json.load(open(last_cats_json, 'r'))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))

    # tranfer list to set
    first_cats_image_ids = [image['id'] for image in first_cats_dataset['images']]
    first_cats_image_id_set = set(first_cats_image_ids)

    last_cats_image_ids = [image['id'] for image in last_cats_dataset['images']]
    last_cats_image_id_set = set(last_cats_image_ids)

    # find intersections
    intersections = first_cats_image_id_set.intersection(last_cats_image_id_set)
    print(len(intersections))  # 34507

    # find first 40 cats that not in intersections
    appear_cat_ids = set()
    for anno in first_cats_dataset['annotations']:
        if anno['image_id'] in intersections:
            appear_cat_ids.add(anno['category_id'])
    full_cats = set([cate['id'] for cate in first_cats_dataset['categories']])
    miss_cats = full_cats - appear_cat_ids
    print(miss_cats)  # {}

    # find last 40 cats that not in intersections
    appear_cat_ids = set()
    for anno in last_cats_dataset['annotations']:
        if anno['image_id'] in intersections:
            appear_cat_ids.add(anno['category_id'])
    full_cats = set([cate['id'] for cate in last_cats_dataset['categories']])
    miss_cats = full_cats - appear_cat_ids
    print(miss_cats)  # {}


if __name__ == "__main__":
    first_cats_json = '/data-nas/sy/coco/annotations/instances_train2017_sel_first_40_cats.json'
    last_cats_json = '/data-nas/sy/coco/annotations/instances_train2017_sel_last_40_cats.json'
    find_repeat(first_cats_json, last_cats_json)
