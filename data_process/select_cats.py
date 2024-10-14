import argparse
import json
import time
import os.path as osp


def arg_parse():
    parser = argparse.ArgumentParser(description='COCO Dataset Loader')

    parser.add_argument('--dataset', default='COCO', help='dataset type')
    parser.add_argument('--data_path',
                        default='/home/bl/Documents/Projects/Space/detections/datasets_ssd/coco_root/annotations',
                        help='annotation path')
    parser.add_argument('--anno_file', default='instances_train2017', help='annotation file without suffix')
    # parser.add_argument('--anno_file', default='instances_val2017', help='annotation file without suffix')
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()
    anno_file = osp.join(args.data_path, args.anno_file)
    print('loading annotations into memory ...')
    tic = time.time()
    dataset = json.load(open(anno_file + '.json', 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    # sort by cat_ids
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])

    # ========================================>
    # select specific cat_ids
    sel_num_0, sel_num_1 = 40, 80
    filename_suffix = '_last_40_catss'
    # <========================================

    sel_cats = dataset['categories'][sel_num_0:sel_num_1]  # select first 40 casts

    # select specific annotations
    sel_cats_ids = [cat['id'] for cat in sel_cats]

    sel_anno = []
    sel_images_ids = []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_images_ids.append(anno['image_id'])
    sel_images_ids = set(sel_images_ids)

    sel_images = []
    for img_ in dataset['images']:
        if img_['id'] in sel_images_ids:
            sel_images.append(img_)

    # selected dataset dict
    sel_dataset = dict()
    sel_dataset['categories'] = sel_cats
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images

    fp = open(anno_file + filename_suffix + '.json', 'w')
    json.dump(sel_dataset, fp)


if __name__ == '__main__':
    main()
