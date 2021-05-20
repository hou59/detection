import json
import numpy as np
import os
from yolo3.yolo import YOLO
from PIL import Image
import glob
import matplotlib.pyplot as plt

def parse_rec(json_file):
    objects = []
    ori_json = json.load(open(json_file, 'r'))
    for it, em in ori_json.items():
        if isinstance(em, dict) and 'bounding_box' in list(em.keys()):
            obj_struct = {}
            bbox = em['bounding_box']
            label = em['category_name']
            obj_struct['name'] = label
            obj_struct['pose'] = ''
            obj_struct['truncated'] = 0
            obj_struct['difficult'] = 0
            x1, y1, x2, y2 = bbox
            obj_struct['bbox'] = [int(x1),int(y1),int(x2),int(y2)]
            objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_recs(json_files):
    recs = {}
    for ith, json_file in enumerate(json_files):
        print('%d / %d'%(ith, len(json_files)))
        json_name = os.path.basename(json_file).split('.')[0]
        recs[json_name] = parse_rec(json_file)
    return recs

def get_class_recs(json_files,classname):
    class_recs = {}
    npos = 0
    for ith, json_file in enumerate(json_files):
        # print('%d / %d'%(ith, len(json_files)))
        json_name = os.path.basename(json_file).split('.')[0]
        R = [obj for obj in recs[json_name] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[ith] = {'bbox': bbox,
                           'difficult': difficult,
                           'det': det,
                           'json_name':json_name}
    return class_recs,npos

def get_imgID_con_BB(detector,jpeg,jpg_path):
    image_ids = []
    confidence = []
    BB = []
    for ith, json_file in enumerate(json_files):
        print('%d / %d'%(ith, len(json_files)))
        json_name = os.path.basename(json_file).split('.')[0]
        img_path = os.path.join(jpg_path,json_name+'.'+jpeg)
        image = Image.open(img_path)
        detecte_image, [b, s, c] = detector.detect_image(image)
        for jth in range(len(c)):
            image_ids.append(ith)
            confidence.append(s[jth])
            top, left, bottom, right = b[jth]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            BB.append([ left,top,  right,bottom,])
    confidence = np.array(confidence)
    BB = np.array(BB)
    return image_ids, confidence, BB

if __name__ == '__main__':

    labels = [
        'pavement',
        'dish',
        'bird',
        'gate',
        'towel',
        'spots',
        'woman',
        'orange',
        'necklace',
        'walkway',
        'number',
        'vegetable',
        'street light',
        'post',
        'tablecloth',
        'handle',
        'faucet',
        'signs',
        'painting',
        'paint',
        'seat',
        'road',
        'black',
        'column',
        'pattern',
        'frame',
        'dog',
        'skier',
        'bathroom',
        'words',
        'poles',
        'hot dog',
        'ground',
        'sheep',
        'edge',
        'rack',
        'house',
        'lines',
        'net',
        'pipe',
        'beach',
        'jar',
        'sneaker',
        'object',
        'corner',
        'ski pole',
        'ring',
        'meat',
        'bicycle',
        'rug',
        'knife',
        'bag',
        'ceiling',
        'jacket',
        'top',
        'dirt',
        'scarf',
        'door',
        'container',
        'band',
        'mountain',
        'surfer',
        'buildings',
        'wires',
        'tray',
        'gloves',
        'area',
        'wing',
        'doors',
        'curb',
        'metal',
        'wave',
        'phone',
        'purse',
        'plant',
        'window',
        'sink',
        'palm tree',
        'label',
        'coat',
        'pizza',
        'tip',
        'bowl',
        'zebra',
        'sleeve',
        'bread',
        'bottom',
        'doorway',
        'girl',
        'mirror',
        'books',
        'photo',
        'a',
        'tag',
        'bus',
        'camera',
        'street',
        'horns',
        'tiles',
        'stand',
        'plane',
        'television',
        'uniform',
        'candle',
        'concrete',
        'ear',
        'clothes',
        'room',
        'watch',
        'trim',
        'teddy bear',
        'cup',
        'boats',
        'banana',
        'pillows',
        'cord',
        'suitcase',
        'suit',
        'nose',
        'backpack',
        'windshield',
        'chain',
        'cabinet',
        'hands',
        'apple',
        'park',
        'donut',
        'frisbee',
        'curtains',
        'star',
        'tomato',
        'desk',
        'cloud',
        'people',
        'lights',
        'name',
        'car',
        'rocks',
        'roof',
        'plants',
        'outlet',
        'reflection',
        'screen',
        'snow',
        'refrigerator',
        'socks',
        'cover',
        'spectator',
        'bracelet',
        'wall',
        'mug',
        'crust',
        'glasses',
        'goggles',
        'racket',
        'feet',
        'paper',
        'cheese',
        'picture',
        'foot',
        'bun',
        'jersey',
        'tv',
        'truck',
        'grass',
        'giraffe',
        'traffic light',
        'shoes',
        'writing',
        'whiskers',
        'path',
        'the',
        'eye',
        'sweater',
        'part',
        'shorts',
        'wood',
        'beak',
        'bear',
        'ripples',
        'cushion',
        'cows',
        'chair',
        'book',
        'stick',
        'train tracks',
        'curtain',
        'mouth',
        'sneakers',
        'bucket',
        'man',
        'guy',
        'shadows',
        'hoof',
        'headlights',
        'sidewalk',
        'bat',
        'baby',
        'basket',
        'bushes',
        'umpire',
        'leaf',
        'word',
        'trash can',
        'horn',
        'legs',
        'train car',
        'blue',
        'skateboard',
        'boots',
        'weeds',
        'collar',
        'snowboard',
        'glass',
        'railing',
        'sun',
        'airplane',
        'rope',
        'structure',
        'windows',
        'microwave',
        'van',
        'shadow',
        'pillar',
        'tile',
        'drawer',
        'ears',
        'stove',
        'pen',
        'birds',
        'ocean',
        'bar',
        'giraffes',
        'brick',
        'tie',
        'button',
        'toilet',
        'hand',
        'street sign',
        'finger',
        'banner',
        'fence',
        'oven',
        'arm',
        'field',
        'pocket',
        'front',
        'cone',
        'gravel',
        'logo',
        'license plate',
        'lettering',
        'graffiti',
        'sky',
        'spoon',
        'food',
        'bolt',
        'bananas',
        'table',
        'cloth',
        'tire',
        'knob',
        'log',
        'tennis court',
        'back',
        'jeans',
        'stripe',
        'surface',
        'computer',
        'wine',
        'strap',
        'onion',
        'elephants',
        'boat',
        'wrist',
        'red',
        'skis',
        't-shirt',
        'city',
        'bed',
        'crack',
        'horse',
        'head',
        'clock',
        'helmet',
        'pillow',
        'branch',
        'cars',
        'child',
        'stairs',
        'batter',
        'circle',
        'shirt',
        'pants',
        'boy',
        'wetsuit',
        'stone',
        'lady',
        'wheels',
        'surfboard',
        'branches',
        'zebras',
        'fur',
        'engine',
        'cellphone',
        'cap',
        'tusk',
        'air',
        'key',
        'hydrant',
        'building',
        'baseball',
        'animal',
        'buttons',
        'statue',
        'lamp',
        'carrot',
        'shoulder',
        'motorcycle',
        'skateboarder',
        'numbers',
        'spot',
        'blinds',
        'cell phone',
        'sign',
        'tent',
        'monitor',
        'wire',
        'player',
        'keyboard',
        'rail',
        'square',
        'bike',
        'tree',
        'waves',
        'blanket',
        'sticker',
        'board',
        'rock',
        'arrow',
        'mane',
        'jet',
        'kitchen',
        'hill',
        'men',
        'patch',
        'plate',
        'paw',
        'mouse',
        'bricks',
        'eyes',
        'light',
        'pot',
        'belt',
        'balcony',
        'runway',
        'bottle',
        'vase',
        'dress',
        'broccoli',
        'hood',
        'river',
        'ramp',
        'glove',
        'distance',
        'arms',
        'kite',
        'kid',
        'beard',
        'base',
        'hair',
        'hole',
        'poster',
        'shade',
        'catcher',
        'person',
        'fire hydrant',
        'elephant',
        'remote',
        'body',
        'letter',
        'pole',
        'cart',
        'court',
        'handles',
        'floor',
        'slice',
        'cabinets',
        'cake',
        'box',
        'pan',
        'chimney',
        'string',
        'luggage',
        'letters',
        'stop sign',
        'duck',
        'horses',
        'panel',
        'sand',
        'chairs',
        'fork',
        'mountains',
        'trunk',
        'leaves',
        'white',
        'leg',
        'carpet',
        'tennis racket',
        'steps',
        'shelf',
        'holder',
        'pepper',
        'background',
        'controller',
        'train',
        'ski',
        'stripes',
        'ball',
        'speaker',
        'sauce',
        'umbrella',
        'vehicle',
        'face',
        'bench',
        'boot',
        'fruit',
        'can',
        'neck',
        'tower',
        'vegetables',
        'sandwich',
        'bridge',
        'awning',
        'tail',
        'couch',
        'hat',
        'line',
        'wheel',
        'skirt',
        'tracks',
        'stem',
        'clouds',
        'sofa',
        'this',
        'trees',
        'vest',
        'shoe',
        'flag',
        'lid',
        'tree trunk',
        'side',
        'cow',
        'water',
        'sock',
        'track',
        'shore',
        'headlight',
        'napkin',
        'bush',
        'tennis ball',
        'cat',
        'sunglasses',
        'knee',
        'design',
        'flower',
        'platform',
        'counter',
        'laptop',
        'lettuce',
        'flowers'
    ]
    jpeg = 'jpg'

    input_path = '../jsons'
    jpg_path = '../images'
    json_files = glob.glob(os.path.join(input_path,'*.json'))
    recs = get_recs(json_files)

    detector = YOLO()
    image_ids, confidence, BB = get_imgID_con_BB(detector,jpeg,jpg_path)

    all_p,all_r,all_ap = [],[],[]
    for cls_name in list(labels.keys()):
        classname = cls_name                                            # replace
        class_recs,npos = get_class_recs(json_files,classname)

        ovthresh = 0.5

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)

        all_p.append(prec)
        all_r.append(rec)
        all_ap.append(ap)

        # print('%s precision is %.2f' %( classname,prec))
        # print('%s recall is %.2f' %( classname,rec))
        print('%s ap is %.2f' %( classname,ap))

    print('-----------------------------------------------------------------')
    # print('avg precision is %.2f' % sum(all_p)/len(all_p))
    # print('avg rec is %.2f' % sum(all_r)/len(all_r))
    print('avg ap is %.2f' % (sum(all_ap)/len(all_ap)))
    print('-----------------------------------------------------------------')
    print('ok')

    classes = list(labels.keys())
    n_classes = len(classes)
    mean_average_precision = sum(all_ap)/len(all_ap)
    average_precisions = all_ap
    precisions = all_p
    recalls = all_r

    for i in range(len(average_precisions)):
        print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
    print()
    print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
