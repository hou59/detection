# coding: utf-8
import os
from tqdm import tqdm
import json

ori_jpg_path = 'datasets/train/train'
oir_json_path = 'datasets/train/train.json'
savepath = 'model_data/trainlist.txt'
labels =[
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
n_classes = len(labels)

ori_json = json.load(open(oir_json_path, 'r'))

with open(savepath, 'w') as list_file:
    for items in tqdm(ori_json.items(),total=len(ori_json)):
        jpg_path = os.path.join(ori_jpg_path,items[0])
        newline = jpg_path
        for box_idx,obj in enumerate(items[1]['objects'].values()):
            cls = obj['category']
            if cls not in labels:
                continue
            label = labels.index(cls)
            bbox = obj['bbox']
            newline = (newline + " " + ",".join([str(a) for a in bbox]) + "," + str(label) )
        list_file.write(newline+ '\n')
print('ok')

ori_jpg_path = 'datasets/val/val'
oir_json_path = 'datasets/val/val.json'
ori_json = json.load(open(oir_json_path, 'r'))

with open(savepath, 'a') as list_file:
    for items in tqdm(ori_json.items(),total=len(ori_json)):
        jpg_path = os.path.join(ori_jpg_path,items[0])
        newline = jpg_path
        for box_idx,obj in enumerate(items[1]['objects'].values()):
            cls = obj['category']
            if cls not in labels:
                continue
            label = labels.index(cls)
            bbox = obj['bbox']
            newline = (newline + " " + ",".join([str(a) for a in bbox]) + "," + str(label) )
        list_file.write(newline+ '\n')
print('ok')