# UCF101 Few-shot Action Recognition
sample code for few-shot action recognition on UCF101

```UCF101.py``` sampler supports autoaugment[1] when scarcity of frames in dataset.

## Requirements
*   torch>=1.6.0
*   torchvision>=0.7.0
*   tensorboard>=2.3.0

## Usage
download and extract frame from UCF101 videos. [UCF101 Frame Extractor](https://github.com/titania7777/UCF101FrameExtrcator)

split dataset for few-shot learning(if you already has csv files then you can skip this step)
```
python splitter.py
```
train(resnet18)
```
python train.py --frames-path /path/to/frames --save-path /path/to/save --tensorboard-path /path/to/tensorboard --model resnet --random-pad-sample --uniform-frame-sample --random-start-position --random-interval --bidirectional --pad-option autoaugment --way 5 --shot 1 --query 10
```
train(r2plus1d18)
```
python train.py --frames-path /path/to/frames --save-path /path/to/save --tensorboard-path /path/to/tensorboard --model r2plus1d --random-pad-sample --uniform-frame-sample --random-start-position --random-interval --pad-option autoaugment --way 5 --shot 1 --query 1
```
test(resnet18)
```
python test.py --frames-path /path/to/frames --save-path /path/to/load/saved/ --model resnet --bidirectional --way 5 --shot 1 --query 10
```
test(r2plus1d18)
```
python test.py --frames-path /path/to/frames --save-path /path/to/load/saved/ --model r2plus1d --way 5 --shot 1 --query 1
```

## ```UCF101.py``` Options
### common options
1. frames_path: directory path of extracted frames
2. labels_path: directory path of labels
4. frame_size: frame size(width and height is should be same)
5. sequence_length: number of frames sequence
6. setname: sampling mode, if this mode is 'train' then the sampler read a 'train.csv' file to load train dataset [default: 'train', others: 'train', 'val', 'test']
### pad options
7. random_pad_sample: sampling frames from existing frames with randomly when frames are insufficient, if this value is False then only use first frame repeatedly [default: True, other: False]
8. pad_option: when adds some pad for insufficient frames of video, if this value is 'autoaugment' then pads will augmented by autoaugment policies [default: 'default', other: 'autoaugment']
### frame sampler options
9. uniform_frame_sample: sampling frames with same interval, if this value is False then sampling frames with ignored interval [default: True, other: False]
10. random_start_position: decides the starting point with randomly by considering the interval, if this value is False then starting point is always 0 [default: True, other, False]
11. max_interval: setting of maximum frame interval, if this value is high then probability of missing sequence of video is high [default: 7]
12. random_interval: decides the interval value with randomly, if this value is False then use a maximum interval [default: True, other: False]

## CategoriesSampler Options in ```UCF101.py```
1. labels: this parameter receive of classes in csv files, so this value must be ```UCF101.classes```
2. iter_size: number of iteration per episodes
3. way: number of way(number of class)
4. shot: number of shot(number of shot data)
5. query: number of query(number of query data)  
*way, shot, query => we follow episodic training stratiegy[2], 

## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  
[2] Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and kavukcuoglu, koray and Wierstra, Daan, "Matching Networks for One Shot Learning", Neural Information Processing Systems(NIPS), 2016, pp. 3630-3638