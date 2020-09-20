## UCF101
-------------
this UCF101 sampler support a autoaugment[1] when scarcity of frames in dataset, and you can check another options additionally.

* this sampler is working for few-shot video action recognition on UCF101 !!
if you want action recognition(not few-shot) then [here](https://github.com/titania7777/Pytorch_Sampler/tree/master/UCF_101)

## UCF101 Options
### common options
1. frames_path: directory path of extracted frames(you have to use our frame extractor !!)
2. labels_path: directory path of labels
4. frame_size: frame size(width and height is same)
5. sequence_length: number of video frames
6. setname(default: 'train', others: 'train', 'val', 'test'): sampling mode, if this mode is 'train' then the sampler read a 'train.csv' file to load train dataset.
7. random_pad_sample(default: True, other: False): sampling frames from existing frames with randomly when frames are insufficient, if this value is False then only use first frame.
### pad options
8. pad_option(default: 'default', other: 'autoaugment'): add some pads to insufficient frames of video, if this value is 'autoaugment' then pads will augmented by autoaugment policies.
9. uniform_frame_sample(default: True, other: False): sampling frames with random uniformly, if this value is False then sampling frames with random normally.
### frame sampler options
10. random_start_position(default: True, other, False): decides the starting point with randomly by considering the interval, if this value is False then starting point is 0.
11. max_interval(default: 7): setting of maximum frame interval, if this value is high then probability of missing sequence of video is high.
12. random_interval(default: True, other: False): decides the interval value with randomly, if this value is False then use a maximum interval.

## CategoriesSampler Options
1. labels: this parameter receive of classes in csv files so this value must be UCF101.classes
2. iter_size: number of iteration per episodes
3. way: number of way(number of class)
4. shot: number of shot(number of shot data)
5. query: number of query(number of query data)  
*way, shot, query => we follow episodic training stratiegy[2]


download UCF101 dataset.
```
wget http://hcir.iptime.org/UCF101.tar
```
extract frames from UCF101 videos.
```
python frame_extractor.py
```
split dataset for few-shot learning(if you already has csv files then you can skip this step)
```
python splitter.py
```
train(example)
```
python train.py --frames-path <your frames path> --labels-path ./UCF101_few_shot_labels/
```
## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  
[2] Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and kavukcuoglu, koray and Wierstra, Daan, "Matching Networks for One Shot Learning", Neural Information Processing Systems(NIPS), 2016, pp. 3630-3638