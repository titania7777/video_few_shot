## UCF101
-------------
this UCF101 sampler utilize autoaugment[1] when scarcity of frame in dataset, and additionally you can check another options.

* this sampler is working for few-shot video action recognition !!
if you want action recognition(not few-shot) then [here](https://github.com/titania7777/Pytorch_Sampler/tree/master/UCF_101)

## UCF101 Options
### common options
1. frames_path: extracted frames folder path(you need to use our frame extractor !!)
2. labels_path: labels folder path
4. frame_size: frame size(width and height is same)
5. sequence_length: number of video frames
6. setname(default: 'train'): sampling mode, if this mode is 'train' then the sampler read a 'train.csv' file to load dataset('train', 'val', 'test')
7. random_pad_sample(default: True): randomly sample from existing frames when frames are insufficient(False: only use first frame)
### pad options
8. pad_option(default: 'default'): augment option, there is two option('default', 'autoaugment')
9. uniform_frame_sample(default: True): uniformly sampled the frame(False: random normally)
### frame sampler options
10. random_start_position(default: True): randomly decides the starting point by considering the interval(False: 0)
11. max_interval(default: 7): maximum frame interval, this value is high then you may miss the sequence of video
12. random_interval(default: True): randomly decides the interval value(False: use maximum interval)

## CategoriesSampler Options
1. labels: this parameter receive of classes in csv files so must be UCF101.classes
2. batch_size: batch size
3. way: number of way(number of class)[2]
4. shot: number of shot(number of shot data)[2]
5. query: number of query(number of query data)[2]

labels, batch_size, way, shot, query


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
python train.py
```
## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  
[2] Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and kavukcuoglu, koray and Wierstra, Daan, "Matching Networks for One Shot Learning", Neural Information Processing Systems(NIPS), 2016, pp. 3630-3638