

## Test
We prepare pretrained models on SemanticKITTI/ScanNet datset, you can evaluate the compression performamce by running
```shell
python test_single.py -d kitti
```
or 
```shell
python test_single.py -d scannet
```
## Evaluate PSNR
You can test the PSNR of different depths of octree.
You will get the ply file of an octree by runnning
```shell
cd utils
python generateOctreePly.py
```
Then you can get into the metrics folder and evaluate the PSNR
```shell
cd metrics
python calPSNR.py
```

## Training
For training on SemanticKITTI dataset, you need to prepare the training data by running 
```shell
cd utils
python generateOctreeNpy
```
You need to set the original KITTI ply folder and the target npy folder in `generateOctreeNpy.py`.
After that, you should get the npy folder organize like:
```
root
    sequence_512_octree13_split
        train
            00_000000.npy
            00_000001.npy
            ...
        test
            11_000000.npy
            11_000001.npy
            ...
        val
            08_000000.npy
            08_000001.npy
            ...
```
When the data is well prepared, you should speicify the root location in `/data/kitti_dataset_transformer.py` and run the training command as followed:
```shell
python train.py --sequence_size 512 --hidden 256 --batch_size 128 --epochs 15 --print_freq 100 --lr 0.0005 --val_freq 1 --weight-decay 0.001 --nhead 8 --num_layer 6 --tree_depth 13 --dataset kitti  --use_absolute_pos False --use_OctLeFF True --use_OctPEG True --save_dir ./training_logs/tmp/
```

Also, when the data is well prepared, you can run the following command to test the whole average bpp on test set:
```sheel
python test.py
```


For training on ScanNet, you should prepare the ScanNet dataset like this:
```
scannet_50000_normalize
    train
        scene0000_00_vert.npy
        scene0000_01_vert.npy
        ...
    test
        scene0011_00_vert.npy
        scene0011_01_vert.npy
        ...
    val
        scene0003_00_vert.npy
        scene0003_01_vert.npy
        ...
```
Also, you should you should speicify the root location in `/data/scannet_dataset_tranformer_mem.py` and run the training command as followed:
```shell
python train.py --sequence_size 1024 --hidden 256 --batch_size 32 --epochs 15 --print_freq 100 --lr 0.0005 --val_freq 1 --weight-decay 0.001 --nhead 8 --num_layer 6 --tree_depth 13 --dataset scannet  --use_absolute_pos False --use_OctLeFF True --use_OctPEG True --save_dir ./training_logs/tmp/
```

## Speed test
You can run the speed test by running
```shell
cd speed_test
python our_speed.py
python octattention_speed.py
python voxelcontext_speed.py
```

## Visulization
We prepare the visulization tool to visulize point cloud.
You can run different command to visualize different point clouds.
```shell
cd utils
python vis.py -f ../plys/13_000000.ply
python vis.py -f ../plys/13_000000_depth12.ply
...
```











