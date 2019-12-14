## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)

## Usage

### 1. Cloning the repository
```bash
$ git https://github.com/Ha0Tang/AsymmetricGAN
$ cd AsymmetricGAN_multi/
```

### 2. Downloading the dataset
To download the CelebA dataset:
```bash
$ bash download.sh celeba
```

To download the RaFD dataset, you must request access to the dataset from [the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main). Then, you need to create a folder structure as described [here](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md).

### 3. Training

```bash
$ bash train_asymmetricgan.sh
```

### 4. Testing

```bash
$ bash test_asymmetricgan.sh
```

### 5. Pretrained modelTo download a pretrained model checkpoint, run the script below.

```bash
$ bash download_pretrained.sh rafd_generator1
$ bash download_pretrained.sh rafd_generator2
$ bash download_pretrained.sh rafd_generator3
```

To translate images using the pretrained model, run the evaluation script below. 

```bash
$ python main.py --mode test --dataset RaFD --image_size 256 --c_dim 8 \
                 --rafd_image_dir data/RaFD/test \
                 --model_save_dir rafd_generator1_pretrained/models \
                 --result_dir rafd_generator1_pretrained/results
```
