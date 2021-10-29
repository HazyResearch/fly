# Adapted from https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# and https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py
# and https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-lightning.html
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI.")


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_train=True,
                         shuffle=False):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=shuffle,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_train:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.75, 4.0 / 3.0],
                                               random_area=[0.08, 1.0],
                                               num_attempts=100)
        # INTERP_TRIANGULAR produces results much closer to torchvision default (bilinear) mode.
        # For example, on T2T-ViT-7, I get 71.576 if using torchvision loader, 71.574 if using
        # INTERP_TRIANGULAR, and 71.0 if using INTERP_LINEAR.
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


class DALIClassificationIteratorWrapper(DALIClassificationIterator):
    """Wrap it to return tuple instead of dictionary, and to squeeze the labels.
    """
    def __init__(self, *kargs, is_train=True, **kvargs):
        super().__init__(*kargs, **kvargs)
        self._is_train = is_train

    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process
        # also we need to transform dict returned by DALIClassificationIterator to iterable
        # and squeeze the labels
        out = out[0]
        # TD [2021-08-28] Without .clone(), I get garbage results (acc=0.1%).
        # I think it's because DALI replaces the buffer content with the next batch before the
        # backward pass.
        return out['data'].clone(), out['label'].squeeze(-1).long().clone()

    # HACK: TD [2021-08-29] Pytorch-lightning relies on the length of the dataloader to count
    # how many times to advance the dataloader (but only for evaluation). However, DALI iterator
    # requires advancing to the end every time before resetting.
    # So there's a bug here. Suppose that there are 10 batches.
    # PL will only advance 10 times, but DALI iterator requires calling __next__ 11 times. On the 11th
    # time, the DALI iterator will raise StopIteration. This means that from PL's perspective,
    # the val loader has 10 batches on epoch 1, but 0 batches on epoch 2, and 10 batches on epoch 3,
    # etc. As a result, validation is run every 2 epochs instead of every epoch.
    # We fake the length (increase by 1) to trick PL into calling __next__ 11 times each epoch, so
    # that it plays well with DALI iterator.
    def __len__(self):
        return super().__len__() + int(not self._is_train)


def get_dali_loader(data_dir, crop, size, is_train, batch_size, shuffle, drop_last, num_threads,
                    device_id, shard_id, num_shards, dali_cpu):
    pipe = create_dali_pipeline(data_dir=data_dir, crop=crop, size=size, is_train=is_train,
                                batch_size=batch_size, shuffle=shuffle, seed=12 + device_id,
                                num_threads=num_threads, device_id=device_id, shard_id=shard_id,
                                num_shards=num_shards, dali_cpu=dali_cpu)
    pipe.build()
    last_batch_policy = LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL
    return DALIClassificationIteratorWrapper(pipe, is_train=is_train, reader_name="Reader",
                                             last_batch_policy=last_batch_policy, auto_reset=True)
