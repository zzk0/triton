import nvidia.dali as dali
import nvidia.dali.fn as fn
import multiprocess as mp

@dali.pipeline_def(batch_size=128, num_threads=4, device_id=0)
def pipeline():
    images = fn.external_source(device='cpu', name='DALI_INPUT_0')
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.transpose(images, perm=[2, 0, 1])
    images = images / 255
    return images


pipeline().serialize(filename='./1/model.dali')
