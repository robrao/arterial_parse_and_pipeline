import tensorflow as tf
import display_imgs
import numpy as np

tf.compat.v1.enable_eager_execution()

img_mask_feature_desc = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'img_width': tf.io.FixedLenFeature([], tf.int64),
    'img_height': tf.io.FixedLenFeature([], tf.int64),
    'mask_raw': tf.io.FixedLenFeature([], tf.string),
    'mask_erroneous': tf.io.FixedLenFeature([], tf.int64)
}

def get_features(features):
    data = {}
    width = features['img_width']
    height = features['img_height']
    img_np_string = features['image_raw'].numpy()
    mask_np_string = features['mask_raw'].numpy()
    img_np_1d = np.fromstring(img_np_string, dtype=np.float)
    mask_np_1d = np.fromstring(mask_np_string, dtype=np.uint8)
    data['img'] = img_np_1d.reshape(height, width)
    data['mask'] = mask_np_1d.reshape(height, width)
    data['width'] = width
    data['height'] = height
    data['mask_erroneous'] = features['mask_erroneous']

    return data

def _parse_tfrecord(example_proto):
    return tf.io.parse_single_example(example_proto, img_mask_feature_desc)

def load_data(batch_size, shuffle=False):
    all_data = []
    # shuffling 1k to ensure all samples get randomized, since we certainly have less than 1k samples.
    raw_image_dataset = tf.data.TFRecordDataset('dicom_imgs_masks.tfrecord').shuffle(1000).batch(batch_size)

    for raw_batch in raw_image_dataset:
        for raw_data in raw_batch:
            features = _parse_tfrecord(raw_data)
            data = get_features(features)
            all_data.append(data)

    return all_data

if __name__ == "__main__":
    dataset = load_data(8, True)

    for data in dataset:
        display_imgs.display_img_with_mask(data['img'], data['mask'], 'unknown')
