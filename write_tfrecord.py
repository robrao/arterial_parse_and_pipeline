import prepare_data
import tensorflow as tf

def _bytes_features(value):
    """Returns a bytes_list from a string / byte."""
    string_value = value.tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[string_value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tf_example(img, mask, img_width, img_height, err):
    feature = {
        'image_raw': _bytes_features(img),
        'img_width': _int64_feature(img_width),
        'img_height': _int64_feature(img_height),
        'mask_raw': _bytes_features(mask),
        'mask_erroneous': _int64_feature(err),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tf_records():
    imgs_masks = prepare_data.map_imgs_to_masks()

    with tf.io.TFRecordWriter('dicom_imgs_masks.tfrecord') as writer:
        for img_path in imgs_masks:
            img = imgs_masks[img_path]['image']
            img_width = imgs_masks[img_path]['img_width']
            img_height = imgs_masks[img_path]['img_height']
            mask = imgs_masks[img_path]['mask']
            err = imgs_masks[img_path]['mask_erroneous']

            tf_example = create_tf_example(img, mask, img_width, img_height, err)
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    write_tf_records()