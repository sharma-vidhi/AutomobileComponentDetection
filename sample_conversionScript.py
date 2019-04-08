# from __future__ import with_statement
import tensorflow as tf
import os
import io
import glob
import json
from PIL import Image

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_path', '', 'Path to input image')
FLAGS = flags.FLAGS

if (FLAGS.image_path is None):
    print("Please indicate image directory")
    exit()
try:
    with open(FLAGS.image_path+"/info.json") as jfile:
        data = json.loads(jfile.read())
        data = data['data']

except Exception as e:
    print(e)
    exit()
# print(data)

def create_tf_example(fDir):
    # TODO(user): Populate the following variables from your example.
    id = (int)(fDir.split('/')[-1].split('.')[0])
    found = False
    for x in data:
        if x['id']==id:
            found = True
            tmp = x
            break
    if not found:
        print("json obj doesn't exist for this img")
        exit()

    try:
        with tf.gfile.GFile(fDir, 'rb') as fid:
            encoded_jpg = fid.read()
    except Exception as e:
        print(e)
        exit()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # height = image.get_height() # Image height
    # width = image.get_width() # Image width

    filename = fDir.split('/')[-1] # Filename of the image. Empty if image is not from file
    filename = filename.encode('utf8')
    print(filename)

    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmins = tmp['xmins']
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    xmax = tmp['xmaxs']
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymins = tmp['ymins']
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs = tmp['ymaxs']
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes_text = tmp['classes_text']
    classes = [] # List of integer class id of bounding box (1 per box)
    classes = tmp['classes']

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # TODO(user): populate data

    for example in glob.glob(FLAGS.image_path+"/*.jpg"):
        # print(example)
        # print("ok")
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
