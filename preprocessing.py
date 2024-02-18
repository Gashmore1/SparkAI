from pyspark.sql.types import FloatType, LongType, StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.sql import SparkSession
import os
import time
import cv2
import numpy as np
from skimage.feature import hog
import hashlib

spark = SparkSession.builder.getOrCreate()
SOURCE_TABLE = "facial_recognition.image_path"
TARGET_TABLE = "facial_recognition.historgram_oriented_gradients"
MAX_DF_SIZE = 2000

def process_image(path):
    image = cv2.imread(path.ImagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    old_shape = image.shape

    scale = 640 / np.min(old_shape)

    new_shape = (scale * np.array(old_shape)).astype(int)

    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

    new_shape = np.array(image.shape)
    slice_shape = np.array((100,200))
    slices = np.divide(new_shape,slice_shape)
    slices_int = (2*np.ceil(slices)).astype(int)

    slice_step = np.divide((new_shape - slice_shape),slices_int)

    hog_results = []

    for row in range(slices_int[0]+1):
        for col in range(slices_int[1]+1):
            x = int(slice_step[0] * row)
            y = int(slice_step[1] * col)

            image_slice = image[x:x+slice_shape[0], y:y+slice_shape[1]]
            slice_hash = hashlib.md5(image_slice.tobytes()).hexdigest()

            cell_shape = (8, 8)
            cells_per_block = (2,2)

            fd = hog(image_slice, orientations=8, pixels_per_cell=cell_shape,
                            cells_per_block=cells_per_block, visualize=False,
                            block_norm="L2-Hys")
            
            max_fd = np.max(fd)
            min_fd = np.min(fd)

            fd = (fd - min_fd) * 1/(max_fd-min_fd)

            hog_results.append(
                {
                    "hash": slice_hash,
                    "path":path.ImagePath,
                    "feature_descriptors": list(map(float, fd)),
                    "x": int(x),
                    "y": int(y),
                    "slice_x": int(slice_shape[0]),
                    "slice_y": int(slice_shape[1])
                })

    return hog_results

def main():
    count = MAX_DF_SIZE
    while count >= MAX_DF_SIZE:
        df = spark.read.table(SOURCE_TABLE)

        target_table_exists = spark.catalog.tableExists(TARGET_TABLE)

        if target_table_exists:
            target = spark.read.table(TARGET_TABLE).select("path")
            df = df.join(target, df.ImagePath == target.path,"anti").limit(MAX_DF_SIZE)
        else:
            df = df.limit(MAX_DF_SIZE)
        
        count = df.count()
        print(count)
        if count == 0:
            break

        df = df.repartition(df.count())

        df = df.rdd.flatMap(process_image)

        schema = StructType([
                        StructField("hash", StringType(), False),
                        StructField("path",StringType(), False),
                        StructField("feature_descriptors", ArrayType(FloatType(), False), False),
                        StructField("x", IntegerType(), False),
                        StructField("y", IntegerType(), False),
                        StructField("slice_x", IntegerType(), False),
                        StructField("slice_y", IntegerType(), False)
                    ])

        df = spark.createDataFrame(df, schema)

        if target_table_exists:
            df.writeTo(TARGET_TABLE).using("iceberg").append()
        else:
            df.writeTo(TARGET_TABLE).using("iceberg").createOrReplace()

if __name__ == "__main__":
    main()