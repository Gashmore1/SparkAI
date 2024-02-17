from pyspark.sql.types import DoubleType, FloatType, LongType, StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession
import os
import time

spark = SparkSession.builder.getOrCreate()

def get_roots(path):
    folders = [path]
    contents = os.listdir(path)

    for content in contents:
        if content != ".DS_Store":
            new_path = os.path.join(path,content)
            if os.path.isdir(new_path):
                folders.append(str(new_path))
                folders.extend(get_roots(new_path))

    return list(set(folders))

# Cannot be recursive since FIFO means that new tasks will never be completed due to the old ones still waiting to be finished
# FIFO means that inital discovery of root may be better and to work from roots up 

def file_filter(path):
    return not os.path.isdir(path)

def file_type_filter(file):
    file = file.upper()
    file_types = ['.JPG','.PNG','.JPEG']
    return any(ext in file for ext in file_types)

def find_images(path):
    time.sleep(10)
    files = os.listdir(path)

    paths = [os.path.join(path, image) for image in files]

    files = list(filter(file_filter, paths))
    image_paths = list(filter(file_type_filter, files))

    image_paths = [{"ImagePath":str(image)} for image in image_paths]

    return image_paths

def main():
    paths = get_roots("/mnt/photos/Camera Roll/")

    print(len(paths))

    output = spark.sparkContext.parallelize(paths, len(paths)).flatMap(find_images).collect()

    create_table(output)
    read_table()

def create_table(data):
    schema = StructType([
        StructField("ImagePath", StringType(), False)
    ])

    df = spark.createDataFrame(data, schema)

    df.writeTo("facial_recognition.image_path").using("iceberg").replace()

def read_table():
    df = spark.read.table("facial_recognition.image_path")

    print(df.collect())
    print(df.count())


if __name__ == "__main__":
    main()