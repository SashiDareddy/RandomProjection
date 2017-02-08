import logging
import os
import numpy as np
import math
import scipy.sparse as ssp
from pyspark.sql import functions as f
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT
from sklearn.externals import joblib


# to run this code use: spark-submit  --master yarn --executor-memory 3G --driver-memory 6G --num-executors 90 --executor-cores 1 randomProjection.py
def random_project_mappartitions_function(rdd_row_iterator, local_csr_matrix):
    """
    This function is intended to be used in a <df>.rdd.mapPartitions(lambda rdd_map: random_project_mapf(rdd_map, local_rnd_mat))
    setting.
    :param rdd_row_iterator: a list of n-dim sparse vectors
    :param local_csr_matrix: the projection matrix - should have dimensions: n x p
    :return: a list of p-dim sparse-vectors - same length as input
    """
    keys = []
    # this will be a list of single row sparsevectors transformed into scipy csr matrix
    features_single_row_matrix_list = []
    PROJECT_DIM_SIZE = local_csr_matrix.shape[1]

    for row in rdd_row_iterator:
        # capture keys
        if "label" in row:
            keys.append((row["id"], row["label"]))
        else:
            keys.append((row["id"]))
        # work on values:
        feature_dim_size = row["features"].size  # feature dimensionality before projection
        col_indices = row["features"].indices
        row_indices = [0] * len(col_indices)  # defaulting to 0 as we are creating single row matrix
        data = row["features"].values.astype(np.float32)

        feature_mat = ssp.coo_matrix((data, (row_indices, col_indices)), shape=(1, feature_dim_size)).tocsr()
        features_single_row_matrix_list.append(feature_mat)
    # vstacking single row matrices into one large sparse matrix
    features_matrix = ssp.vstack(features_single_row_matrix_list)
    del features_single_row_matrix_list

    projected_features = features_matrix.dot(local_csr_matrix)
    del features_matrix, local_csr_matrix

    projected_features_list = (Vectors.sparse(PROJECT_DIM_SIZE, zip(i.indices, i.data))
                               for i in projected_features)
    if "label" in row:
        return zip((i[0] for i in keys), (i[1] for i in keys), projected_features_list)
    else:
        return zip((i[0] for i in keys), projected_features_list)


if __name__ == '__main__':
    N_FEATURES=54686452     # these are the no. of features in the dataset we are goining to use in this app
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # fire up a spark session
    spark = SparkSession \
        .builder \
        .appName("PySpark Random Projection Demo") \
        .getOrCreate()

    sc = spark.sparkContext

    DATA_DIR = 'wasb://kdd12-blob@sashistorage.blob.core.windows.net/' # Azure Storage blob

    train = spark.read.format("libsvm").load(DATA_DIR, numFeatures=N_FEATURES)\
            .withColumn("id", f.monotonically_increasing_id())
    print(train.show())

    train_schema = StructType(
        [StructField("id", LongType(), False)
           ,StructField("label", FloatType(), False)
            , StructField("features", VectorUDT(), False)
            ]
    )

    # generating the random projection  matrix
    dummy_X_rows = train.count()
    dummy_X_cols = train.first()["features"].size
    dummy_X = ssp.csr_matrix((dummy_X_rows, dummy_X_cols), dtype=np.float32)  # the shape-only of this dummy

    # find the optimal (conservative estimate) no. of dimensions required according to the
    # johnson_lindenstrauss_min_dim function
    # rproj_ndim = johnson_lindenstrauss_min_dim(dummy_X_rows, eps=0.2) # returns 4250

    rproj_ndim = 4096

    logging.info("Fitting Sparse Random Projection to have %d columns after projection " % rproj_ndim)
    srp = SparseRandomProjection(n_components=rproj_ndim, random_state=123)

    srp.fit(dummy_X)
    logging.info("Saving the projection matrix to disk... ")
    joblib.dump(srp, os.path.join("/tmp","srp.pkl"))


    local_rnd_mat = srp.components_.T.astype(np.float32)

    # broadcast the local_rnd_mat so it is available on all nodes
    local_rnd_mat_bc_var = sc.broadcast(local_rnd_mat)

    logging.info("Applying random projection to  rdd map partitions")
    train_projected_df = train.rdd\
        .mapPartitions(lambda rdd_map_partition: random_project_mappartitions_function(rdd_map_partition,
                                                                                       local_rnd_mat_bc_var.value))\
        .toDF(train_schema)

    logging.info("Writing projected data to disk...")
    train_projected_df.write.mode("overwrite").parquet(DATA_DIR+"/train_features_random_projected.parquet/")

    logging.info("Sample rows from training set before projection...")
    print(train.show())
    logging.info("Sample rows from training set after projection...")
    print(spark.read.parquet(DATA_DIR+"/train_features_random_projected.parquet/").show())

    spark.stop()