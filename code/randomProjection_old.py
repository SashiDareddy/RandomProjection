# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
import scipy.sparse as ssp
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from pyspark.sql import functions as f
import numpy as np
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT
from sklearn.externals import joblib
import math
import pandas as pd

def rproj(feature_vector, local_csr_matrix):
    data =feature_vector.values.astype(np.float32)
    
    col_indices = feature_vector.indices
    row_indices = [0]*len(col_indices) #[0]* len of col indices because we are dealing with a matrix with 1 row and n cols
    feature_mat = ssp.csr_matrix((data, (row_indices, col_indices )), shape=(1,feature_vector.size))
    prod_result = feature_mat.dot(local_csr_matrix).tocoo()
    indices = prod_result.col
    values = prod_result.data.astype(np.float32)
    ndim = prod_result.shape[1]
    return Vectors.sparse(ndim, dict(zip(indices, values)))  

def random_project_mapf(rdd_row_iterator, local_csr_matrix):
    keys=[]
    #this will be a list of single row sparsevectors transformed into scipy csr matrix
    features_single_row_matrix_list=[] 
    PROJECT_DIM_SIZE = local_csr_matrix.shape[1]

    for row in rdd_row_iterator:
        #capture keys
        if "clicked" in row:
            keys.append((row["display_id"], row["ad_id"], row["clicked"]))
        else:
            keys.append((row["display_id"], row["ad_id"]))
        #work on values:
        feature_dim_size = row["features"].size #feature dimensionality before projection
        col_indices = row["features"].indices
        row_indices = [0]*len(col_indices) #defaulting to 0 as we are creating single row matrix
        data = row["features"].values.astype(np.float32)
        
        feature_mat = ssp.coo_matrix((data, (row_indices,col_indices)), shape=(1, feature_dim_size)).tocsr()
        features_single_row_matrix_list.append(feature_mat)
    #vstacking single row matrices into one large sparse matrix
    features_matrix = ssp.vstack(features_single_row_matrix_list)
    del features_single_row_matrix_list
    
    projected_features = features_matrix.dot(local_csr_matrix)
    del features_matrix, local_csr_matrix
    
    projected_features_list= (Vectors.sparse(PROJECT_DIM_SIZE, zip(i.indices,i.data))
                                for i in projected_features)
    if "clicked" in row:
        return zip((i[0] for i in keys), (i[1] for i in keys), (i[2] for i in keys), projected_features_list)
    else:
        return zip((i[0] for i in keys), (i[1] for i in keys), projected_features_list)
    
"""        
>>> row = np.array([0, 0, 1, 2, 2, 2])
>>> col = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> projected_features=ssp.csr_matrix((data, (row, col)), shape=(3, 3))


v3 = valid.limit(50).repartition(10)
v3.cache()
v3.count()
v1 = v3.rdd.map(lambda row: ((row["display_id"], row["ad_id"], row["clicked"]), row["features"]))

%time a=v3.rdd.mapPartitions(lambda rdd_map : random_project_mapf(rdd_map, local_rnd_mat)).collect()
v3.rdd.mapPartitions( random_project_mapf, local_rnd_mat).glom().collect()
"""

def getNumNonZeroFeatures(sparseVector):
    return sparseVector.numNonzeros()

getNumNonZeroFeatures_UDF = f.udf(getNumNonZeroFeatures, IntegerType())

        
if __name__ =='__main__':
    
    DATA_DIR='/srv/kaggle/outbrain/data/'
    valid = spark.read.parquet(DATA_DIR+"valid_features.parquet")
    test = spark.read.parquet(DATA_DIR+"test_features.parquet")
    #the reason we are partitioning train but not valid/test is because
    #there are some part* files in train which have zero rows which messes up the 
    #map partition funtion : random_project_mapf
    n_workers=32
    train_num_partitions = spark.read.parquet(DATA_DIR+"train_features.parquet").rdd.getNumPartitions()
    train_num_repartitions = math.ceil(train_num_partitions/n_workers)*n_workers
    train = spark.read.parquet(DATA_DIR+"train_features.parquet").repartition(train_num_repartitions)
    
    
    train_schema = StructType(
    [StructField("display_id",IntegerType(),False)
    ,StructField("ad_id",IntegerType(),False)
    ,StructField("clicked",ShortType(),False)
    ,StructField("features",VectorUDT(),False)]
    )
    
    test_schema = StructType(
    [StructField("display_id",IntegerType(),False)
    ,StructField("ad_id",IntegerType(),False)
    ,StructField("features",VectorUDT(),False)]
    )
    
    #generating the random projection  matrix
    dummy_X_rows = train.count()
    dummy_X_cols = train.first()["features"].size
    dummy_X=ssp.csr_matrix( (dummy_X_rows, dummy_X_cols), dtype=np.float32) #the shape-only of this dummy 

    #rproj_ndim=24000 #24k corresponds to ~ 1.5 trillion sample ##sd: takes too long for downstream xgb
    #ex: johnson_lindenstrauss_min_dim(1e12) returns 23683.
    rproj_ndim=4096 #johnson_lindenstrauss_min_dim(69711098, [0.1, 0.2, 0.3]), corres to 0.2
    
    try:
        srp = joblib.load( "/srv/kaggle/outbrain/model_shelf/srp_{0}.pkl".format(rproj_ndim))
    except FileNotFoundError:
        srp = SparseRandomProjection(n_components=rproj_ndim, random_state=123)
        logging.info("Fitting Sparse Random Projection to have %d columns after projection " % rproj_ndim)
        srp.fit(dummy_X)
        joblib.dump(srp, "/srv/kaggle/outbrain/model_shelf/srp_{0}.pkl".format(rproj_ndim))
    
    
    local_rnd_mat = srp.components_.T.astype(np.float32)
    
    
    logging.info("Apply rdd maps")
    valid_projected_df = valid.rdd.mapPartitions(lambda rdd_map : random_project_mapf(rdd_map, local_rnd_mat)).toDF(train_schema)
    train_projected_df = train.rdd.mapPartitions(lambda rdd_map : random_project_mapf(rdd_map, local_rnd_mat)).toDF(train_schema)
    test_projected_df = test.rdd.mapPartitions(lambda rdd_map : random_project_mapf(rdd_map, local_rnd_mat)).toDF(test_schema)
    
    #logging.info("unravelling pair rdd after projection into a dataframe")
    #valid_projected_df = valid_projected.map(lambda row: (row[0][0], row[0][1], row[0][2], row[1])).toDF(train_schema)
    #train_projected_df = train_projected.map(lambda row: (row[0][0], row[0][1], row[0][2], row[1])).toDF(train_schema)
    #test_projected_df = test_projected.map(lambda row: (row[0][0], row[0][1], row[1])).toDF(test_schema)
    
    
    logging.info("Writing projected data to disk...")
    valid_projected_df.write.mode("overwrite").parquet("/srv/kaggle/outbrain/data/valid_features_random_projected.parquet")
    test_projected_df.write.mode("overwrite").parquet("/srv/kaggle/outbrain/data/test_features_random_projected.parquet")
    train_projected_df.write.mode("overwrite").parquet("/srv/kaggle/outbrain/data/train_features_random_projected.parquet")
    
    
    #check nnz distribution
    new_valid_projected_df = spark.read.parquet("/srv/kaggle/outbrain/data/valid_features_random_projected.parquet")
    new_test_projected_df = spark.read.parquet("/srv/kaggle/outbrain/data/test_features_random_projected.parquet")
    new_train_projected_df = spark.read.parquet("/srv/kaggle/outbrain/data/train_features_random_projected.parquet")
    
    new_valid_projected_dim = new_valid_projected_df.groupBy(getNumNonZeroFeatures_UDF("features")).agg(f.countDistinct("display_id", "ad_id").alias("cnt")).withColumn("type", f.lit("valid_projected")).toPandas()
    new_train_projected_dim = new_train_projected_df.groupBy(getNumNonZeroFeatures_UDF("features")).agg(f.countDistinct("display_id", "ad_id").alias("cnt")).withColumn("type", f.lit("train_projected")).toPandas()
    new_test_projected_dim = new_test_projected_df.groupBy(getNumNonZeroFeatures_UDF("features")).agg(f.countDistinct("display_id", "ad_id").alias("cnt")).withColumn("type", f.lit("test_projected")).toPandas()
    
    valid_dim = valid.groupBy(getNumNonZeroFeatures_UDF("features")).agg(f.countDistinct("display_id", "ad_id").alias("cnt")).withColumn("type", f.lit("valid")).toPandas()
    train_dim = train.groupBy(getNumNonZeroFeatures_UDF("features")).agg(f.countDistinct("display_id", "ad_id").alias("cnt")).withColumn("type", f.lit("train")).toPandas()
    test_dim = test.groupBy(getNumNonZeroFeatures_UDF("features")).agg(f.countDistinct("display_id", "ad_id").alias("cnt")).withColumn("type", f.lit("test")).toPandas()
    
    
    dim_data = pd.concat([new_valid_projected_dim,new_train_projected_dim, new_test_projected_dim,
                          valid_dim, train_dim, test_dim], axis=0, ignore_index=True)
    
    dim_data.to_csv("/srv/kaggle/outbrain/data/dim_data2.csv", header=True, index=False)
    
    summary_cnt_df = dim_data.groupby("type").describe()
    