# Scaling up Scikit-Learn's Random Projection using Apache Spark

###### What is Random Projection (RP)?
Random Projection is a mathematical technique to reduce the dimensionality of a problem much like Singular Value Decomposition (SVD) or Principal Component Analysis (PCA) but only simpler & computationaly faster. 

[Throught out this article I will use Random Projection and Sparse Random Projection interchangeably.]

It is particularly useful when :
* Calculating PCA/SVD can be very prohibitive in terms of both running time & memory/hardware constraints.
* Working with extremely large dimensional **_sparse_** datasets with both large **n** (>> 100million rows/samples) & large **m** (>> 10 million columns/features) in a distributed setting. 
* When you require sparsity in the low-demensional projected data

###### Why can't we use Scikit-Learn?
You can use Scikit-Learn if your dataset is small enough to fit in a single machine's memory. But, when it comes to the actual projection you will find that it doesn't scale well beyond a single machine's resources.

###### Why can't we use Apache Spark?
As at version [2.1.0](http://spark.apache.org/docs/2.1.0/mllib-dimensionality-reduction.html) of Apache Spark MLLib, the following dimensionality reduction techniques are available:
* Singular Value Decomposition (SVD)
* Principal Component Analysis (PCA)

There has been atleast [one attempt](https://github.com/apache/spark/pull/6613) to implement Random Projection in Apache Spark MLLib but those efforts don't appear to have made it through to the latest release.


In this article I will present a recipe to perform Random Projection using PySpark. It brings the scalability of Apache Spark to the Random Projection implementation in Scikit-Learn. As a bonus, you can extend the idea presented in this article to perform general ***sparse matrix*** by ***sparse matrix*** multiplication (as long as one of the sparse matrix is small enough to fit in memory) resulting in another ***sparse matrix***.  
 
###### Further reading:
* [4.5. Random Projection](http://scikit-learn.org/stable/modules/random_projection.html) - particularly, **_The Johnson-Lindenstrauss lemma_** and **_Sparse random projection_** sections

###### So, how do we apply Random Projection? 
There are two main steps in projecting a **n x m** matrix into a low demensional space using Random Projection:

1. Generating a **m x p** Projection Matrix with a pre-specified sparsity factor  - this is where we will leverage Scikit-Learn's implementation of Sparse Random Projection and generate the projection matrix. 
2. Matrix multipication - wherein we multiply an **n x m** input dataset with an **m x p** Projection Matrix yielding a new **n x p** matrix (which is said to be projected into a lower dimension) - we will leverage Scipy & Spark to deliver the large scale ***sparse matrix*** by ***sparse matrix*** multiplication resulting in another ***sparse matrix*** in a lower dimension


##### The Setup
Here, I will show how you how to apply Random Projection by way of an example.
###### Data
We will be working with [KDD2012 datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) available in LibSVM format. 

The training dataset has 119,705,032 rows and 54,686,452 features - we will apply Random Projection and transform this dataset into one which has 119,705,032 rows and ~4k features. In order to demonstrate the scalability of my approach, I'm going to artifically increase the size of the training set by a factor of 9 - giving us just over 1 Billion rows.

Note, even the transformed dataset needs to be in sparse format otherwise the dataset could take up ~16TB in dense format.

###### Compute Resources
I will be using a 6-node (90-core, 300G RAM) Spark [HDInsight](https://azure.microsoft.com/en-gb/services/hdinsight/) Cluster from [Microsoft Azure](https://azure.microsoft.com/en-gb/)  
pssst - Microsoft is giving away **£150** in _free credit_ to experiment in Azure.

I will also be using [Azure Storage](https://azure.microsoft.com/en-gb/services/storage/), particularly the Blob Service to store my input/output datasets. 

It roughly takes 1ms per row (single core performance) including I/O to process the data.
To project a dataset with 1 billion rows and 54 million columns into a new dataset with 1 billion rows and 4096 features took me about 3.75 hours on the above mentioned cluster.
###### Memory Consideration:
Your spark executors would need enough RAM to hold 2 x size of a Spark DataFrame partition plus a projection matrix (this is usually a few 100 MBs). If you are running short of memory then you can repartition your Spark DataFrame to hold smaller chunks of data.

###### Code walkthrough:
There are two versions of random projection code depending on whether you want to run the code on a single machine or on a cluster. Refer to  `./code/localmode/` or `./code/clustermode/` depending on your requirement. But here I will discuss the cluster mode code.  
 

**In [1]:** First, let's get the required imports out of the way. Of particular note is the ***johnson_lindenstrauss_min_dim*** function - this function, given an input matrix, returns the no. of dimensions the projected space should have. In my case I'm going to fix the no. of dimensions at 4096.
```python
# imports
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
```
**In [2]:** Below we will generate a Sparse Random Projection Matrix which will be a Scipy's CSR matrix which needs to be saved to disk to transform future data.

Note: the dimensionality of the reduced dimension space is independent of the no. of features (in our case 54 million) - it only depends on the no. of rows (in our case 1 Billion) according to the **_The Johnson-Lindenstrauss lemma_**.
```python
# generating the random projection  matrix
dummy_X_rows = train.count() # 1,077,345,288 rows
dummy_X_cols = train.first()["features"].size # 54,686,452 features
# we only use the shape-info of this dummy matrix
dummy_X = ssp.csr_matrix((dummy_X_rows, dummy_X_cols), dtype=np.float32)  

# find the optimal (conservative estimate) no. of dimensions required according to the
# johnson_lindenstrauss_min_dim function
# rproj_ndim = johnson_lindenstrauss_min_dim(dummy_X_rows, eps=0.2) # returns 4250

rproj_ndim = 4096 # using a fixed value instead of johnson_lindenstrauss_min_dim suggestion

logging.info("Fitting Sparse Random Projection to have %d columns after projection " % rproj_ndim)
srp = SparseRandomProjection(n_components=rproj_ndim, random_state=123)

srp.fit(dummy_X)
logging.info("Saving the projection matrix to disk... ")
joblib.dump(srp, os.path.join("/tmp","srp.pkl"))
```

**In [3]:** Here we will define a Python function which takes a whole Spark DataFrame partition and a projection matrix to return projected data.
In a nutshell, this function converts a whole Spark DataFrame partition into a Scipy CSR matrix and then simply mulitplies it with our projection matrix.
```python
def random_project_mappartitions_function(rdd_row_iterator, local_csr_matrix):
    """
    This function is intended to be used in a <df>.rdd.mapPartitions(lambda rdd_map: random_project_mapf\
    (rdd_map, local_rnd_mat))
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

        feature_mat = ssp.coo_matrix((data, (row_indices, col_indices)), 
                                    shape=(1, feature_dim_size)).tocsr()
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
```
**In [4]:** and here is where the main action takes place
```python
if __name__ == '__main__':
    N_FEATURES=54686452 # these are the no. of features in the dataset we are goining to use in this app
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
        .mapPartitions(lambda rdd_map_partition: 
            random_project_mappartitions_function(rdd_map_partition,local_rnd_mat_bc_var.value))\
        .toDF(train_schema)

    logging.info("Writing projected data to disk...")
    train_projected_df\
    .write\
    .mode("overwrite")\
    .parquet(DATA_DIR+"/train_features_random_projected.parquet/")

    logging.info("Sample rows from training set before projection...")
    print(train.show())
    logging.info("Sample rows from training set after projection...")
    print(spark.read.parquet(DATA_DIR+"/train_features_random_projected.parquet/").show())

    spark.stop()
```

**In [5]:** Finally, here is how we submit the PySpark app to the cluster.
```bash
#!/usr/bin/env bash
export AZURE_STORAGE_ACCOUNT=<storage account name>
export AZURE_STORAGE_ACCESS_KEY=<storage account access key>
echo "Submitting PySpark app..."
spark-submit  \
--master yarn \
--executor-memory 3G \
--driver-memory 6G \
--num-executors 85 \
--executor-cores 1 \
code/clustermode/randomProjection.py

echo "Exporting Projection Matrix to Azure Storage..."
zip /tmp/srp.pkl.zip /tmp/srp.pkl*
azure storage blob upload --container modelling-outputs --file /tmp/srp.pkl.zip
```