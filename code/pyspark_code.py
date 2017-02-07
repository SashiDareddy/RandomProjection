N_FEATURES=54686452
train_df= spark.read.format("libsvm").load("/Users/sashi/IdeaProjects/RandomProjection/data/kdd12.tr", numFeatures=N_FEATURES)

train_df.count() #119,705,032