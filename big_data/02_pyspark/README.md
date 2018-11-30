This folder contains small exercises I wrote to practice pyspark concepts.

All the notebooks in this folder can be executed from within the following Docker container (the Docker image will be pulled into our system if we did not download it yet):

```
docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work/ jupyter/pyspark-notebook
```

If we run the command above from the directory in which this README.md file is located we will be able to access all the pyspark notebooks from the `work` directory in the container.

The notebooks in this folder run Spark jobs locally. Since the Docker container above does not generate a local Spark infrastructure, we can only execute these jobs in development mode, that is, within a [single JVM process](https://stackoverflow.com/questions/39986507/spark-standalone-configuration-having-multiple-executors). Therefore, there is no real multiprocessing. This fact does not have an effect on the real purpose of these notebookes: to practice pyspark coding. 

In terms of data, I am using the [Google Play dataset provided by Kaggle](https://www.kaggle.com/lava18/google-play-store-apps). The data is stored in the `data/` folder.
