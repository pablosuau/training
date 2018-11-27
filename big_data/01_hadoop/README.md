This folder contains code I created to practice concepts related to big data tools such as Hadoop and Spark. Both the `data/` and `src/` folders are split into subfolders corresponding to different subprojects. 

## 01_hadoop_word_count

[[code]](src/01_hadoop_word_count/)
[[data]](data/01_hadoop_word_count/)

A classical word count example implemented for Hadoop's MapReduce. The code counts the frequency of each word in Bram Stoker's Dracula, as downloaded from the [Project Gutenberg website](https://www.gutenberg.org/).

To run the code locally:

```
cat data/01_hadoop_word_count/book.txt | \
   python src/01_hadoop_word_count/mapper.py | \
   sort | python src/01_hadoop_word_count/reducer.py
```

To test the code in a Hadoop environment based on the [Hadoop Python streaming Docker image](https://github.com/audip/hadoop-python-streaming):

* Run the Docker container (the Docker image will be automatically pulled if it does not already in the system): `docker run -it sequenceiq/hadoop-docker /etc/bootstrap.sh -bash`
* Download the Streamer code: `curl -O http://central.maven.org/maven2/org/apache/hadoop/hadoop-streaming/2.7.3/hadoop-streaming-2.7.3.jar` 
* Copy files to the Docker container from the host machine: `docker cp src/01_hadoop_word_count/mapper.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/01_hadoop_word_count/reducer.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp data/01_hadoop_word_count/book.txt [CONTAINER_ID]:/usr/local/hadoop`
* In the Docker container, copy the data to HDFS: `/usr/local/hadoop/bin/hdfs dfs -mkdir  data/` and `/usr/local/hadoop/bin/hdfs dfs -put /usr/local/hadoop/book.txt data/`
* Make sure that the output directory is deleted before running the MapReduce job, if that directory already exists: `/usr/local/hadoop/bin/hdfs dfs -rm -r -skipTrash output`

* Finally, launch the MapReduce job:

```
/usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.3.jar \
   -file /usr/local/hadoop/mapper.py \
   -mapper /usr/local/hadoop/mapper.py \
   -file /usr/local/hadoop/reducer.py \
   -reducer /usr/local/hadoop/reducer.py \
   -input data/book.txt \
   -output output
```

## 02_hadoop_top_k

[[code]](src/02_hadoop_top_k/)
[[data]](data/02_hadoop_top_k/)

The aim of this job is to produce a list with the top 10 visiting URLs according to the [NASA-HTTP log dataset](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html). The data is stored in compressed format in the repository:

```
gzip -dk data/02_hadoop_top_k/NASA_access_log_Jul95.gz
```

This job requires two chained MapReduce jobs. The first one is a usual word counter, whereas the second one sorts and extracts the URLs with the highest visit counts. This can be simulated from the command line as:

```
cat data/02_hadoop_top_k/NASA_access_log_Jul95 | \
   python src/02_hadoop_top_k/mapper_1.py | \
   sort | python src/02_hadoop_top_k/reducer_1.py | \
   python src/02_hadoop_top_k/mapper_2.py | \
   sort -n -r | python src/02_hadoop_top_k/reducer_2.py
```

*Note*: the first mapper raises an `UnicodeDecodeError` exception in Python 3. We still obtain a result, but this differs from the result obtained in Python 2. The Docker container uses Python 2.

We prepare the running environment similarly to what we did for the first job:

* Run the Docker container (the Docker image will be automatically pulled if it does not already in the system): `docker run -it sequenceiq/hadoop-docker /etc/bootstrap.sh -bash`
* Download the Streamer code: `curl -O http://central.maven.org/maven2/org/apache/hadoop/hadoop-streaming/2.7.3/hadoop-streaming-2.7.3.jar` 
* Copy files to the Docker container from the host machine: `docker cp src/02_hadoop_top_k/mapper_1.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/02_hadoop_top_k/mapper_2.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/02_hadoop_top_k/reducer_1.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/02_hadoop_top_k/reducer_2.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp data/02_hadoop_top_k/NASA_access_log_Jul95 [CONTAINER_ID]:/usr/local/hadoop` 
* In the Docker container, copy the data to HDFS: `/usr/local/hadoop/bin/hdfs dfs -mkdir  data/` and `/usr/local/hadoop/bin/hdfs dfs -put /usr/local/hadoop/NASA_access_log_Jul95 data/`
* Make sure that the output directories are deleted before running the MapReduce job, if those directories already exists: `/usr/local/hadoop/bin/hdfs dfs -rm -r -skipTrash output`, `/usr/local/hadoop/bin/hdfs dfs -rm -r -skipTrash output2`

Once the environment is ready, we can run the two MapReduce jobs. The first one is a simple word counter that counts the number of requests coming from each URL:

```
usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.3.jar \
   -file /usr/local/hadoop/mapper_1.py \
   -mapper /usr/local/hadoop/mapper_1.py \
   -file /usr/local/hadoop/reducer_1.py \
   -reducer /usr/local/hadoop/reducer_1.py \
   -input data/NASA_access_log_Jul95 \
   -output output
```

The second MapReduce job takes the output from the first one as its input. The mapper simply switches the key and values, so then we can use a comparator to sort the records numerically and in descending order by the number of requests. A single reducer is used. Its only task is to print the 10 first received records from the shuffle process.

```
/usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.3.jar \
   -D mapred.reduce.tasks=1 \
   -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
   -D mapred.text.key.comparator.options=-nr \
   -file /usr/local/hadoop/mapper_2.py \
   -mapper /usr/local/hadoop/mapper_2.py \
   -file /usr/local/hadoop/reducer_2.py \
   -reducer /usr/local/hadoop/reducer_2.py \
   -input output \
   -output output2
```

## 03_hadoop_salting

[[code]](src/03_hadoop_salting/)
[[data]](data/03_hadoop_salting/)

In this example we apply salting to deal with a skewed dataset. A skewed dataset is a dataset in which most of the records are assigned the same key. The implication of this is that one of the reducers will end up processing most of the records, and therefore the benefits of parallelism are lost. In this example, we calculate the average word length for each letter. The average length is calculated over all the words that contain that letter. After running a first version of this job using the book in the first program as input data, I found out that the frequency of letter e is much higher than that of the rest of letters. This does not make this data a skewed dataset, but allows me to practice the concept of salting by creating a chain of two MapReduce jobs. In the first one we add a different suffix to each half of the records for letter e and produce a first aggregation. In the second one we remove the suffixes and calculate the final aggregation. As usual, we can launch these jobs from command line for debugging purposes:

```
cat data/01_hadoop_word_count/book.txt | \
   python src/03_hadoop_salting/mapper_1.py | \
   sort | python src/03_hadoop_salting/reducer_1.py | \
   python src/03_hadoop_salting/mapper_2.py | \
   sort | python src/03_hadoop_salting/reducer_2.py
```

We prepare the running environment similarly to what we did for the first two jobs:

* Run the Docker container (the Docker image will be automatically pulled if it does not already in the system): `docker run -it sequenceiq/hadoop-docker /etc/bootstrap.sh -bash`
* Download the Streamer code: `curl -O http://central.maven.org/maven2/org/apache/hadoop/hadoop-streaming/2.7.3/hadoop-streaming-2.7.3.jar` 
* Copy files to the Docker container from the host machine: `docker cp src/03_hadoop_salting/mapper_1.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/03_hadoop_salting/mapper_2.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/03_hadoop_salting/reducer_1.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/03_hadoop_salting/reducer_2.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp data/01_hadoop_word_count/book.txt [CONTAINER_ID]:/usr/local/hadoop` 
* In the Docker container, copy the data to HDFS: `/usr/local/hadoop/bin/hdfs dfs -mkdir  data/` and `/usr/local/hadoop/bin/hdfs dfs -put /usr/local/hadoop/book.txt data/`
* Make sure that the output directories are deleted before running the MapReduce job, if those directories already exists: `/usr/local/hadoop/bin/hdfs dfs -rm -r -skipTrash output`, `/usr/local/hadoop/bin/hdfs dfs -rm -r -skipTrash output2`

Once the environment is ready, we can run the two MapReduce jobs. The first job returns the mean word length for each letter. We split the records for letter `e` in two groups by adding a different suffix to each group:

```
usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.3.jar \
   -file /usr/local/hadoop/mapper_1.py \
   -mapper /usr/local/hadoop/mapper_1.py \
   -file /usr/local/hadoop/reducer_1.py \
   -reducer /usr/local/hadoop/reducer_1.py \
   -input data/book.txt \
   -output output
```

The second MapReduce job produces the final aggregation after removing the salting suffix for the letter `e` groups:

```
/usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.3.jar \
   -file /usr/local/hadoop/mapper_2.py \
   -mapper /usr/local/hadoop/mapper_2.py \
   -file /usr/local/hadoop/reducer_2.py \
   -reducer /usr/local/hadoop/reducer_2.py \
   -input output \
   -output output2
```

