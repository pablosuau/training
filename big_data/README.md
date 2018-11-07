This folder contains code I created to practice concepts related to big data tools such as Hadoop and Spark. Both the `data/` and `src/` folders are split into subfolders corresponding to different subprojects. 

## 01_hadoop_word_count

[[code]](src/01_hadoop_word_count/)
[[data]](data/01_hadoop_word_count/)

A classical word count example implemented for Hadoop's MapReduce. The code counts the frequency of each word in Bram Stoker's Dracula, as downloaded from the [Project Gutenberg website](https://www.gutenberg.org/).

To run the code locally:

```
cat data/01_hadoop_word_count/book.txt | python src/01_hadoop_word_count/mapper.py | sort | python src/01_hadoop_word_count/reducer.py
```

To test the code in a Hadoop environment based on the [Hadoop Python streaming Docker image](https://github.com/audip/hadoop-python-streaming):

* Run the Docker container (the Docker image will be automatically pulled if it does not already in the system): `docker run -it sequenceiq/hadoop-docker /etc/bootstrap.sh -bash`
* Download the Streamer code: `curl -O http://central.maven.org/maven2/org/apache/hadoop/hadoop-streaming/2.7.3/hadoop-streaming-2.7.3.jar` 
* Copy files to the Docker container from the host machine: `docker cp src/01_hadoop_word_count/mapper.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp src/01_hadoop_word_count/reducer.py [CONTAINER_ID]:/usr/local/hadoop`, `docker cp data/01_hadoop_word_count/book.txt [CONTAINER_ID]:/usr/local/hadoop`
* In the Docker container, copy the data to HDFS: `/usr/local/hadoop/bin/hdfs dfs -mkdir  data/` and `/usr/local/hadoop/bin/hdfs dfs -put /usr/local/hadoop/book.txt data/`
* Make sure that the output directory is deleted before running the MapReduce job, if that directory already exists: `/usr/local/hadoop/bin/hdfs dfs -rm -r output`

* Finally, launch the MapReduce job:

```
/usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.3.jar -file /usr/local/hadoop/mapper.py -mapper /usr/local/hadoop/mapper.py -file /usr/local/hadoop/reducer.py -reducer /usr/local/hadoop/reducer.py -input data/book.txt -output output
``

## 02_hadoop_top_k

Based on the [NASA-HTTP dataset](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html). The data is stored in the repository in compressed format. In order to use it:

```
gzip -dk data/02_hadoop_top_k/NASA_access_log_Jul95.gz
```
