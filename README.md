A PySpark application that efficiently computes text similarity across large datasets using MinHash Locality Sensitive Hashing (LSH) and Jaccard distance, optimized for parallel and distributed processing.
Requirements
Python 3.x

pyspark library installed (pip install pyspark)

Usage
Run locally with default settings:
python script_name.py

Run with custom parameters (e.g., 50k records, 4 cores, 16 partitions):
python script_name.py --nodes 4 --partitions 16 --records 50000 --threshold 0.6

Run on a dedicated Spark cluster:
python script_name.py --master spark://192.168.1.10:7077 --records 100000

CLI Arguments
--nodes : Number of local threads/cores (default: 2)

--partitions : Number of Spark shuffle partitions (default: 8)

--records : Number of mock documents to generate (default: 10000)

--threshold : Minimum similarity threshold from 0.0 to 1.0 (default: 0.5)

--master : External Spark master URL. Omit to run locally

--driver-host : Driver IP address for distributed networking

Output
Console: Prints execution time per stage and a table of the Top 10 most similar document pairs.

results.csv: Appends a new row (nodes, partitions, records, total_time) after each run to help track and chart performance scaling.
