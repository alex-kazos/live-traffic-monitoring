# Base image: official Apache Spark 3.5.3
# Ubuntu 20.04 (Focal) — Python 3.8 already pre-installed
FROM apache/spark:3.5.3

# Switch to root to install packages
USER root

# 1. Upgrade pip (pre-installed pip 20 doesn't support all features)
#    then install PySpark and kafka-python for the alerts consumer
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    pyspark==3.5.3 \
    kafka-python-ng==2.2.2

# 2. Download the Kafka connector JAR for Spark Structured Streaming
#    at build time so the container doesn't need internet access at runtime.
#    These JARs go into SPARK_HOME/jars/ and are picked up automatically.
RUN curl -fL -o /opt/spark/jars/spark-sql-kafka.jar \
    "https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.3/spark-sql-kafka-0-10_2.12-3.5.3.jar" \
    && curl -fL -o /opt/spark/jars/kafka-clients.jar \
    "https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar" \
    && curl -fL -o /opt/spark/jars/spark-token-provider.jar \
    "https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.3/spark-token-provider-kafka-0-10_2.12-3.5.3.jar" \
    && curl -fL -o /opt/spark/jars/commons-pool2.jar \
    "https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar"

WORKDIR /app

COPY Spark/analytics.py        .
COPY Spark/alerts_consumer.py  .

# Drop back to the non-root spark user
USER 185
