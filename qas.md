## Introduction

## Installation
- Install an ElasticSearch server, run it as a daemon at port 9200: `bin/elasticsearch -d -p 9200`.
- Index a FAQ dataset into an ElasticSearch server (ESS) by running `vlp.qas.Indexer`. This will create an index `qas`. There should be an index with 3,794 samples.
- Check the index by curling `"http://localhost:9200/_cat/indices?v"` or `"http://localhost:9200/qas/_search?pretty"`.
- Start a ranker server by running `vlp.qas.RankerServer`. A ranker service will be created and bound to the port 8085 of the local host.

## Running

- Search for a query: 
  `curl -X POST -H "Content-Type: application/json; charset=utf-8" -d "{\"query\" : \"trang bị\"}" http://localhost:8085/search`