const fetch = require("node-fetch");

fetch("http://vlp.group:9200/_cat/indices?v")
  .then(response => response.text())
  .then(console.log);