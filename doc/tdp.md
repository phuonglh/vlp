
1. Check the existence of trained model in folder `/opt/models/tdp/`. This directory contains all necessary models to run the transition-based dependency parser.

2. Start the Spark JobServer:
    - Go into `spark-jobserver`
    - Invoke the command `sbt`
    - Start the server: `job-server-extra/reStart`
    - Check the server status: `http://localhost:8090`

3. Upload the JAR file `sjs.jar` to the server
       `curl --data-binary @sjs/target/scala-2.11/sjs.jar -X POST localhost:8090/binaries/sjs -H "Content-Type: application/java-archive"`

4. Create a persistent context with a name `vlp`, use 4 CPU cores and 1GB RAM:
       `curl -d "" "localhost:8090/contexts/vlp?num-cpu-cores=4&memory-per-node=1g"`
   You can change these values to use more CPU cores and RAM if the model is bigger.
   
5. Export base URLs for the two service
       `export SJS_URL="http://localhost:8090/jobs?appName=sjs&classPath=vlp.tdp.ParserJob&context=vlp&sync=true"`
       `export SJS_URL="http://112.137.134.8:8090/jobs?appName=sjs&classPath=vlp.tdp.ParserJob&context=vlp&sync=true"`
        
The setup is now complete. We can test from command line:
      `curl -d "input = \"Nên/SCONJ/SC trước_nhất/N/N người/N/Nc đảng_viên/N/N phải/VERB/V làm_gương/VERB/V ./PUNCT/PUNCT\"" -H "Content-Type: text/plain; charset=UTF-8"  $SJS_URL`
