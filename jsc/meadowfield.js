const roads = [
  "Alice's House-Bob's House", 
  "Alice's House-Post Office", 
  "Daria's House-Ernie's House", 
  "Ernie's House-Grete's House", 
  "Grete's House-Shop", 
  "Marketplace-Post Office", 
  "Marketplace-Town Hall",
  "Alice's House-Cabin", 
  "Bob's House-Town Hall", 
  "Daria's House-Town Hall", 
  "Grete's House-Farm", 
  "Marketplace-Farm", 
  "Marketplace-Shop", 
  "Shop-Town Hall"
];

function buildGraph(roadList) {
  let graph = Object.create(null)
  function addEdge(u, v) {
    if (graph[u] == null)
      graph[u] = [v];
    else graph[u].push(v);
  }
  let edges = roadList.map(road => road.split("-"))
  for (let [u, v] of edges) {
    addEdge(u, v);
    addEdge(v, u)
  }
  return graph;
}

const roadGraph = buildGraph(roads)
console.log(roadGraph)

class VillageState {
  constructor(place, parcels) {
    this.place = place;
    this.parcels = parcels; // a parcel has place and address information
  }
  move(destination) {
    if (!roadGraph[this.place].includes(destination))
      return this;
    else {
      let parcels = this.parcels.map(p => {
        if (this.place != p.place) return p;
        return {place: destination, address: p.address};
      }).filter(p => p.place != p.address)
      return new VillageState(destination, parcels);
    }
  }
}

let first = new VillageState("Post Office", [{place: "Post Office", address: "Alice's House"}]);
let next = first.move("Alice's House");
console.log(next);
console.log(first)
