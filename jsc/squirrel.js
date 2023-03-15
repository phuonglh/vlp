let journal = [];

function addEntry(events, squirrel) {
  journal.push({events, squirrel})
}

console.log(journal)
addEntry(["work", "coffee", "touched tree", "learning"], false)
addEntry(["work", "coffee", "play", "watch", "programming"], true)
console.log(journal)

for (let entry of journal) {
  console.log(`${entry.events.length} events.`);
}

console.log("Journal in JSON format:")
console.log(JSON.stringify(journal))