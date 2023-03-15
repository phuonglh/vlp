// case-insensitive with i at the end
let cartoonCrying = /boo+(hoo+)+/i; 
console.log(cartoonCrying.test("Boohoooohoohooo"));

// matches and groups
let match = /\d+/.exec("one two 100"); 
console.log(match);
console.log(match.index)