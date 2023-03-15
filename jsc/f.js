function isPrime(k) {
  if (k < 2) return false;
  for (let i = 2; i < Math.sqrt(k); i++)
    if (k %i == 0)
      return false;
  return true;
}

console.log(isPrime(97))
console.log(isPrime(20))

function minus(a, b) {
  if (b == undefined) return -a;
  else return a - b;
}

console.log(minus(10))
console.log(minus(10, 6))

function multiply(factor) {
  return number => number * factor;
}

let twice = multiply(2)
console.log(twice(5))

function findSolution(target) {
  function find(current, history) {
    if (current == target)
      return history;
    else {
        if (current > target)
          return null;
        else return find(current + 5, `(${history} + 5)`) || find(current * 3, `(${history} * 3)`)
    }
  }
  return find(1, "1")
}

console.log(findSolution(39))
console.log(`findSolution is of type: ${typeof findSolution}.`)

function repeat(n, action) {
  for (let i = 0; i < n; i++)
    action(i);
}

let xs = [];
repeat(5, i => xs.push(`Unit ${i}`))
console.log(xs)

let as = [1, 2, 3, 4, 5];
let sum = as.reduce((a, b) => a + b)
console.log(sum)