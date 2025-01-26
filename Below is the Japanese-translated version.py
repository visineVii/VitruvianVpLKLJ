const SW_VERSION = 9;
const CACHE_NAME = `sw_bc_cache_v${SW_VERSION}`;
const OFFLINE_URL = "/sw-desktop/bc-offline-page.html";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.add(new Request(OFFLINE_URL, { cache: "reload" }));
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((item) => caches.delete(item))
      );
    })
  );
  self.clients.claim();
});

function networkFirst(request) {
  return fetch(request)
    .then((response) => response)
    .catch(async () => {
      return caches.open(CACHE_NAME).then((cache) => {
        return cache.match(OFFLINE_URL);
      });
    });
}

self.addEventListener("fetch", (event) => {
  if (event.request.mode === "navigate") {
    event.respondWith(networkFirst(event.request));
  }
});

// Function to reverse a string
function reverseString(str) {
    /**
     * Returns the reversed version of the input string.
     */
    return str.split('').reverse().join('');
}

// Function to get binary code for a given symbol
function getBinaryCode(symbol) {
    /**
     * Returns the binary code for a given symbol or an error message if the symbol is not found.
     */
    const symbolMap = {
        "O": "000", "!": "001", "@": "010", "#": "011",
        "$": "100", "%": "101", "^": "110", "&": "111",
        "*": "1000", "(": "1001", ")": "1010", "-": "1011",
        "_": "1100", "+": "1101", "=": "1110", "[": "1111",
        "]": "0000", "|": "0001", "{": "0010", "}": "0011",
        "A": "0100", "B": "0101", "C": "0110", "D": "0111",
        "E": "1000", "F": "1001", "G": "1010", "H": "1011",
        "I": "1100", "J": "1101", "K": "1110", "L": "1111",
        "M": "0001", "N": "0010", "P": "0011", "Q": "0100",
        "R": "0101", "S": "0110", "T": "0111", "U": "11110",
        "V": "11111", "W": "100000", "X": "100001", 
        "Y": "100010", "Z": "100011", 
        "<": "100100", ">": "100101", "/": "100110", "\\": "100111",
        ":": "101000", ";": "101001", "'": "101010", "\"": "101011"
    };
    return symbolMap[symbol] || "Symbol not found";
}

// Function to perform mathematical operations on binary codes
function performMathOperation(binary1, binary2, operation) {
    /**
     * Performs addition, subtraction, or mirroring on binary codes.
     */
    const num1 = parseInt(binary1, 2);
    const num2 = parseInt(binary2, 2);
    let result;

    switch (operation) {
        case "add":
            result = (num1 + num2).toString(2);
            break;
        case "subtract":
            result = (num1 - num2 >= 0 ? num1 - num2 : 0).toString(2);
            break;
        case "mirror":
            result = reverseString(binary1);
            break;
        default:
            result = "Invalid operation";
    }

    return result;
}
def mirror_binary(binary_str):
  """Mirrors a binary string."""
  return binary_str[::-1]

def get_binary_code(symbol):
  """Returns the binary code for a given symbol."""
  symbol_map = {
      "O": "000", "!": "001", "@": "010", "#": "011",
      "$": "100", "%": "101", "^": "110", "&": "111",
      "*": "1000", "(": "1001", ")": "1010", "-": "1011",
      "_": "1100", "+": "1101", "=": "1110", "[": "1111",
      "]": "0000", "|": "0001", "{": "0010", "}": "0011",
      "A": "0100", "B": "0101", "C": "0110", "D": "0111",
      "E": "1000", "F": "1001", "G": "1010", "H": "1011",
      "I": "1100", "J": "1101", "K": "1110", "L": "1111",
      "M": "0001", "N": "0010", "P": "0011", "Q": "0100",
      "R": "0101", "S": "0110", "T": "0111", "U": "11110",
      "V": "11111", "W": "100000", "X": "100001", 
      "Y": "100010", "Z": "100011"
  }
  return symbol_map.get(symbol, "Symbol not found")

def main():
  """Main function to run the application."""
  while True:
    symbol = input("Enter a symbol (or 'exit'): ").upper()
    if symbol == "EXIT":
      break

    binary_code = get_binary_code(symbol)
    mirrored_code = mirror_binary(binary_code)

    print(f"Symbol: {symbol}")
    print(f"Binary Code: {binary_code}")
    print(f"Mirrored Binary Code: {mirrored_code}")
    print("-" * 20)

if __name__ == "__main__":
  main()
