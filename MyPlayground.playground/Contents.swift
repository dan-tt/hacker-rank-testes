import UIKit

func minimumSwaps(brackets: String) -> Int {
    var swaps = 0
    var openStack: [Character] = []
    var closeStack: [Character] = []

    for char in brackets {
        if char == "(" {
            if let _ = closeStack.popLast() {
                
            } else {
                openStack.append(char)
            }
        } else if char == ")" {
            // If the stack is empty or if the top of the stack doesn't match the current closing bracket
            if let _ = openStack.popLast() {
                
            } else {
                closeStack.append(char)
                swaps += 1
            }
        }
    }
    
    // If stack is not empty, sequence is unbalanced
    return (openStack.isEmpty && closeStack.isEmpty) ? swaps : -1
}

func isMatchingPair(_ left: Character, _ right: Character) -> Bool {
    return (left == "(" && right == ")")
}

// Example usage
let brackets1 = "()())()"
let minSwaps1 = minimumSwaps(brackets: brackets1)

let brackets2 = "()((()"
let minSwaps2 = minimumSwaps(brackets: brackets2)

let brackets3 = "(()))(" // Empty string
let minSwaps3 = minimumSwaps(brackets: brackets3)

let brackets4 = "()())))(((" // Empty string
let minSwaps4 = minimumSwaps(brackets: brackets4)

/*
 You have an infinite number of 4 types of lego blocks of sizes given as (d x h x w):

 d    h    w
 1    1    1
 1    1    2
 1    1    3
 1    1    4
 Using these blocks, you want to make a wall of height n and width m. Features of the wall are:

 - The wall should not have any holes in it.
 - The wall you build should be one solid structure, so there should not be a straight vertical break across all rows of bricks.
 - The bricks must be laid horizontally.

 How many ways can the wall be built?

 Function Description

 Complete the legoBlocks function in the editor below.

 legoBlocks has the following parameter(s):

 int n: the height of the wall
 int m: the width of the wall
 Returns
 - int: the number of valid wall formations modulo (10^9 + 7)
 
 */

func legoBlocks(n: Int, m: Int) -> Int {
    let mod = 1000000007
    var dp = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)

    // Base case: there's only one way to build a wall with zero height
    dp[0][0] = 1

    // Fill the dp array
    for i in 1...n {
        for j in 1...m {
            for k in 1...4 {
                if j >= k {
                    dp[i][j] += dp[i - 1][j - k]
                }
            }
        }
    }
    print(dp)
    return dp[n][m]
}

// MARK: - flippingMatrix

func flippingMatrix(matrix: [[Int]]) -> Int {
    let n = matrix.count / 2
    var totalSum = 0
    
    for i in 0..<n {
        for j in 0..<n {
            let maxElement = max(matrix[i][j], matrix[2 * n - i - 1][j], matrix[i][2 * n - j - 1], matrix[2 * n - i - 1][2 * n - j - 1])
            print("Max element: \(maxElement)")
            totalSum += maxElement
        }
    }
    
    return totalSum
}

// Example usage:
let matrix = [
    [112, 42, 83, 119],
    [56, 125, 56, 49],
    [15, 78, 101, 43],
    [62, 98, 114, 108]
]

let maxSum = flippingMatrix(matrix: matrix)
print("Maximum sum of elements in the upper-left quadrant: \(maxSum)")

// MARK: - PALINDROME INDEX
func isPalindrome(_ s: String, _ left: Int, _ right: Int) -> Bool {
    var l = left
    var r = right
    let chars = Array(s)
    
    while l < r {
        if chars[l] != chars[r] {
            return false
        }
        l += 1
        r -= 1
    }
    return true
}

func palindromeIndex(s: String) -> Int {
    let chars = Array(s)
    var left = 0
    var right = s.count - 1
    
    while left < right {
        if chars[left] != chars[right] {
            if isPalindrome(s, left+1, right) {
                return left
            }
            if isPalindrome(s, left, right-1) {
                return right
            }
            return -1
        }
        left += 1
        right -= 1
    }
    
    return -1
}

// Example usage:
let str1 = "abca"
let str2 = "racecar"
let str3 = "abbac"
let str4 = "abcd"

print("Palindrome index for \(str1): \(palindromeIndex(s: str1))") // Output: 3
print("Palindrome index for \(str2): \(palindromeIndex(s: str2))") // Output: -1
print("Palindrome index for \(str3): \(palindromeIndex(s: str3))") // Output: -1
print("Palindrome index for \(str4): \(palindromeIndex(s: str4))") // Output: 0

// MARK: - getTotalX
// Find the greatest common divisor (GCD) of all the elements in the second array
func gcd(_ a: Int, _ b: Int) -> Int {
    if b == 0 {
        return a
    }
    return gcd(b, a % b)
}
// Find the least common multiple (LCM) of all the elements in the first array
func lcm(_ a: Int, _ b: Int) -> Int {
    return (a * b) / gcd(a, b)
}

func getTotalX(a: [Int], b: [Int]) -> Int {
    let lcmA = a.reduce(1, lcm)
    let gcdB = b.reduce(b[0], gcd)
    
    var count = 0
    var multiple = lcmA
    
    while multiple <= gcdB {
        if gcdB % multiple == 0 {
            count += 1
        }
        multiple += lcmA
    }
    
    return count
}

// MARK: - ANAGRAM
func anagram(s: String) -> Int {
     let length = s.count
    guard length % 2 == 0 else {
        // Input string cannot be split into two substrings of equal length
        return -1
    }
    
    let mid = length / 2
    
    var firstHalfFreq = [Character: Int]()
    var secondHalfFreq = [Character: Int]()
    
    for (i, char) in s.enumerated() {
        if i < mid {
            // Count characters in the first half
            firstHalfFreq[char, default: 0] += 1
        } else {
            // Count characters in the second half
            secondHalfFreq[char, default: 0] += 1
        }
    }
    
    var numOfCharsToChange = 0
    
    // Compare frequencies of characters present in both halves
    for (char, count1) in firstHalfFreq {
        if let count2 = secondHalfFreq[char] {
            // Subtract the minimum count of characters between the two substrings
            numOfCharsToChange += abs(count1 - count2)
            // Remove the character from the second half frequency map
            secondHalfFreq.removeValue(forKey: char)
        } else {
            // Characters present in the first half but not in the second half
            numOfCharsToChange += count1
        }
    }
    
    // Characters present in the second half but not in the first half
    for (_, count2) in secondHalfFreq {
        numOfCharsToChange += count2
    }
    
    return numOfCharsToChange / 2
}

// MARK: - DRAWING BOOK

func pageCount(n: Int, p: Int) -> Int {
    // Write your code here
    guard p > 1, p < n else { return 0 }
    
    if p == n - 1 {
        return n % 2 == 0 ? 1 : 0
    }
    
    if p > n / 2 {
        return (n - p) / 2
    }
    
    return p / 2
    
}


// Example usage:
let n = 5 // Total number of pages
let p = 4 // Page to turn to
let minTurns = pageCount(n: n, p: p)
print("Minimum number of pages to turn:", minTurns) // Output: 0

// MARK: - TOWER BREAKERS

func towerBreakers(n: Int, m: Int) -> Int {
    // Write your code here
    guard m > 1 else {
        return 2
    }
    guard n > 1 else {
        return 1
    }
    // Calculate the nim-sum of tower heights
    var nimSum = 0
    for _ in 0..<n {
        nimSum ^= m
    }
    
    // Return the winner based on nim-sum
    return nimSum == 0 ? 2 : 1
}

// MARK: - Caesar Cipher

func caesarCipher(s: String, k: Int) -> String {
    // Write your code here
    let alphabet = "abcdefghijklmnopqrstuvwxyz"
    let shiftedAlphabet = alphabet + alphabet
    let alphabetUpper = alphabet.uppercased()
    let shiftedAlphabetUpper = alphabetUpper + alphabetUpper
    let shiftAmount = k % alphabet.count
    
    var encryptedText = ""
    
    func encryptedChar(_ char: Character) -> Character {
        if let index = alphabet.firstIndex(of: char) {
            let shiftedIndex = shiftedAlphabet.index(index, offsetBy: shiftAmount)
            return shiftedAlphabet[shiftedIndex]
        }
        
        if let index = alphabetUpper.firstIndex(of: char) {
            let shiftedIndex = shiftedAlphabetUpper.index(index, offsetBy: shiftAmount)
            return shiftedAlphabetUpper[shiftedIndex]
        }
        
        return char
    }

     s.forEach { char in
        encryptedText.append(encryptedChar(char))
     }
     
     return encryptedText
}

// MARK: - maxMin
func maxMin(k: Int, arr: [Int]) -> Int {
    // Write your code here
    let n = arr.count
    guard n >= 2, k <= n else { return -1 }
    // Sort the array in ascending order
    let sortedArr = arr.sorted()
    
    var minUnfairness = Int.max
    
    // Iterate through the array and create subarrays of length k
    for i in 0...(n - k) {
        let subarray = Array(sortedArr[i..<(i + k)])
        let unfairness = subarray.last! - subarray.first!
        minUnfairness = min(minUnfairness, unfairness)
    }
    
    return minUnfairness
}

// MARK: - dynamicArray
func dynamicArray(n: Int, queries: [[Int]]) -> [Int] {
    // Write your code here
    var arr = Array(repeating: [Int](), count: n)
    var lastAnswer = 0
    var answers = [Int]()
    
    for query in queries {
        let type = query[0]
        let x = query[1]
        let y = query[2]
        
        let idx = (x ^ lastAnswer) % n
        
        if type == 1 {
            arr[idx].append(y)
        } else if type == 2 {
            let idxInArr = y % arr[idx].count
            lastAnswer = arr[idx][idxInArr]
            answers.append(lastAnswer)
        }
    }
    
    return answers
}

// MARK: - gridChallenge

func gridChallenge(grid: [String]) -> String {
    // Write your code here
    // Convert each string into an array of characters
    let charGrid = grid.map { Array($0) }
    
    // Rearrange elements of each row alphabetically
    var sortedGrid = charGrid
    for i in 0..<sortedGrid.count {
        sortedGrid[i].sort()
    }
    
    // Check if columns are in ascending alphabetical order
    for j in 0..<sortedGrid[0].count {
        for i in 1..<sortedGrid.count {
            if sortedGrid[i][j] < sortedGrid[i - 1][j] {
                return "NO"
            }
        }
    }
    
    return "YES"
}

// MARK: - findPrimeDates

func findPrimeDates(_ date1: String, _ date2: String) -> Int {
    let calendar = Calendar.current
    
    func isLuckyDate(_ date: Date) -> Bool {
        let components = calendar.dateComponents([.day, .month, .year], from: date)
        if let day = components.day,
            let month = components.month,
            let year = components.year
        {
            let num = Int(String(format: "%02d%02d%04d", day, month, year))!
            if num % 4 == 0 || num % 7 == 0 {
                return true
            }
        }
        return false
    }
    
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "dd-MM-yyyy"
    
    guard let startDate = dateFormatter.date(from: date1),
          let endDate = dateFormatter.date(from: date2)
    else {
        return 0
    }
    
    var luckyCount = 0
    
    var currentDate = startDate
    while currentDate <= endDate {
        if isLuckyDate(currentDate) {
            luckyCount += 1
        }
        
        currentDate = calendar.date(byAdding: .day, value: 1, to: currentDate)!
    }
    
    return luckyCount
}

let stdout = ProcessInfo.processInfo.environment["OUTPUT_PATH"]!
FileManager.default.createFile(atPath: stdout, contents: nil, attributes: nil)
let fileHandle = FileHandle(forWritingAtPath: stdout)!

guard let arTemp = readLine()?.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression) else { fatalError("Bad input") }

let ar: [String] = arTemp.split(separator: " ").map {
    String($0)
}
let result = findPrimeDates(ar[0], ar[1])
fileHandle.write(String(result).data(using: .utf8)!)
fileHandle.write("\n".data(using: .utf8)!)

// MARK: - balancedSums
func balancedSums(arr: [Int]) -> String {
    // Write your code here
    let n = arr.count
    guard n > 1 else { return "YES" }
    
    let totalSum = arr.reduce(0, +)
    var leftSum = 0
    
    for num in arr {
        if leftSum == totalSum - num - leftSum {
            return "YES"
        }
        leftSum += num
    }
    
    return "NO"
}

// MARK: - superDigit

func superDigit(n: String, k: Int) -> Int {
    // Write your code here
    let digitSum = superDigit(of: n)
    let superDigitSum = (digitSum * k) % 9
    return superDigitSum == 0 ? (digitSum == 0 ? 0 : 9) : superDigitSum
}

func superDigit(of number: String) -> Int {
    if number.count == 1 {
        return Int(number)!
    }
    
    var sum = 0
    for char in number {
        sum += Int(String(char))!
    }
    return sum % 9 == 0 ? (sum == 0 ? 0 : 9) : sum % 9
}

// MARK: - counterGame

func counterGame(n: Int) -> String {
    // Write your code here
    var number = n
    var isLouiseTurn = true
    
    while number > 1 {
        // Check if number is a power of 2
        if number & (number - 1) == 0 {
            number /= 2 // Divide by 2 if it's a power of 2
        } else {
            // Find the next lower number which is a power of 2
            var powerOf2 = 1
            while powerOf2 <= number {
                powerOf2 *= 2
            }
            print(powerOf2 / 2)
            number -= powerOf2 / 2
        }
        
        // Toggle turn
        isLouiseTurn = !isLouiseTurn
    }
    // Determine winner based on whose turn it is
    return !isLouiseTurn ? "Louise" : "Richard"
}

// MARK: - bomberMan
func bomberMan(n: Int, grid: [String]) -> [String] {
    let rows = grid.count
    let cols = grid[0].count
    
    // Function to fill the grid with bombs
    func fillGridWithBombs() -> [[Character]] {
        Array(repeating: Array(repeating: "O", count: cols), count: rows)
    }
    
    var currentGrid = grid.map { Array($0) }

    var bombs = [(Int, Int)]()
    func findBombs() {
        bombs.removeAll()
        for i in 0..<rows {
            for j in 0..<cols {
                 if currentGrid[i][j] == "O" {
                    bombs.append((i, j))
                }
            }
        }
    }
    // Function to detonate bombs
    func detonateBombs(grid: [[Character]]) -> [[Character]] {
        var newGrid = grid
        let dx = [-1, 1, 0, 0]
        let dy = [0, 0, -1, 1]
        
        for (i, j) in bombs {
                    // Detonate bomb
                    newGrid[i][j] = "."
                    for k in 0..<4 {
                        let nx = i + dx[k]
                        let ny = j + dy[k]
                        if nx >= 0 && nx < rows && ny >= 0 && ny < cols {
                            newGrid[nx][ny] = "."
                        }
                    }
                }
        return newGrid
    }
    
    // Simulate Bomberman's actions
    guard n > 1 else {
        return grid
    }
    
    guard n % 2 != 0 else {
        // Fill grid with bombs
        currentGrid = fillGridWithBombs()
        return currentGrid.map { String($0) }
    }
    
    guard n > 3 else {
        findBombs()
        currentGrid = detonateBombs(grid: fillGridWithBombs())
        return currentGrid.map { String($0) }
    }
    
    for _ in 0...(n % 4) {
        findBombs()
        currentGrid = detonateBombs(grid: fillGridWithBombs())
    }
    
    return currentGrid.map { String($0) }
}

// Example usage:
let grid = [
    ".......",
    "...O.O.",
    "....O..",
    "..O....",
    "OO...OO",
    "OO.O..."
]
print(bomberMan(n: 5, grid: grid))

// MARK: - minimumBribes
func minimumBribes(q: [Int]) -> Void {
    // Write your code here
    var bribes = 0
    
    // Iterate through the queue
    for (index, person) in q.enumerated() {
        let position = index + 1
        
        // Check if the person has moved more than 2 positions forward
        if person - position > 2 {
            print("Too chaotic")
            return
        }
        
        // Count the number of bribes made by the person
        for j in max(0, person - 2)..<position {
            if q[j] > person {
                bribes += 1
            }
        }
    }
    
    print(bribes)
}

// MARK: - Sherlock isValid
func isValid(s: String) -> String {
    // Write your code here
    guard s.count > 1 else { return "YES" }
    // Count the frequency of each character
    var charCount = [Character: Int]()
    for char in s {
        charCount[char, default: 0] += 1
    }
    
    // Count the frequency counts
    var frequencyCount = [Int: Int]()
    for (_, count) in charCount {
        frequencyCount[count, default: 0] += 1
    }
    guard frequencyCount.count > 1 else {
        return "YES"
    }
    
    guard frequencyCount.count == 2 else {
        return "NO"
    }
    let distinctFrequencies = frequencyCount.keys.sorted()
    
    // If there is only one distinct frequency or if there are exactly two distinct frequencies,
    // and one of them occurs only once and its count is 1 greater than the other, then the string is valid
    if (distinctFrequencies.count == 2 &&
         (frequencyCount[distinctFrequencies[0]] == 1 && distinctFrequencies[0] == 1 ||
          frequencyCount[distinctFrequencies[1]] == 1 && distinctFrequencies[1] == distinctFrequencies[0] + 1)) {
        return "YES"
    } else {
        return "NO"
    }
}

// MARK: climbingLeaderboard
func climbingLeaderboard(ranked: [Int], player: [Int]) -> [Int] {
    var uniqueRanked = [Int]()
    var prevScore = Int.max
    
    // Constructing a sorted set of unique scores on the leaderboard
    for score in ranked {
        if score < prevScore {
            uniqueRanked.append(score)
            prevScore = score
        }
    }
    
    var result = [Int]()
    
    // Finding the rank for each score in the player's scores
    for score in player {
        var left = 0
        var right = uniqueRanked.count - 1
        
        // Binary search to find the rank
        while left <= right {
            let mid = left + (right - left) / 2
            if uniqueRanked[mid] <= score {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        
        // Adjusting the rank based on the binary search result
        let rank = left + 1
        result.append(rank)
    }
    
    return result
}

// MARK: - reverse(llist

final class SinglyLinkedListNode {
    var data: Int
    var next: SinglyLinkedListNode?

    public init(nodeData: Int) {
        self.data = nodeData
    }
}

final class SinglyLinkedList {
    var head: SinglyLinkedListNode?
    var tail: SinglyLinkedListNode?

    public init() {}

    public func insertNode(nodeData: Int) {
        self.insertNode(node: SinglyLinkedListNode(nodeData: nodeData))
    }

    private func insertNode(node: SinglyLinkedListNode) {
        if let tail = tail {
            tail.next = node
        } else {
            head = node
        }

        tail = node
    }
}

func printSinglyLinkedList(head: SinglyLinkedListNode?, sep: String, fileHandle: FileHandle) {
    var node = head

    while node != nil {
        fileHandle.write(String(node!.data).data(using: .utf8)!)

        node = node!.next

        if node != nil {
            fileHandle.write(sep.data(using: .utf8)!)
        }
    }
}

/*
 * Complete the 'reverse' function below.
 *
 * The function is expected to return an INTEGER_SINGLY_LINKED_LIST.
 * The function accepts INTEGER_SINGLY_LINKED_LIST llist as parameter.
 */

func reverse(llist: SinglyLinkedListNode?) -> SinglyLinkedListNode? {
    // Write your code here
    var prev: SinglyLinkedListNode? = nil
    var current = llist
    
    // Traverse the list and reverse the next pointers
    while current != nil {
        let nextNode = current?.next
        current?.next = prev
        prev = current
        current = nextNode
    }
    
    // Return the new head of the reversed list
    return prev
}

// MARK: - reverseDoubly
final class DoublyLinkedListNode {
    var data: Int
    var next: DoublyLinkedListNode?
    weak var prev: DoublyLinkedListNode?

    public init(nodeData: Int) {
        self.data = nodeData
    }
}

final class DoublyLinkedList {
    var head: DoublyLinkedListNode?
    var tail: DoublyLinkedListNode?

    public init() {}

    public func insertNode(nodeData: Int) {
        self.insertNode(node: DoublyLinkedListNode(nodeData: nodeData))
    }

    private func insertNode(node: DoublyLinkedListNode) {
        if let tail = tail {
            tail.next = node
            node.prev = tail
        } else {
            head = node
        }

        tail = node
    }
}

func printDoublyLinkedList(head: DoublyLinkedListNode?, sep: String, fileHandle: FileHandle) {
    var node = head

    while node != nil {
        fileHandle.write(String(node!.data).data(using: .utf8)!)

        node = node!.next

        if node != nil {
            fileHandle.write(sep.data(using: .utf8)!)
        }
    }
}

func reverse(llist: DoublyLinkedListNode?) -> DoublyLinkedListNode? {
    // Write your code here
    var current = llist
    var prev: DoublyLinkedListNode? = nil
    
    // Traverse the list and swap next and prev pointers
    while let node = current {
        let nextNode = node.next
        node.next = prev
        node.prev = nextNode
        prev = node
        current = nextNode
    }
    
    // Return the new head of the reversed list
    return prev
}

// MARK: - removeDuplicates
func removeDuplicates(llist: SinglyLinkedListNode?) -> SinglyLinkedListNode? {
    // Write your code here
    var current = llist
    
    while current != nil && current!.next != nil {
        if current!.data == current!.next!.data {
            current!.next = current!.next!.next
        } else {
            current = current!.next
        }
    }
    
    return llist
}

// MARK: - insertNodeAtPosition

func insertNodeAtPosition(llist: SinglyLinkedListNode?, data: Int, position: Int) -> SinglyLinkedListNode? {
    // Write your code here
    let newNode = SinglyLinkedListNode(nodeData: data)
    
    // If position is 0, insert newNode at the head
    guard position > 0 else {
        newNode.next = llist
        return newNode
    }
    
    var current = llist
    var currentPosition = 0
    
    // Traverse the list to find the position to insert newNode
    while currentPosition < position - 1 && current?.next != nil {
        current = current?.next
        currentPosition += 1
    }
    
    // Insert newNode at the desired position
    let nextNode = current?.next
    current?.next = newNode
    newNode.next = nextNode
    
    return llist
}


// MARK: - truckTour
func truckTour(petrolpumps: [[Int]]) -> Int {
    var start = 0 // Start index
    var petrolBalance = 0 // Current petrol balance

    var totalPetrol = 0
    var totalDistance = 0
    
    // Iterate through each petrol pump
    for i in 0..<petrolpumps.count {
        let pump = petrolpumps[i]
        guard pump.count == 2 else {
            fatalError("Each petrol pump should have exactly two values (petrol and distance)")
        }
        let petrol = pump[0]
        let distance = pump[1]
        totalPetrol += petrol
        totalDistance += distance
        
        // Update petrol balance
        petrolBalance += petrol - distance
        
        // If petrol balance is negative, reset start index to next petrol pump and reset petrol balance
        if petrolBalance < 0 {
            start = i + 1
            petrolBalance = 0
        }
    }
    
    // If total petrol is less than total distance, there is no solution
    if totalPetrol < totalDistance {
        return -1
    }
    
    return start
}

// MARK: - pairs
func pairs(k: Int, arr: [Int]) -> Int {
    var count = 0
    let set = Set(arr)
    
    // Iterate through each element in the array
    for num in arr {
        // Check if num + k exists in the set and num + k is not equal to num (to avoid counting the same pair twice)
        if set.contains(num + k) && num + k != num {
            count += 1
        }
        // Check if num - k exists in the set and num - k is not equal to num (to avoid counting the same pair twice)
        if set.contains(num - k) && num - k != num {
            count += 1
        }
    }
    
    // Return the count of pairs
    return count / 2 // Since we increment count twice for each pair, divide by 2 to get the correct count
}

// MARK: - bigSorting
func bigSorting(unsorted: [String]) -> [String] {
    return unsorted.sorted { (a, b) -> Bool in
        if a.count != b.count {
            return a.count < b.count // Sort based on length if lengths are different
        } else {
            return a < b // Sort lexicographically if lengths are the same
        }
    }
}

// MARK: - componentsInGraph
func dfs(node: Int, visited: inout Set<Int>, adjList: [[Int]]) -> Int {
    visited.insert(node)
    var componentSize = 1
    for neighbor in adjList[node] {
        if !visited.contains(neighbor) {
            componentSize += dfs(node: neighbor, visited: &visited, adjList: adjList)
        }
    }
    return componentSize
}

func componentsInGraph(gb: [[Int]]) -> [Int] {
    let n = gb.count
    var adjList = [[Int]](repeating: [], count: 2 * n + 1)
    for edge in gb {
        let u = edge[0], v = edge[1]
        adjList[u].append(v)
        adjList[v].append(u)
    }
    
    var visited = Set<Int>()
    var componentSizes = [Int]()
    
    for node in 1...(2 * n) {
        if !visited.contains(node) {
            let componentSize = dfs(node: node, visited: &visited, adjList: adjList)
            if componentSize > 1 {
                componentSizes.append(componentSize)
            }
        }
    }
    
    let smallestComponent = componentSizes.min() ?? 0
    let largestComponent = componentSizes.max() ?? 0
    
    return [smallestComponent, largestComponent]
}

// MARK: - findMedian
func findMedian(arr: [Int]) -> Int {
    // Write your code here
    // Step 1: Sort the list of numbers
    let sortedArr = arr.sorted()

    // Step 2: Determine the median
    let count = sortedArr.count
    guard count % 2 == 0 else {
        // Odd number of elements
        return sortedArr[count / 2]

    }
    
    // Even number of elements
    let mid1 = sortedArr[count / 2 - 1]
    let mid2 = sortedArr[count / 2]
    return (mid1 + mid2) / 2
}

// MARK: - modifiedFibonacci
func modifiedFibonacci(t1: Int, t2: Int, n: Int) -> String {
    guard n >= 3, n <= 20 else {
        fatalError("The value of n should be between 3 and 20")
    }
    
    var current = String(t1).map { Int(String($0))! }
    var next = String(t2).map { Int(String($0))! }
    
    for _ in 3...n {
        var carry = 0
        var result = [Int]()
        
        for i in 0..<max(current.count, next.count) {
            let sum = (i < current.count ? current[i] : 0) + (i < next.count ? next[i] : 0) + carry
            result.append(sum % 10)
            carry = sum / 10
        }
        
        if carry > 0 {
            result.append(carry)
        }
        
        current = next
        next = result
    }
    
    return next.reversed().map { String($0) }.joined()
}

// Example usage:
let _ = modifiedFibonacci(t1: 0, t2:  2, n: 10)

// MARK: - almostSorted
func almostSorted(_ array: [Int]) {
    // Check if the array is already sorted]
    let sortedArray = array.sorted()
    guard array != sortedArray else {
        print("yes")
        return
    }
    
    // Find misplaced elements
    var misplacedIndices = [Int]()
    for (i, element) in array.enumerated() {
        if element != sortedArray[i] {
            misplacedIndices.append(i)
        }
    }
    
    // If there are no misplaced elements, array is already sorted
    if misplacedIndices.isEmpty {
        print("yes")
        return
    }
    
    // If there are exactly two misplaced elements, try swapping
    if misplacedIndices.count == 2,
       let first = misplacedIndices.first,
       let last = misplacedIndices.last
    {
        let swappedArray = array.swapElements(at: first, and: last)
        
        // If swapping makes the array sorted, output swap
        if swappedArray == swappedArray.sorted() {
            print("yes")
            print("swap \(first + 1) \(last + 1)")
            return
        }
    }
    
    // If the array can be sorted by reversing a subsegment, output reverse
    if let first = misplacedIndices.first,
       let last = misplacedIndices.last
    {
        let subArr1 = array[0..<first]
        let subArr2 = array[first...last]
        let subArr3 = array[(last + 1)...]
        let reversedSubArr = Array(subArr2.reversed())
        let newSortedArray = Array(subArr1) + reversedSubArr + Array(subArr3)
        if newSortedArray == sortedArray {
            print("yes")
            print("reverse \(first + 1) \(last + 1)")
            return
        }
    }
    
    // If none of the above conditions are met, array cannot be sorted
    print("no")
}

extension Array where Element == Int {
    // Function to swap two elements in an array
    func swapElements(at index1: Int, and index2: Int) -> [Element] {
        var newArray = self
        newArray.swapAt(index1, index2)
        return newArray
    }
}

// Example usage:
//let array = [1, 5, 4, 3, 2, 6]
//let array = [3, 1, 2]
//let array = [3, 2, 1]
let array = [3, 2, 1, 4]
almostSorted(array)

// MARK: - connectedCells
func connectedCells(martix grid: [[Int]]) -> Int {
    var maxRegionSize = 0
    var grid = grid // Make a mutable copy of the grid
    
    for i in 0..<grid.count {
        for j in 0..<grid[0].count {
            if grid[i][j] == 1 {
                let regionSize = dfs(&grid, i, j)
                maxRegionSize = max(maxRegionSize, regionSize)
            }
        }
    }
    
    return maxRegionSize
}

// perform a depth-first search (DFS) traversal of the grid
func dfs(_ grid: inout [[Int]], _ row: Int, _ col: Int) -> Int {
    let numRows = grid.count
    let numCols = grid[0].count
    let directions: [(Int, Int)] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    var regionSize = 0
    if grid[row][col] == 1 {
        grid[row][col] = 0 // Mark cell as visited
        regionSize += 1
        
        for dir in directions {
            let newRow = row + dir.0
            let newCol = col + dir.1
            
            if newRow >= 0, newRow < numRows, newCol >= 0, newCol < numCols {
                regionSize += dfs(&grid, newRow, newCol)
            }
        }
    }
    
    return regionSize
}

