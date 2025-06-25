SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."
SIMPLE_CHAT_INSTRUCTION_V2 = """You are an AI that only responds with only {lang} code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."""

aggregate_prompt = """You are a programming assistant, who is helping user to write efficient and correct codes. You will be given multiple implementations of the same function. You should choose the {k} best implementation based on the following criterias:
1. Correctness: The implementation should return the correct output.
2. Efficiency: The implementation should be efficient in terms of time and space complexity.
3. Readability: The implementation should be readable and understandable.
4. Style: The implementation should follow the style guide of the language.
5. Testability: The implementation should be testable.

Remember your task is to choose the {k} best implementation based on the above criterias. Make sure to return only the indexes of the selected implementations, separated by commas. Do not include any other explanations, introduction, conclusions or thoughts. Just return the indexes of the selected implementations.
Function signature and docstring:
{prompt}
Implementations:
{implementations}
Chosen implementation:
"""

evaluation_prompt = """You are a programming assistant, who is helping the user to evaluate a generated code. You will be given a single implementation of a function, and you should evaluate it based on the following criteria:

1. **Correctness**: Does the implementation return the correct output for different inputs?
2. **Efficiency**: Is the implementation efficient in terms of time and space complexity?
3. **Readability**: Is the code readable and understandable? Is it easy to follow?
4. **Style**: Does the implementation follow the style guide of the language (naming conventions, indentation, etc.)?
5. **Testability**: Is the implementation testable? Can it be easily tested with unit tests?

Evaluate the code on each criterion with a score from 1 to 10 (integers only, no fractions). Then give an overall score as the sum of all scores.

Function signature and docstring:
{prompt}

Implementation:
{implementation}

Evaluation scores:
- Correctness: <score>
- Efficiency: <score>
- Readability: <score>
- Style: <score>
- Testability: <score>

Overall Score: <final score>

Do not include any further thoughts or reasoning, just the evaluation scores and the final overall score."""


SIMPLE_CHAT_INSTRUCTION_BFS = """
You are an AI that only responds with {lang} code. You will be given a function signature and its docstring by the user.
Write multiple full implementations (at least two), each restating the function signature. Use a different approach for each.
Mark the start and end of each implementation using triple backticks, like this:
\`\`\`
<Code implementation here>
\`\`\`
Each implementation should be fully contained within its own set of backticks, without any additional markers.
"""

react = """You are a programming assistant solving a coding task. Think step by step and plan your implementation carefully. You will be given a function signature and docstring, and you need to implement the function.

For each step:
1. Think about what needs to be done
2. Write code to implement that step
3. Consider edge cases and error handling

Example:
Function signature and docstring:
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''

Thought: I need to implement a simple addition function. The function takes two integers and returns their sum. I should handle basic input validation.

Action: ```python
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''
    # Input validation
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b
```

Thought: The implementation looks good. It includes:
1. Type hints for parameters and return value
2. Input validation to ensure both arguments are integers
3. Simple and efficient addition operation
4. Proper docstring preservation

Action: Finish[The implementation is complete and correct]

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Remember to think step by step and write clear, efficient code."""

self_evaluate_step = '''You are evaluating a reasoning step in a code generation task. Given the function signature, current implementation, and the proposed step, determine if this step is correct and logical. Consider:
1. Is the code syntactically correct?
2. Does it follow the function's requirements?
3. Is it a logical next step in the implementation?
4. Does it handle edge cases appropriately?

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Proposed step:
{step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a code generation task. Given the function signature, the implementation steps, and the final code, determine if the solution is correct. Consider:
1. Does the implementation match the function signature and docstring?
2. Is the code syntactically correct and follows language style guidelines?
3. Does it handle all edge cases and error conditions?
4. Is it efficient and readable?
5. Does it include appropriate tests or validation?

Function signature and docstring:
{prompt}

Implementation steps:
{steps}

Final code:
{answer}

Is this solution correct? Answer with a single word: Yes or No.
'''


reflect = """You are an advanced {lang} programming agent that can improve based on self reflections. You will be given the previous implementation of a function. The task was, given a function signature and its docstring by the user write the full implementation (restate the function signature). You were unsuccessful in answering the question either because you reached the wrong answer, or you used up your set number of reasoning steps, or your actions were inefficient. In a few sentences, diagnose a possible reason for failure or inefficiency and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  You have been given a specific output format to follow. Strictly follow the output format.
Output format:
Diagnosis: <your diagnosis>
Corrections required:
<problem 1> <explanation> <solution>
<problem 2> <explanation> <solution>
...
<problem n> <explanation> <solution>

(END OF OUTPUT FORMAT)

{examples}

(END OF EXAMPLES)

Previous trial:
function impl: {function_impl}
Reflection:"""

act_with_summary = """Here is a summary of reflections from previous attempts:
{summary}

Based on this, please provide a new implementation for:
{current_state}
"""

summarize_reflections = """You are a programming assistant. You will be given a list of reflections on previous failed attempts to solve a programming problem. Your task is to synthesize these reflections into a short, concise summary that can guide the next attempt. The summary should highlight the key mistakes and suggest a better approach.

Reflections:
{reflections}

Concise Summary:"""

examples_reflect_py = [
'''Previous trial:
function impl:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

Reflection:
Diagnosis: The implementation fails to handle cases where no subarray fulfills the condition.

Corrections required:
1. Incorrect condition operator <explanation> The use of >= in the condition to update the result causes it to return a subarray even when the sum exceeds the target, since it updates when current length equals previous longest length <solution> Replace >= with > in the condition to only update when strictly greater
2. Missing edge case handling <explanation> The implementation does not properly handle cases where no valid subarray exists <solution> Add a check to ensure the current_sum is within target before updating result
'''

'''
Previous trial:
function impl:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

Reflection:
Diagnosis: The implementation fails to handle cases where the sum of the subarray exceeds the target.

Corrections required:
1. Incorrect condition operator <explanation> The use of <= in the condition to update the result causes it to return a subarray even when the sum exceeds the target, since it updates when current length equals previous longest length <solution> Replace <= with < in the condition to only update when strictly less
2. Missing edge case handling <explanation> The implementation does not properly handle cases where no valid subarray exists <solution> Add a check to ensure the current_sum is within target before updating result
''',
'''Previous trial:
function impl:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

Reflection:
Diagnosis: The implementation fails to handle cases where the input integers are 1 and 2.

Corrections required:
1. Incorrect operator <explanation> The use of - in the return statement causes the function to subtract the second integer from the first <solution> Change the operator from - to + in the return statement
''',

'''Previous trial:
function impl:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```

Reflection:
Diagnosis: The implementation fails to handle cases where the input list of words is empty.

Corrections required:
1. Missing edge case handling <explanation> The implementation does not properly handle cases where no valid subarray exists <solution> Add a check to ensure the current_sum is within target before updating result
'''
]




examples_reflect_rs = [
'''Previous trial:
function impl:
```rust
fn add(a: i32, b: i32) -> i32 {
    // Given integers a and b, return the total value of a and b.
    a - b
}
```

Reflection
Diagnosis: The implementation fails to handle cases where the input integers are 1 and 2.

Corrections required:
1. Incorrect operator <explanation> The use of - in the return statement causes the function to subtract the second integer from the first <solution> Change the operator from - to + in the return statement
''',

'''Previous trial:
function impl:
```rust
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
// Given an array of strings strs, group the anagrams together. You can return the answer in any order.
// An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
  use std::collections::HashMap;
  let mut map: HashMap<[u8;26], Vec<String>> = HashMap::with_capacity(strs.len());
  let offset = 'a' as usize;

  for str in strs.into_iter() {
    let mut chars: [u8; 26] = [0; 26];

    for char in str.chars() {
      chars[char.to_ascii_lowercase() as usize - offset] += 1;
    }

    // Flaw: using str.len() instead of chars in the hashmap key
    map.entry(str.len())
      .and_modify(|v| v.push(str.clone()))
      .or_insert(vec![str]);
  }
  
  let mut arr: Vec<Vec<String>> = Vec::new();
  for v in map.into_values() {
    arr.push(v);
  }
  arr
}
```

Reflection:
Diagnosis: The implementation fails to group the anagrams together correctly.

Corrections required:
1. Incorrect hashmap key <explanation> The implementation uses the length of the input strings (str.len()) as the key for the hashmap, rather than the count of each character in the strings (chars) <solution> Change the hashmap key to the character count array (chars)
''',
'''Previous trial:
function impl:
```rust
fn reverse_string(s: &str) -> String {
    // Reverses a string by reversing its bytes.
    s.bytes().rev().map(|b| b as char).collect()
}
```

Reflection:
Diagnosis: The implementation fails to handle strings with multi-byte UTF-8 characters.

Corrections required:
1. Incorrect string reversal <explanation> The implementation reverses the string by reversing its underlying bytes, which fails for strings containing multi-byte UTF-8 characters like 'é' and 'ü' <solution> Reverse the sequence of Unicode scalar values (chars) rather than bytes by calling `.chars().rev().collect()` on the string slice
''',
'''Previous trial:
function impl:
```rust
fn fibonacci(n: u32) -> u32 {
    // Calculates the Nth Fibonacci number.
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    let mut a = 0;
    let mut b = 1;
    // Loop to calculate Fibonacci sequence
    for _ in 2..n { // Intends to loop up to n-1 times
        let temp = a + b;
        a = b;
        b = temp;
    }
    b // Returns the (N-1)th Fibonacci number for N >= 2
}
```

Reflection:
Diagnosis: The implementation fails to handle cases where the input integer is 2.

Corrections required:
1. Incorrect loop bounds <explanation> The loop `for _ in 2..n` iterates from 2 up to `n-1`. If `n` is 2, the loop doesn't run, and `b` (which is 1, the 1st Fibonacci number) is returned. If `n` is 3, the loop runs once (for `_ = 2`), `b` becomes `0+1=1` (still the 1st Fibonacci), which is incorrect for F(3). To compute the Nth Fibonacci number, the loop should effectively run `n-1` times after establishing F(0) and F(1). A common fix is to loop `for _ in 2..=n` or adjust the initial values/return value. For instance, returning `a` if `n=0` and `b` otherwise after the loop `for _ in 0..n-1 { ... }` (with `a=0, b=1` initially for n>0) or ensuring the loop iterates correctly to compute up to the Nth term. <solution> Loop `for _ in 2..=n` or adjust the initial values/return value. For instance, returning `a` if `n=0` and `b` otherwise after the loop `for _ in 0..n-1 { ... }` (with `a=0, b=1` initially for n>0) or ensuring the loop iterates correctly to compute up to the Nth term.
'''
]



