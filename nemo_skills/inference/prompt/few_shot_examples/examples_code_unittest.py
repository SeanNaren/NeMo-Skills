# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

text_with_code = [
    {
        "question": """\"\"\"
Write a function to find the shared elements from the given two lists.
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
\"\"\"""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)	
  
# generate some test cases
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)	
```""".strip(),
    },
    {
        "question": """\"\"\"
Write a python function to identify non-prime numbers.
assert is_not_prime(2) == False
\"\"\"""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n))):
        if n % i == 0:
            result = True
    return result

# generate some test cases
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
{end_code}
{start_code_output}
Traceback (most recent call last):
  line 12, in <module>
    assert is_not_prime(35) == True
AssertionError
{end_code_output}
One of our test cases failed. I see an error in the logic found in the for loop. Here is the corrected code.
{start_code}
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)+1)):
        if n % i == 0:
            result = True
    return result

# generate some test cases
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
```""".strip(),
    },
    {
        "question": """\"\"\"
Write a function to find the n largest integers from a given list of numbers, returned in descending order.
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
\"\"\"""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
  
# generate some test cases
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
{end_code}
{start_code_output}
NameError: name 'hq' is not defined
{end_code_output}
We are missing an import, here is the corrected solution:
{start_code}
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums

# generate some test cases
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
```""".strip(),
    },
    {
        "question": """\"\"\"
Write a python function to check whether the two numbers differ at one bit position only or not.
assert differ_At_One_Bit_Pos(13,9) == True
\"\"\"""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
def is_Power_Of_Two (x): 
    return x and (not(x & (x - 1))) 
def differ_At_One_Bit_Pos(a,b): 
    return is_Power_Of_Two(a ^ b)

# generate some test cases
assert differ_At_One_Bit_Pos(13,9) == True
assert differ_At_One_Bit_Pos(15,8) == False
assert differ_At_One_Bit_Pos(2,4) == False
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
def is_Power_Of_Two (x): 
    return x and (not(x & (x - 1))) 
def differ_At_One_Bit_Pos(a,b): 
    return is_Power_Of_Two(a ^ b)
```""".strip(),
    },
    {
        "question": """
def find_char_long(text):
    ''' Write a function to find all words which are at least 4 characters long in a string.
    >>> find_char_long('Please move back to stream')
    ['Please', 'move', 'back', 'stream']
    '''
""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
import re
def find_char_long(text):
  return (re.findall(r"\b\w{4,}\b", text))

# generate some test cases
assert find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']
assert find_char_long('Jing Eco and Tech') == ['Jing', 'Tech']
assert find_char_long('Jhingai wulu road Zone 3') == ['Jhingai', 'wulu', 'road', 'Zone']
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
import re
def find_char_long(text):
  return (re.findall(r"\b\w{4,}\b", text))
```""".strip(),
    },
    {
        "question": """
def square_nums(nums):
    ''' Write a function to find squares of individual elements in a list.
    >>> square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    '''
""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
def square_nums(nums):
 square_nums = list(map(lambda x: x ** 2, nums))
 return square_nums

# generate some test cases
assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
assert square_nums([10,20,30])==([100,400,900])
assert square_nums([12,15])==([144,225])
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
def square_nums(nums):
 square_nums = list(map(lambda x: x ** 2, nums))
 return square_nums
```""".strip(),
    },
    {
        "question": """
def find_Rotations(str): 
    ''' Write a python function to find the minimum number of rotations (greater than 0) required to get the same string.
    >>> find_Rotations(\"aaaa\")
    1
    '''
""".strip(),
        "generation": """
Let's solve this problem using Python code.
{start_code}
def find_Rotations(str): 
    tmp = str + str
    n = len(str) 
    for i in range(1,n + 1): 
        substring = tmp[i: i+n] 
        if (str == substring): 
            return i 
    return n

# generate some test cases
assert find_Rotations(\"aaaa\") == 1
assert find_Rotations(\"ab\") == 2
assert find_Rotations(\"abc\") == 3
{end_code}
{start_code_output}
{end_code_output}
All test cases passed, we have the final solution:
```python
def find_Rotations(str): 
    tmp = str + str
    n = len(str) 
    for i in range(1,n + 1): 
        substring = tmp[i: i+n] 
        if (str == substring): 
            return i 
    return n
```""".strip(),
    },
]

examples_map = {
    "mbpp_unittest_with_code": text_with_code,
}
