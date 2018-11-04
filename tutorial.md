# Whispers Tutorial

For this explanation, we'll be using this fairly complicated program as a walk-through/tangible example:

    > Input
    > Input
    >> 1…2
    >> ∤L
    >> L’
    >> Each 4 3
    >> Select∧ 5 L
    >> Each 7 6
    >> Lⁿ10
    > -1
    >> Each 9 8
    >> L‖R
    >> Each 12 11 3
    >> 13ᴺ
    >> Each 9 14
    > 0
    >> 15ⁿ16
    >> Output 17
    
[Try it online!](https://tio.run/##K8/ILC5ILSo2@v/fTsEzr6C0hAtO2ykYPmpYZgRiPOpY4gOifR41zATRronJGQomCsYgdnBqTmpyyaOO5QqmCj5wSXMFM6BJuoYQXY37LeEyhgYKFlCzpgUhRI0UDA0hBhoaP9yyC1m5oQnQKAOwlCnQKEMzENO/tAToSgVD8///DU3NuYxMLAA "Whispers v2 – Try It Online")

## Syntax

All lines in a Whispers program can be split into one of three categories, based off the number of `>` at the start of the line:

- A single `>` indicates a **nilad line** and returns a constant value, as specified on that line. The various syntaxes supported include:
  - `Input` to input a single evaluated line, or `InputAll` to return the entire contents of STDIN
  - Integers, both positive and negative
  - Real numbers, also of either sign
  - Sets or arrays (denoted with `{}` and `[]`, respectively) containing only real numbers, comma (`,`) separated
  - Complex numbers in the form `(a+bj)`
  - Strings opening and closing with one of `"` or `'`
  - Co-ordinates in the form `<name>(<x>, <y>)` or `<name>(<x>, <y>, <z>)` e.g. `P(1, 4)`
  - Vectors in the form `<from>→<to>(<x>, <y>, <z>)` e.g. `A→B(1, 1, 1)`
  - One of the constants represented by:
    - `½`: 0.5
    - `1j`: [*i*](https://en.wikipedia.org/wiki/Imaginary_number)
    - `∅`: The [empty set](https://en.wikipedia.org/wiki/Empty_set)
    - `φ`: The [golden ratio](https://en.wikipedia.org/wiki/Golden_ratio)
    - `π`: [Pi](https://en.wikipedia.org/wiki/Pi)
    - `e`: [e](https://en.wikipedia.org/wiki/E_(mathematical_constant))
    - `𝔹`: The set of boolean values i.e. `{0, 1}`
    - `ℂ`: The set of complex numbers
    - `ℕ`: The set of all positive integers
    - `ℙ`: The set of prime numbers
    - `ℝ`: The set of real numbers
    - `𝕌`: The universal set
    - `ℤ`: The set of integers

- A double `>>` indicates either an **operator line** or a **command line**. On either, numbers do not represent their numerical value. Instead, they represent the value found on that line. For example, in the above program, `>> 1…2` yields the range between the value on line *1* (`> Input`) and the value on line *2* (`> Input`), not a range between **1** and **2**. For this explanation, numbers in *italics* represent the line and numbers in **bold** represent the integer value.

  Operator lines consist of between 2 and 3 things: an operator, and one or more arguments. The arguments are either line references, or `L` or `R`, in certain cases (which wil be expanded on later). An operator can be one of many different options, but it allows us to classify operator lines under one of five sub-categories:
  - **Infix**, the only sub-category which takes two arguments, such as `+` or `≠`. An example from the above program could be `1…2` (the [range](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L478) command), or `L‖R` (the [concatenate](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L482) command).
  - **Postfix** and **Prefix**, where the command goes before or after the argument. This is based entirely off mathematical notation, such as `²` (the [square](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L543) command), or `√` (the [square root](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L521) command)
  - **Surround**, where two complementing characters surround the argument. No examples exist in the above code, but one could be `|x|`, the [absolute value](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L556) function.
  - **Functions**, where, rather than obscure symbols, we simply use the actual function used in normal notation, such as [`sin`](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L611) or [`exp`](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L653)

  Command lines consist of a word (e.g. `Each` or `Select∧`) and a series of line references. Each of the 14 different options have a completely different behaviour, but, again, we can sub-divide them, this time into only **2** groups: dependent and independent. Dependent commands pass a value onto certain lines that they reference, values referred to with `L` and `R` (for left and right, respectively), whereas independent commands don't. As there are only 14 of these commands, the full list (with descriptions) can be listed here:
  - `Output` (independent): Output the result of each argument, newline separated
  - `Error` (independent): Copy `Output`, but to STDERR, then terminate execution.
  - `While` (independent): While the result of the first argument is true, run the second argument. Return the final value before the loop was broken
  - `DoWhile` (independent): Same as `While`, but the second argument is run first before the loop
  - `For` (independent): Run the second argument *n* times, where **n** is the first argument. Return **n**
  - `If` (independent): If the first argument is true, run the second argument. Else, try to run the third argument. If there is no third argument, return **0**
  - `Then` (independent): Run each other arguments in order and return a list of results
  - `Each` (dependent): Iterate a line over an array, with the current element represented by `L`. If two arrays are passed, they are zipped and `R` represents the element in the second array
  - `Select∧` (dependent): Select elements from the final argument (an array) which are truthy when passed through all other arguments
  - `Select∨` (dependent): Select elements from the final argument (an array) which are truthy when passed through any of the other arguments
  - `Select∘` (dependent): Slightly complicated. If we say that we have three functions and an array as arguments to this statement, we select elements by first passing them through the first function, then that result through the second function and that result through the third function. If this final result is truthy, we keep the element, otherwise it is discarded. For a mathematical definition (including the other two `Select` statements), check out [this](https://codegolf.stackexchange.com/a/175186/66833) post.
  - `∑` (dependent): Take a minimum of three arguments, two integers (**x** and **y**) and the rest are lines. Iterate over the range [**x**, ..., **y**], let's call the current element **i**. Pass **i** through each function and take the sum of the results. Finally, add this to a total value, and, once all elements have been iterated over, return that total
  - `…` (dependent): Similar to `∑`, but rather than take the sum of all results, store and return a list of them instead
  - `∏` (dependent): Also similar to `∑`, but takes the product instead of the sum

- A triple `>>>` denotes a **predicate line**. Predicate lines are currently very cumbersome and, to be frank, useless unless checking a single condition (e.g. [primality](https://tio.run/##K8/ILC5ILSo2@v/fzs5O4VHXkkctM7nsFDzzCkpLuIAiho86ZgAlDY0A))

## Structure

A Whispers program only runs one line by default: the final line. All other actions in the program come as branches from running this line, and can be visualised by running [this program][1], with the Whispers program in the **Input** box, and pasting the result into [this website](http://www.webgraphviz.com/). However, usually, a Whispers program is written top-down (although whatever works). With this program below (line numbers added for clarity), execution has the following steps:

     1 > Input
     2 > Input
     3 >> 1…2
     4 >> ∤L
     5 >> L’
     6 >> Each 4 3
     7 >> Select∧ 5 L
     8 >> Each 7 6
     9 >> Lⁿ10
    10 > -1
    11 >> Each 9 8
    12 >> L‖R
    13 >> Each 12 11 3
    14 >> 13ᴺ
    15 >> Each 9 14
    16 > 0
    17 >> 15ⁿ16
    18 >> Output 17

- First, we run the last line: `>> Output 17`

  We can do the first step: output something. However, in order to fully run this line, we need the result of line *17*: `>> 15ⁿ16`
  
  - Here we encounter an **infix operator line**. The operator is `ⁿ` i.e the [index command](https://github.com/cairdcoinheringaahing/Whispers/blob/master/whispers.py#L480) and the two arguments are lines *15* and *16*, so we come to our first "fork" in execution. The Whispers interpreter would run *15*, then *16*, but for this, we'll do it the other way round for simplicity
    - Line *16* is a lot simplier than line *15*: `> 0`
    
      This is a **nilad line** yielding the integer **0**, so the right argument to the index command is **0** i.e. the first element of the array found in line *15*. As line *16* didn't take us to the rest of the program, we know that line *15* will
      
    - Line *15* is an example of an `Each` statement: `>> Each 9 14`
  
      This is a **dependent comand line** that, again, creates another fork, this time to lines *9* and *14*. Due to the structure of `Each` statements, we know that line *9* contains a function and line *14* yields an array.
    
      - We check line *9* as it's most likely to be simpler: `>> Lⁿ10`
    
        As we knew, this is a **infix functional operator line**, which takes one argument from the `Each` statement in line *15* and the other argument from line *10*.
      
        - A quick visit tells us: `> -1` that line *10* is a **nilad line**, yielding **-1**. Now that we have a value, we drop back to line *9*
      
      - We know now one argument and the operator: the index command. With **-1** as a right argument, we know this is a function to retrieve the final element of an array.
    - We now know what line *9* does (retrieve the final element of an array), so we can drop back to line *15*, where we now need the result of line *14*
      - Line *14* is `>> 13ᴺ`, a **postfix operator line**, we takes the array found on line *13* and sorts it
        - Line *13* is a complicated one: `>> Each 12 11 3`. This has three separate parts to it
   
   
   
   
   
[1]: https://tio.run/##bVM9j9swDN31K4gskpCcgbRbgLu1S6YsHRwPQkzHamXZoBS0QS6/PaXkjybATVFIPvLxPXq4xrb33x8P2w09RQjXIKYnoRCnvkZ4T9EixNr6gtDUShdhcDYqefRSC2c9Bi5yNkSF/tIhmYgqQTew1VqIGhuIZHxoeuqUIdI7AdbX@HcDIZL1Z4ZzmIPNFOBxhmL4Y2Or5AfIhGBG8UIe5Op2B3W765UsUkcT1Uuz8tuu0gLQYZd4ERYN541ziuT@8/B5rNdyruU6AZw2jivLit89ZSTzGzukwUxrjsn9QabQGGBQYnO7r@DtA273hVDKbuYd8w/rkEABM9r5rBgrV1o/lmt4g21VbqtUmEY6/6UKE/i/oF@10SPFadqISFmxUGfmLOMra@efyWZZCjMM6OuxtZgtyCkh8ntyvjODWjhtxt3YfEdhVFYkZe0GMCu7nElqkc@hARus54X9CRVmf/K6CGumCuWpdz29E9aVFMtmL27tJt3Y8ICGTi37Xe4P1eh3XmBSg8K8llxJWL8g1ATRM6Y4U38ZlOY6uZp5nAnRV1Jngszv6AFk8au3XiEH01alrZLoQiwnLqWs7ZnM0MIPuKWDNv53bYlT@wP/9elrK0O8OnxvrHO8a/KBYTxbPc/gDTKfo38KllJWHMyKTtm7FGKgfBq@Th/yePSPx8/WhgHZm4F6ZtTBueePuEXCfw
