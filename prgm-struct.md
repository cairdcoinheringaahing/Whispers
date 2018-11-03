## Whispers Program Structure

For this explanation, we'll be using this fairly complicated program as a walk-through/tangible example:

    > Input
    > Input
    >> 1…2
    >> ∤L
    >> L’
    >> Each 4 3
    >> Select∧ 5 L
    >> Each 7 6
    > -1
    >> Lⁿ9
    >> Each 10 8
    >> L‖R
    >> Each 12 11 3
    >> 13ᴺ
    >> Each 10 14
    > 0
    >> 15ⁿ16
    >> Output 17
    
[Try it online!](https://tio.run/##K8/ILC5ILSo2@v/fTsEzr6C0hAtO2ykYPmpYZgRiPOpY4gOifR41zATRronJGQomCsYgdnBqTmpyyaOO5QqmCj5wSXMFM6BJuoYQXY37LeEyhgYKFlCzpgUhRI0UDA0hBhoaP9yyC1m5oQnQKAOwlCnQKEMzENO/tAToSgVD8///DU3NuYxMLAA "Whispers v2 – Try It Online")

All lines in a Whispers program can be split into one of three categories, based off the number of `>` at the start of the line:

- A single `>` indicates a **nilad line**
