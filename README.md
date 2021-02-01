# Whispers v3

### Hello, World!

    > "Hello, World!"
    >> Output 1

---

To run a Whispers program, you'll need Python 3.7 installed, along with the `mpmath`, `sympy` and `scipy` modules. To install these modules, use

    pip install mpmath sympy scipy
    
or, if this doesn't work, use

    python3 -m pip install mpmath sympy scipy
    
Once these have been installed, create a file that stores your input e.g. `input.txt`. As Whispers reads the entirety of STDIN before execution, you'll need to pipe STDIN in, otherwise it will hang for a prompt forever.

Finally, put your program into a file e.g. `file.whispers`

Then, run the command

    python3 whispers\ v3.py file.whispers < input.txt 2> /dev/null
