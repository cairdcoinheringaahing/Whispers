try:
    import regex
except:
    import re as regex

import sys
import unicodedata

import mathparser
from mathparser import BinaryOp, UnaryOp, Variable, Constant
import simplify

sys.setrecursionlimit(1 << 30)

REGEX = {

    'exponent':     regex.compile(r'^[¹²³⁴⁵⁶⁷⁸⁹][⁰¹²³⁴⁵⁶⁷⁸⁹]*$'),
    'number':       '(?:(?:[1-9]\d*)|0)',

}

ONE = Constant(1)
TWO = Constant(2)

def make_power(num):
    digits = list(map(int, str(num)))
    out = ''
    for dig in digits:
        out += '⁰¹²³⁴⁵⁶⁷⁸⁹'[dig]
    return out

def isvar(var):
    return bool(
        isinstance(var, Variable) and
        var.name is not None and \
        regex.search(r'^[a-z]$', var.name)
    )

def bracketed(string, char):
    inside = 0
    for c in string:
        if c == '(': inside += 1
        if c == ')': inside -= 1
        if c == char: return bool(inside)

def nest(branch, fruit, target = isvar):
    new = branch.copy()
    
    if isinstance(branch, UnaryOp):
        new.operand = nest(branch.operand, fruit)

    if isinstance(branch, BinaryOp):
        new.left = nest(branch.left, fruit)
        new.right = nest(branch.right, fruit)

    if target(branch):
        new = fruit.copy()

    return new
                
## Standard derivatives

standard_derivatives = {

    'sin' : mathparser.parse('cos(x)'),
    'cos' : mathparser.parse('-sin(x)'),
    'tan' : mathparser.parse('sec(x)²'),
    'csc' : mathparser.parse('-cot(x)⋅csc(x)'),
    'sec' : mathparser.parse('tan(x)⋅sec(x)'),
    'cot' : mathparser.parse('-csc(x)²'),

    'sinh' : mathparser.parse('cosh(x)'),
    'cosh' : mathparser.parse('sinh(x)'),
    'tanh' : mathparser.parse('sech(x)²'),
    'csch' : mathparser.parse('-coth(x)⋅csch(x)'),
    'sech' : mathparser.parse('-tanh(x)⋅sech(x)'),
    'coth' : mathparser.parse('-csch(x)²'),

    'arcsin' : mathparser.parse('1÷√(1-x²)'),
    'arccos' : mathparser.parse('-1÷√(1-x²)'),
    'arctan' : mathparser.parse('1÷(1+x²)'),
    'arccsc' : mathparser.parse('-1÷(x⋅√(x²-1))'),
    'arcsec' : mathparser.parse('1÷(x⋅√(x²-1))'),
    'arccot' : mathparser.parse('-1÷(1+x²)'),

    'arsinh' : mathparser.parse('1÷√(x²+1)'),
    'arcosh' : mathparser.parse('1÷√(x²-1)'),
    'artanh' : mathparser.parse('1÷(1-x²)'),
    'arcsch' : mathparser.parse('-1÷(x⋅√(1+x²))'),
    'arsech' : mathparser.parse('-1÷(x⋅√(1-x²))'),
    'arcoth' : mathparser.parse('1÷(1-x²)'),

    'exp' : mathparser.parse('exp(x)'),
    'ln'  : mathparser.parse('1÷x'),

    '√' : mathparser.parse('1÷(2√x)'),
    '∛' : mathparser.parse('1÷(3∛(x²))'),

}

def exponent_rules(f, g):
    # (f * g)' -> (f*g) ⋅ g' ⋅ ln(f) + ((f*g) ⋅ g ⋅ f')÷f
    #          -> (f*g) ⋅ (g' ⋅ ln(f) + (g ⋅ f')÷f)

    df = derivative(f)
    dg = derivative(g)

    ftg = BinaryOp(left = f, op = '*', right = g)
    lnf = UnaryOp('Prefix', op = 'ln', operand = f)

    return BinaryOp(
        left = ftg,
        op = '⋅',
        right = BinaryOp(
            left = BinaryOp(
                left = dg,
                op = '⋅',
                right = lnf
            ),
            op = '+',
            right = BinaryOp(
                left = BinaryOp(
                    left = g,
                    op = '⋅',
                    right = df
                ),
                op = '÷',
                right = f
            )
        )
    )

def chain_rule(fn, g = None):
    # (f∘g)' -> (f'∘g) ⋅ g'

    if g is None:
        f = fn.op
        g = fn.operand

        df = derivative(f)
        dg = derivative(g)

        return BinaryOp(
            left = nest(df, g),
            op = '⋅',
            right = dg
        )

    else:
        print('Chain Rule (f∘g)’:', f, g)

def derivative(func):
    if isinstance(func, str):
        if func in standard_derivatives:
            return standard_derivatives[func]
        func = mathparser.parse(func)

    if isinstance(func, Constant):
        dfunc = Constant(0)

    if isinstance(func, Variable):
        dfunc = Constant(1)

    if isinstance(func, UnaryOp):
        if func.pos == 'Prefix':
            dfunc = chain_rule(func)

        if func.pos == 'Postfix':
            if REGEX['exponent'].search(func.op):
                op = mathparser.normalise(func.op)
                f = func.operand
                
                dfunc = BinaryOp(
                    left = Constant(op),
                    op = '⋅',
                    right = BinaryOp(
                        left = derivative(f),
                        op = '⋅',
                        right = BinaryOp(
                            left = f,
                            op = '*',
                            right = Constant(op - 1)
                        )
                    )
                )
                            

    if isinstance(func, BinaryOp):
        f = func.left
        op = func.op
        g = func.right

        df = derivative(f)
        dg = derivative(g)

        if op in '+-':
            # Addition rule
            # (f ± g)' -> f' ± g'
            
            dfunc = BinaryOp(
                left = df,
                op = op,
                right = dg
            )

        if op == '⋅':
            # Product rule
            # (f ⋅ g)' -> f' ⋅ g + f ⋅ g'

            dfunc = BinaryOp(
                left = BinaryOp(
                    left = df,
                    op = '⋅',
                    right = g
                ),
                op = '+',
                right = BinaryOp(
                    left = f,
                    op = '⋅',
                    right = dg
                )
            )

        if op == '÷':
            # Quotient rule
            # (f ÷ g)' -> (f' ⋅ g - f ⋅ g')÷g²

            dfunc = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = df,
                        op = '⋅',
                        right = g
                    ),
                    op = '-',
                    right = BinaryOp(
                        left = f,
                        op = '⋅',
                        right = dg
                    )
                ),
                op = '÷',
                right = BinaryOp(
                    left = g,
                    op = '*',
                    right = TWO
                )
            )

        if op == '*':
            dfunc = exponent_rules(f, g)

    return dfunc

def main(expr):
    return simplify.main(derivative(expr))












        
