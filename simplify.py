import math
import operator
import re

import mathparser
from mathparser import BinaryOp, UnaryOp, Variable, Constant

MINUSONE = Constant(-1)
ZERO = Constant(0)
ONE = Constant(1)
TWO = Constant(2)

fMINUSONE = Constant(-1.0)
fZERO = Constant(0.0)
fONE = Constant(1.0)
fTWO = Constant(2.0)

SINGLE_DIGITS = []
for i in range(10):
    SINGLE_DIGITS.append(Constant(i))
    SINGLE_DIGITS.append(Constant(float(i)))

SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'

CMDS = {

    '⋅': '*',
    '÷': '/',
    '+': '+',
    '-': '-',
    '*': '**',
    '/': '//',
    '%': '%',

}

def isnum(value):
    return isinstance(value, Constant) and isinstance(value.value, mathparser.NUMBER)

def simplify_numbers(tree, lorr, opchar):
    cmd = '''
global f
def f(tree):
    if isinstance(tree.{o}, BinaryOp) and tree.{o}.op == '{c}':
        if isnum(tree.{o}.left):
            return BinaryOp(
                left = Constant(tree.{s}.value {f} tree.{o}.left.value),
                op = '{c}',
                right = simplify(tree.{o}.right)
            )
        
        if isnum(tree.{o}.right):
            return BinaryOp(
                left = Constant(tree.{s}.value {f} tree.{o}.right.value),
                op = '{c}',
                right = simplify(tree.{o}.left)
            )
        
    if '{s}' == 'left':
        return BinaryOp(left = tree.left, op = '{c}', right = simplify(tree.right))
    else:
        return BinaryOp(left = simplify(tree.left), op = '{c}', right = tree.right)
        
'''.format(
    c = opchar,
    f = CMDS[opchar],
    o = 'right' if lorr == 'left' else 'left',
    s = lorr
)

    exec(cmd)
    return f(tree)

def simplify(tree):
    if isinstance(tree, BinaryOp):
        if isinstance(tree.left, Constant) and isinstance(tree.right, Constant):
            a = eval('{} {} {}'.format(tree.left.value, CMDS[tree.op], tree.right.value))
            if isinstance(a, float) and a.is_integer():
                a = int(a)
            return Constant(a)
        
        if tree.op == '⋅':
            if ONE in (tree.left, tree.right) or fONE in (tree.left, tree.right):
                if tree.left in (ONE, fONE):
                    return simplify(tree.right)
                if tree.right in (ONE, fONE):
                    return simplify(tree.left)

            if ZERO in (tree.left, tree.right) or fZERO in (tree.left, tree.right):
                return ZERO

            if MINUSONE in (tree.left, tree.right) or fMINUSONE in (tree.left, tree.right):
                if tree.left in (MINUSONE, fMINUSONE):
                    return UnaryOp('Prefix', op = '-', operand = simplify(tree.right))
                if tree.right in (MINUSONE, fMINUSONE):
                    return UnaryOp('Prefix', op = '-', operand = simplify(tree.left))

            if isnum(tree.left):
                return simplify_numbers(tree, 'left', '⋅')

            if isnum(tree.right):
                return simplify_numbers(tree, 'right', '⋅')

            if tree.left == tree.right:
                return UnaryOp('Postfix', op = '²', operand = tree.left)

            if isinstance(tree.left, UnaryOp):
                if tree.left.op == '-' and tree.right == tree.left.operand:
                    return UnaryOp('Prefix', op = '-', operand = UnaryOp('Postfix', op = '²', operand = tree.right))

            if isinstance(tree.right, UnaryOp):
                if tree.right.op == '-' and tree.left == tree.right.operand:
                    return UnaryOp('Prefix', op = '-', operand = UnaryOp('Postfix', op = '²', operand = tree.left))
                if mathparser.isexp(tree.right.op) and tree.left == tree.right.operand:
                    newpow = mathparser.make_power(mathparser.normalise(tree.right.op) + 1)
                    return UnaryOp('Postfix', op = newpow, operand = tree.left)

        if tree.op == '+':
            if ZERO in (tree.left, tree.right) or fZERO in (tree.left, tree.right):
                if tree.left in (ZERO, fZERO):
                    return simplify(tree.right)
                if tree.right in (ZERO, fZERO):
                    return simplify(tree.left)

        if tree.op in ('÷', '/'):
            if tree.right in (ONE, fONE):
                return simplify(tree.left)
            if tree.right in (MINUSONE, fMINUSONE):
                return UnaryOp('Prefix', op = '-', operand = simplify(tree.left))
            if tree.left in (ZERO, fZERO):
                return ZERO
            if tree.left == tree.right:
                return ONE

            if isnum(tree.left) and isnum(tree.right):
                val = tree.left.value / tree.right.value
                if tree.op == '/':
                    val = int(val)
                return Constant(val)

        if tree.op == '%':
            if tree.right in (ONE, MINUSONE, fONE, fMINUSONE):
                return ZERO
            if tree.left in (ZERO, fZERO):
                return ZERO

        if tree.op == '-':
            if tree.right in (ZERO, fZERO):
                return simplify(tree.left)

        if tree.op == '*':
            if tree.left in (ZERO, fZERO):
                return ZERO
            if tree.left in (ONE, fONE):
                return ONE

            if tree.right in (ZERO, fZERO):
                return ONE
            if tree.right in (ONE, fONE):
                return simplify(tree.left)

            if tree.right in SINGLE_DIGITS:
                return UnaryOp('Postfix', op = SUPERSCRIPT[int(tree.right.value)], operand = simplify(tree.left))

        simp_tree = BinaryOp(simplify(tree.left), tree.op, simplify(tree.right))
        return simp_tree

    if isinstance(tree, UnaryOp):
        if mathparser.isexp(tree.op) and isinstance(tree.operand, Constant):
            if isinstance(tree.operand.value, mathparser.NUMBER):
                return Constant(tree.operand.value ** mathparser.normalise(tree.op))
            return tree

        simp_tree = UnaryOp(tree.pos, tree.op, simplify(tree.operand))
        return simp_tree

    return tree

def main(expr):
    ret = []
    while True:
        expr = simplify(expr)
        if expr in ret:
            break
        ret.append(expr)

    return expr

COMMANDS = {

    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'csc': lambda z: 1 / math.sin(z),
    'sec': lambda z: 1 / math.cos(z),
    'cot': lambda z: 1 / math.tan(z),
    
    'sinh': math.sinh,
    'cosh': math.cosh,
    'tanh': math.tanh,
    'csch': lambda z: 1 / math.sinh(z),
    'sech': lambda z: 1 / math.cosh(z),
    'coth': lambda z: 1 / math.tanh(z),
    
    'arcsin': math.asin,
    'arccos': math.acos,
    'arctan': math.atan,
    'arccsc': lambda z: math.asin(1 / z),
    'arcsec': lambda z: math.acos(1 / z),
    'arccot': lambda z: math.atan(1 / z),
    
    'arsinh': math.asinh,
    'arcosh': math.acosh,
    'artanh': math.atanh,
    'arcsch': lambda z: math.asinh(1 / z),
    'arsech': lambda z: math.acosh(1 / z),
    'arcoth': lambda z: math.atanh(1 / z),
    
    'exp': math.exp,
    'ln': math.log,

    '√': math.sqrt,
    '∛': lambda z: z ** (1/3),

    '+': operator.add,
    '⋅': operator.mul,
    '÷': operator.truediv,
    '-': operator.sub,
    '*': operator.pow,
    '%': operator.mod,
    '/': operator.floordiv,
    '⍟': math.log,
    '=': operator.eq,

}

def setvar(branch, val, var = None):
    new = branch.copy()
    
    if isinstance(branch, UnaryOp):
        new.operand = setvar(branch.operand, val, var)

    if isinstance(branch, BinaryOp):
        new.left = setvar(branch.left, val, var)
        new.right = setvar(branch.right, val, var)

    if isinstance(branch, Variable):
        if branch.name == var:
            new.value = val

    return new

def evaluate(tree):

    if isinstance(tree, BinaryOp):
        return COMMANDS[tree.op](evaluate(tree.left), evaluate(tree.right))

    if isinstance(tree, UnaryOp):
        if re.search(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', tree.op):
            return evaluate(tree.operand) ** mathparser.normalise(tree.op)
        return COMMANDS[tree.op](evaluate(tree.operand))

    if isinstance(tree, Variable):
        if tree.value:
            return value
        return 0

    if isinstance(tree, Constant):
        return tree.value








