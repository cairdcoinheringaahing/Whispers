import mathparser
from mathparser import BinaryOp, UnaryOp, Variable, Constant

MINUSONE = Constant(-1)
ZERO = Constant(0)
ONE = Constant(1)
TWO = Constant(2)

SINGLE_DIGITS = []
for i in range(10):
    SINGLE_DIGITS.append(Constant(i))

SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'

def isnum(value):
    return isinstance(value, Constant) and isinstance(value.value, mathparser.NUMBER)

def simplify_numbers(tree, lorr, opchar):
    cmd = '''
global f
def f(tree):
    if isinstance(tree.{o}, BinaryOp) and tree.{o}.op == '{c}':
        if isnum(tree.{o}.left):
            return BinaryOp(
                left = Constant(tree.{s}.value * tree.{o}.left.value),
                op = '{c}',
                right = simplify(tree.{o}.right)
            )
        
        if isnum(tree.{o}.right):
            return BinaryOp(
                left = Constant(tree.{s}.value * tree.{o}.right.value),
                op = '{c}',
                right = simplify(tree.{o}.left)
            )
        
    if '{s}' == 'left':
        return BinaryOp(left = tree.left, op = '{c}', right = simplify(tree.right))
    else:
        return BinaryOp(left = simplify(tree.left), op = '{c}', right = tree.right)
        
'''.format(c = opchar, o = 'right' if lorr == 'left' else 'left', s = lorr)

    exec(cmd)
    return f(tree)

def simplify(tree, second = False):
    if isinstance(tree, BinaryOp):
        if tree.op == '⋅':
            if ONE in (tree.left, tree.right):
                if tree.left == ONE:
                    return simplify(tree.right)
                if tree.right == ONE:
                    return simplify(tree.left)

            if ZERO in (tree.left, tree.right):
                return ZERO

            if MINUSONE in (tree.left, tree.right):
                if tree.left == MINUSONE:
                    return UnaryOp('Prefix', op = '-', operand = simplify(tree.right))
                if tree.right == MINUSONE:
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
            if ZERO in (tree.left, tree.right):
                if tree.left == ZERO:
                    return simplify(tree.right)
                if tree.right == ZERO:
                    return simplify(tree.left)

        if tree.op in ('÷', '/'):
            if tree.right == ONE:
                return simplify(tree.left)
            if tree.right == MINUSONE:
                return UnaryOp('Prefix', op = '-', operand = simplify(tree.left))
            if tree.left == ZERO:
                return ZERO
            if tree.left == tree.right:
                return ONE

        if tree.op == '%':
            if tree.right in (ONE, MINUSONE):
                return ZERO
            if tree.left == ZERO:
                return ZERO

        if tree.op == '-':
            if tree.right == ZERO:
                return simplify(tree.left)

        if tree.op == '*':
            if tree.left == ZERO:
                return ZERO
            if tree.left == ONE:
                return ONE

            if tree.right == ZERO:
                return ONE
            if tree.right == ONE:
                return simplify(tree.left)

            if tree.right in SINGLE_DIGITS:
                return UnaryOp('Postfix', op = SUPERSCRIPT[tree.right.value], operand = simplify(tree.left))

        simp_tree = BinaryOp(simplify(tree.left), tree.op, simplify(tree.right))
        if not second:
            return simplify(simp_tree, second = True)
        return simp_tree

    if isinstance(tree, UnaryOp):
        if mathparser.isexp(tree.op) and isinstance(tree.operand, Constant):
            if isinstance(tree.operand.value, mathparser.NUMBER):
                return Constant(tree.operand.value ** mathparser.normalise(tree.op))
            return tree

        simp_tree = UnaryOp(tree.pos, tree.op, simplify(tree.operand))
        if not second:
            return simplify(simp_tree, second = True)
        return simp_tree

    return tree


