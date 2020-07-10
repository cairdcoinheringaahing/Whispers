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
                if c == '(':
                        inside += 1
                if c == ')':
                        inside -= 1
                if c == char:
                        return bool(inside)

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

        'sin' : UnaryOp('Prefix', op = 'cos', operand = mathparser.X),
        'cos' : UnaryOp('Prefix', op = '-', operand = UnaryOp('Prefix', op = 'sin', operand = mathparser.X)),
        'tan' : BinaryOp(left = UnaryOp('Prefix', op = 'sec', operand = mathparser.X), op = '*', right = TWO),
        'csc' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = UnaryOp('Prefix', op = 'csc', operand = mathparser.X), op = '⋅', right = UnaryOp('Prefix', op = 'cot', operand = mathparser.X))),
        'sec' : BinaryOp(left = UnaryOp('Prefix', op = 'sec', operand = mathparser.X), op = '⋅', right = UnaryOp('Prefix', op = 'tan', operand = mathparser.X)),
        'cot' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = UnaryOp('Prefix', op = 'csc', operand = mathparser.X), op = '*', right = TWO)),
    
        'sinh' : UnaryOp('Prefix', op = 'cosh', operand = mathparser.X),
        'cosh' : UnaryOp('Prefix', op = 'sinh', operand = mathparser.X),
        'tanh' : BinaryOp(left = UnaryOp('Prefix', op = 'sech', operand = mathparser.X), op = '*', right = TWO),
        'csch' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = UnaryOp('Prefix', op = 'csch', operand = mathparser.X), op = '⋅', right = UnaryOp('Prefix', op = 'coth', operand = mathparser.X))),
        'sech' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = UnaryOp('Prefix', op = 'sech', operand = mathparser.X), op = '⋅', right = UnaryOp('Prefix', op = 'tanh', operand = mathparser.X))),
        'coth' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = UnaryOp('Prefix', op = 'csch', operand = mathparser.X), op = '*', right = TWO)),

        'arcsin' : BinaryOp(left = ONE, op = '÷', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = ONE, op = '-', right = BinaryOp(left = mathparser.X, op = '*', right = TWO)))),
        'arccos' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = ONE, op = '÷', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = ONE, op = '-', right = BinaryOp(left = mathparser.X, op = '*', right = TWO))))),
        'arctan' : BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = ONE, op = '+', right = BinaryOp(left = mathparser.X, op = '*', right = TWO))),
        'arccsc' : BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = mathparser.X, op = '⋅', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = BinaryOp(left = mathparser.X, op = '*', right = TWO), op = '-', right = ONE)))),
        'arcsec' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = mathparser.X, op = '⋅', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = BinaryOp(left = mathparser.X, op = '*', right = TWO), op = '-', right = ONE))))),
        'arccot' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = ONE, op = '+', right = BinaryOp(left = mathparser.X, op = '*', right = TWO)))),

        'arsinh' : BinaryOp(left = ONE, op = '÷', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = ONE, op = '+', right = BinaryOp(left = mathparser.X, op = '*', right = TWO)))),
        'arcosh' : BinaryOp(left = ONE, op = '÷', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = BinaryOp(left = mathparser.X, op = '*', right = TWO), op = '-', right = ONE))),
        'artanh' : BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = ONE, op = '-', right = BinaryOp(left = mathparser.X, op = '*', right = TWO))),
        'arcsch' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = mathparser.X, op = '⋅', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = ONE, op = '+', right = BinaryOp(left = mathparser.X, op = '*', right = TWO)))))),
        'arsech' : BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = mathparser.X, op = '⋅', right = UnaryOp('Prefix', op = '√', operand = BinaryOp(left = ONE, op = '-', right = BinaryOp(left = mathparser.X, op = '*', right = TWO))))),
        'arcoth' : UnaryOp('Prefix', op = '-', operand = BinaryOp(left = ONE, op = '÷', right = BinaryOp(left = BinaryOp(mathparser.X, op = '*', right = TWO), op = '-', right = ONE))),

        'exp' : UnaryOp('Prefix', op = 'exp', operand = mathparser.X),
        'ln'  : BinaryOp(left = ONE, op = '÷', right = mathparser.X),

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
                # print(func)

        if isinstance(func, Constant):
                dfunc = Constant(0)

        if isinstance(func, Variable):
                if func.name and regex.search(r'^[a-z]$', func.name):
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
        return simplify.simplify(derivative(expr))

if __name__ == '__main__':

        tests = {

                '3x³+6x²+x+5':     '9x²+12x+1',
                '6x²+5+x+3x³':     '12x+1+9x²',
                '3x³':             '9x²',
                'exp(x)':          'exp(x)',
                'exp(x)+3x³':      'exp(x)+9x²',
                '-x²':             '-2x',
                '2x':              '2',

                'sin(cos(x))':             '-cos(cos(x))⋅sin(x)',
                '2sin(cos(tan(x)))':       '-2cos(cos(tan(x)))⋅sin(tan(x))⋅sec(x)²',
                'sin(cos(x))+x':           '-cos(cos(x))⋅sin(x)+1',
                '2sin(cos(tan(x)))+x':     '-2cos(cos(tan(x)))⋅sin(tan(x))⋅sec(x)²+1',
                '1+2sin(x)':               '2cos(x)',
                'sin(x)²':                 '2cos(x)⋅sin(x)',
                'sin(x²+2x+1)':            '(2x+2)cos(x²+2x+1)',

                'sin(x)÷x':                '(x⋅cos(x)-sin(x))÷x²',
                'cos(x)÷cosh(x)':          '(-sin(x)⋅cosh(x)-cos(x)⋅sinh(x))÷cosh(x)²',

                '-sin(x)⋅cos(x)':          '-cos(x)²+sin(x)²',
                'sin(x)⋅cos(x)⋅tanh(x)':   'cos(x)²⋅tanh(x)-sin(x)²⋅tanh(x)+cos(x)⋅sin(x)⋅sech(x)²',

                'sin(cos(x))⋅tan(x)':      '-cos(cos(x))⋅sin(x)⋅tan(x)+sin(cos(x))⋅sec(x)²',
                
                'sin(cos(x))÷tan(x)':      '(-cos(cos(x))⋅sin(x)⋅tan(x)-sin(cos(x))⋅sec(x)²)÷(tan(x)²)',

                'sin(x)⋅tan(x)÷x':         '((cos(x)⋅tan(x) + sin(x)⋅sec(x)²)⋅x - sin(x)⋅tan(x))÷x²',

                'sin(cos(x))⋅tan(x)÷cos(x)':       '(sin(cos(x))⋅(sin(x)⋅tan(x)+cos(x)⋅sec(x)²)-cos(x)⋅sin(x)⋅tan(x)⋅cos(cos(x)))÷(cos(x)²)',

                'sin(cos(x)+x⋅tan(x))':            '(cos(cos(x)+x⋅tan(x)))(-sin(x)+tan(x)+x⋅sec(x)²)',

                '2*x':     '2*x⋅ln(2)',
                'x*x':     'x*x⋅(ln(x)+1)',

        }

        for i, o in tests.items():
                print('', mathparser.parse(i), main(i), o, sep = '\n~ ', end = '\n\n---\n\n')
      
















        
