import cmath
import functools
import itertools
import math
import operator
import random
import re
import sys
import unicodedata

import __types as types
from __types import Vector, Coords, InfSet, Radian, Matrix

import mathparser
import derivatives
import integrals
import laplace
import diffeqs

sys.setrecursionlimit(1 << 30)

SETS = {

    chr(0x1d539): (lambda a: a in (0, 1)),                                  # B
    chr(0x2102) : (lambda a: isinstance(a, complex)),                       # C
    chr(0x1d53c): (lambda a: a % 2 == 0),                                   # E
    chr(0x1d541): (lambda a: isinstance(a, float) and not a.is_integer()),  # J
    chr(0x1d544): (lambda a: isinstance(a, Matrix)),                        # M
    chr(0x2115) : (lambda a: isinstance(a, int) and a > 0),                 # N
    chr(0x1d546): (lambda a: a % 2 == 1),                                   # O
    chr(0x2119) : (lambda a: isprime(a)),                                   # P
    chr(0x211d) : (lambda a: isinstance(a, (int, float))),                  # R
    chr(0x1d54a): (lambda a: cmath.sqrt(a).is_integer()),                   # S
    chr(0x1d54c): (lambda a: True),                                         # U
    chr(0x1d54d): (lambda a: isinstance(a, Vector)),                        # V
    chr(0x1d54e): (lambda a: isinstance(a, int) and a >= 0),                # W
    chr(0x2124) : (lambda a: isinstance(a, int)),                           # Z

}

if not hasattr(math, 'gcd'):
    math.gcd = lambda a, b: math.gcd(b, a % b) if b else a

math.sec = lambda a: 1 / math.cos(a)
math.csc = lambda a: 1 / math.sin(a)
math.cot = lambda a: 1 / math.tan(a)
math.sech = lambda a: 1 / math.cosh(a)
math.csch = lambda a: 1 / math.sinh(a)
math.coth = lambda a: 1 / math.tanh(a)

math.asec = lambda a: math.acos(1 / a)
math.acsc = lambda a: math.asin(1 / a)
math.acot = lambda a: math.atan(1 / a)
math.asech = lambda a: math.acosh(1 / a)
math.acsch = lambda a: math.asinh(1 / a)
math.acoth = lambda a: math.atanh(1 / a)

φ = (1 + 5 ** 0.5) / 2
π = math.pi
e = math.e

product = functools.partial(functools.reduce, operator.mul)
findall = lambda string, regex: list(filter(None, regex.match(string).groups()))
square = lambda a: a ** 2

INFIX   = '=≠><≥≤+-±⋅×÷*%∆∩∪⊆⊂⊄⊅⊃⊇∖∈∉«»∤∣⊓⊔∘⊤⊥…⍟ⁱⁿ‖ᶠᵗ∓∕∠≮≯≰≱∧∨⋇⊼⊽∢⊿j≪≫⊈⊉½→∥∦⟂⊾∡√CP?∔δM'
PREFIX  = "∑∏#√?'Γ∤℘ℑℜ∁≺≻∪⍎R…∛\-!∂∫IZ"
POSTFIX = '!’#²³ᵀᴺ°ᴿ₁ᶜ?†'
OPEN    = '|(\[⌈⌊{"‖'
CLOSE   = '|)\]⌉⌋}"‖'
NILADS  = '½∅' + ''.join(SETS.keys())
FUNCS   = [
    
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
    'arcsin', 'arccos', 'arctan', 'arcsec', 'arccsc', 'arccot',
    'cosh', 'sinh', 'tanh', 'sech', 'csch', 'coth',
    'arccosh', 'arcsinh', 'arctanh', 'arcsech', 'arccsch', 'arccoth',
           
    '∂sin', '∂cos', '∂tan', '∂sec', '∂csc', '∂cot',
    '∂arcsin', '∂arccos', '∂arctan', '∂arcsec', '∂arccsc', '∂arccot',
    '∂cosh', '∂sinh', '∂tanh', '∂sech', '∂csch', '∂coth',
    '∂arcosh', '∂arsinh', '∂artanh', '∂arsech', '∂arcsch', '∂arcoth',
    
    '∫sin', '∫cos', '∫tan', '∫sec', '∫csc', '∫cot',
    '∫arcsin', '∫arccos', '∫arctan', '∫arcsec', '∫arccsc', '∫arccot',
    '∫cosh', '∫sinh', '∫tanh', '∫sech', '∫csch', '∫coth',
    '∫arcosh', '∫arsinh', '∫artanh', '∫arsech', '∫arcsch', '∫arcoth',
           
    'cis', 'exp', 'ln', 'sgn', 'arg', 'adj', 'inv',
    'φ', 'π', 'μ', 'λ', 'Ω', 'ω', 'Γ', 'δ', 'ℒ', 'ℱ', 'μ(([1-9][0-9]*)|0)',
    
]

OPERATOR = re.compile(r'''
	^
	(>>\ )
	(?:
		(id)
	|
		([1-9]\d*|[LR])([{}])(λ?[1-9]\d*|[LR])
	|
		([{}])([1-9]\d*|[LR])([{}])
	|
		([{}])([1-9]\d*|[LR])
	|
		([1-9]\d*|[LR])([{}])
	)
	(?:
		\s*
		;
		.*
	)?
	$
	'''.format(INFIX, OPEN, CLOSE, PREFIX, POSTFIX), re.VERBOSE)

STREAM = re.compile(r'''
	^(>>?\ )
	(?:
		(Output\ )
		((?:
			(?: \d+|[LR]|LAST)
		\ )*)
		(\d+|[LR]|LAST)
	|
		(Input
			(?: All)?
		)
	|
		(Error\ ?)
		(\d+|[LR])?
	)$
	''', re.VERBOSE)

NILAD = re.compile(r'''
	^(>\ )
	(
		(
			(
				(")
			|
				(')
			)
			(?(5)
					[^"]
				|
					[^']
			)*
			(?(5)
					"
				|
					'
			)
		)
	|
		(
			-?([1-9]\d*|0)\.\d+
		|
			(-?[1-9]\d*|0)
		|
			\((-?[1-9]\d*|0)[+-]([1-9]\d*|0)j\)
		)
	|
		(
			[\[{]
			(
				(
					
					-?([1-9]\d*|0)
						(\.\d+)?
					,\ ?
				)*
				-?([1-9]\d*|0)
					(\.\d+)?
			)*
			[}\]]
		)
	|
		(
			½
		|
			1j
		|
			∅
		|
			φ
		|
			π
		|
			e
		|
			\[]
		|
			{}
		|
			[%s]
		)
	|
		(
			[A-Z]
			\(
			(
				(
					
					(-?[1-9]\d*|0)
						(\.\d+)?
					,\ ?
				)*
				(-?[1-9]\d*|0)
					(\.\d+)?
			)*
			\)
		)
        |
		(
			[A-Z]→[A-Z]
			\(
			(
				(
					
					(-?[1-9]\d*|0)
						(\.\d+)?
					\ 
				)*
				(-?[1-9]\d*|0)
					(\.\d+)?
			)*
                        \)
                )
	)
	$
	''' % ('\n			'.join(NILADS)), re.VERBOSE)

LOOP = re.compile(r'''
	^
	(>>\ )
	(
		While
            |
		For
            |
		If
            |
		Each
            |
		DoWhile
            |
		Then
            |
		Select[∧∨∘]
            |
                Call
            |
                [
                    ∀
                    ∃
                    ∄
                ]
            |
		[
		    ∑
		    ∏
		    …
		]
	)
	(
		(?: \ \d+|\ [LR])+
	)
	$
	''', re.VERBOSE)

FUNCTION = re.compile(r'''
    ^
    (>>\ )
    (
        {}
    )
    (
        \((?: \d+|[LR])\)
    )
    $
    '''.format('|'.join(FUNCS)), re.VERBOSE)

EXPR = re.compile(r'''
    ^
    (>>>\ )
    (.*)
    $
    ''', re.VERBOSE)

PATTERN = re.compile(r'''
    ^
    (>>>>\ )
    (.*?)
    ,\ ?
    (.*?)
    $
    ''', re.VERBOSE)

REGEXES = {
    'Function': FUNCTION,
    'Operator': OPERATOR,
    'Pattern': PATTERN,
    'Stream': STREAM,
    'Nilad': NILAD,
    'Loop': LOOP,
    'Expr': EXPR,
}

CONST_STDIN = '10' # sys.stdin.read()

INFIX_ATOMS = {

    '=': lambda a, b: a == b,
    '≠': lambda a, b: a != b,
    '>': lambda a, b: a > b,
    '<': lambda a, b: a < b,
    '≥': lambda a, b: a >= b,
    '≤': lambda a, b: a <= b,
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '±': lambda a, b: [a+b, a-b],
    '⋅': lambda a, b: a * b,
    '×': lambda a, b: a.x(b) if isinstance(a, Vector) else set(map(frozenset, itertools.product(a, b))),
    '÷': lambda a, b: a / b,
    '*': lambda a, b: a ** b,
    '%': lambda a, b: a % b,
    '∆': lambda a, b: a ^ b,
    '∩': lambda a, b: a & b,
    '∪': lambda a, b: a | b,
    '⊆': lambda a, b: a.issubset(b),
    '⊂': lambda a, b: a != b and a.issubset(b),
    '⊄': lambda a, b: not a.issubset(b),
    '⊅': lambda a, b: not a.issuperset(b),
    '⊃': lambda a, b: a != b and a.issuperset(b),
    '⊇': lambda a, b: a.issuperset(b),
    '∖': lambda a, b: set(i for i in a if i not in b),
    '∈': lambda a, b: a in b,
    '∉': lambda a, b: a not in b,
    '«': lambda a, b: min(a, b),
    '»': lambda a, b: max(a, b),
    '∣': lambda a, b: not (a % b),
    '∤': lambda a, b: bool(a % b),
    '⊓': lambda a, b: math.gcd(a, b),
    '⊔': lambda a, b: a * b // math.gcd(a, b),
    '⊥': lambda a, b: tobase(a, b),
    '⊤': lambda a, b: frombase(a, b),
    '…': lambda a, b: set(range(a, b+1)),
    '⍟': lambda a, b: math.log(a, b),
    'ⁱ': lambda a, b: list(a).index(b),
    'ⁿ': lambda a, b: a[b % len(a)],
    '‖': lambda a, b: (list(a) if isinstance(a, (list, set)) else [a]) + (list(b) if isinstance(b, (list, set)) else [b]),
    'ᶠ': lambda a, b: a[:b],
    'ᵗ': lambda a, b: a[b:],
    '∓': lambda a, b: [a-b, a+b],
    '∕': lambda a, b: int(a / b),
    '∠': lambda a, b: math.atan2(a, b),
    '≮': lambda a, b: not(a < b),
    '≯': lambda a, b: not(a > b),
    '≰': lambda a, b: not(a <= b),
    '≱': lambda a, b: not(a >= b),
    '∧': lambda a, b: a and b,
    '∨': lambda a, b: a or b,
    '⋇': lambda a, b: [a * b, a // b],
    '⊼': lambda a, b: not(a and b),
    '⊽': lambda a, b: not(a or b),
    '⊿': lambda a, b: math.hypot(a, b),
    'j': lambda a, b: complex(a, b),
    '≪': lambda a, b: a << b,
    '≫': lambda a, b: a >> b,
    '⊈': lambda a, b: a.issubset(b) and a != b,
    '⊉': lambda a, b: a.issuperset(b) and a != b,
    '½': lambda a, b: Coords('M', *list(map(lambda a, b: (a + b) / 2, b.axis, a.axis))),
    '→': lambda a, b: Vector(a.name, b.name, *list(map(operator.sub, b.axis, a.axis))),
    '∥': lambda a, b: a.parallel(b),
    '∦': lambda a, b: not a.parallel(b),
    '⟂': lambda a, b: a.perpendicular(b) if isinstance(a, Vector) else (math.gcd(a, b) == 1),
    '⊾': lambda a, b: not a.perpendicular(b) if isinstance(a, Vector) else (math.gcd(a, b) != 1),
    '∡': lambda a, b: a.angle(b),
    '√': lambda a, b: a ** (1 / b),
    'C': lambda a, b: math.factorial(a) / (math.factorial(a - b) * math.factorial(b)),
    'P': lambda a, b: math.factorial(a) / math.factorial(a - b),
    '?': lambda a, b: random.randint(a, b),
    '∔': lambda a, b: [a*b, a+b],
    'δ': lambda a, b: a == b,
    'M': lambda a, b: Matrix(b, b).reshape(b, b, a),

}

PREFIX_ATOMS = {

    '∑': lambda a: sum(a, type(list(a)[0])()),
    '∏': lambda a: product(a),
    '#': lambda a: len(a),
    '√': lambda a: sqrt(a),
    "'": lambda a: chr(a),
    '?': lambda a: ord(a),
    'Γ': lambda a: math.gamma(a),
    '∤': lambda a: [i for i in range(1, a+1) if a%i == 0],
    '℘': lambda a: set(map(frozenset, powerset(a))),
    'ℑ': lambda a: complex(a).imag,
    'ℜ': lambda a: complex(a).real,
    '∁': lambda a: complex(a).conjugate,
    '≺': lambda a: a - 1,
    '≻': lambda a: a + 1,
    '∪': lambda a: deduplicate(a),
    '⍎': lambda a: eval(a) if type(a) == str else round(a),
    'R': lambda a: (abs(a), a.θ),
    '…': lambda a: tobase(a, 10),
    '∛': lambda a: a ** (1/3),
    '-': lambda a: a.neg() if isinstance(a, Matrix) else -a,
    '!': lambda a: subfactorial(a),
    '∂': lambda a: derivatives.main(a),
    '∫': lambda a: integrals.main(a),
    'I': lambda a: Matrix(a, a),
    'Z': lambda a: Matrix(a, a, True),

}

POSTFIX_ATOMS = {

    '!': lambda a: math.factorial(a),
    '’': lambda a: prime(a),
    '#': lambda a: product([i for i in range(1, a+1) if prime(i)]),
    '²': lambda a: a ** 2,
    '³': lambda a: a ** 3,
    'ᵀ': lambda a: transpose(a),
    'ᴺ': lambda a: sorted(a),
    '°': lambda a: math.degrees(a),
    'ᴿ': lambda a: math.radians(a),
    '₁': lambda a: a.unit,
    'ᶜ': lambda a: Radian(math.radians(a)),
    '?': lambda a: random.choice(a),
    '†': lambda a: conjugate_transpose(a),

}

SURROUND_ATOMS = {

    '||': lambda a: abs(a),
    '⌈⌉': lambda a: math.ceil(a),
    '⌊⌋': lambda a: math.floor(a),
    '⌈⌋': lambda a: int(a),
    '[]': lambda a: set(range(a+1)) if type(a) == int else list(a),
    '[)': lambda a: set(range(a)),
    '(]': lambda a: set(range(1, a+1)),
    '()': lambda a: set(range(1, a)),
    '{}': lambda a: set(a),
    '""': lambda a: str(a),
    '‖‖': lambda a: abs(a),

}

NILAD_ATOMS = {

    '½': 0.5,
    '∅': set(),
    chr(0x1d539): {0, 1},                                                                           # B
    chr(0x2102) : InfSet(chr(0x2102),  lambda a: isinstance(a, complex)),                           # C
    chr(0x1d53c): InfSet(chr(0x1d53c), lambda a: a % 2 == 0),                                       # E
    chr(0x1d541): InfSet(chr(0x1d541), lambda a: isinstance(a, float) and not a.is_integer()),      # J
    chr(0x1d544): InfSet(chr(0x1d544), lambda a: isinstance(a, Matrix)),                            # M
    chr(0x2115) : InfSet(chr(0x2115),  lambda a: isinstance(a, int) and a > 0),                     # N
    chr(0x1d546): InfSet(chr(0x1d546), lambda a: a % 2 == 1),                                       # O
    chr(0x2119) : InfSet(chr(0x2119),  lambda a: isprime(a)),                                       # P
    chr(0x211d) : InfSet(chr(0x211d),  lambda a: isinstance(a, (int, float))),                      # R
    chr(0x1d54a): InfSet(chr(0x1d54a), lambda a: cmath.sqrt(a).is_integer()),                       # S
    chr(0x1d54c): InfSet(chr(0x1d54c), lambda a: True),                                             # U
    chr(0x1d54d): InfSet(chr(0x1d54d), lambda a: isinstance(a, Vector)),                            # V
    chr(0x1d54e): InfSet(chr(0x1d54e), lambda a: isinstance(a, int) and a >= 0),                    # W
    chr(0x2124) : InfSet(chr(0x2124),  lambda a: isinstance(a, int)),                               # Z

}

def gen_ints(mult = 1, start = 1):
    value = start
    while True:
        yield value * mult
        value += 1

SET_GENS = {

    chr(0x1d539): (a for a in (0, 1)),                                      # B
    chr(0x1d53c): (a for a in gen_ints() if a % 2 == 0),                    # E
    chr(0x2115) : (a for a in gen_ints()),                                  # N
    chr(0x1d546): (a for a in gen_ints() if a % 2 == 1),                    # O
    chr(0x2119) : (a for a in gen_ints() if isprime(a)),                    # P
    chr(0x1d54a): (a**2 for a in gen_ints()),                               # S
    chr(0x1d54e): (a for a in gen_ints(start = 0)),                         # W
    chr(0x2124) : (b for a in gen_ints() for b in (a, -a)),                 # Z

}

FUNCTIONS = {

    'sin': lambda a: Radian(math.sin(a)).easy(),
    'cos': lambda a: Radian(math.cos(a)).easy(),
    'tan': lambda a: Radian(math.tan(a)).easy(),
    'sec': lambda a: Radian(math.sec(a)).easy(),
    'csc': lambda a: Radian(math.csc(a)).easy(),
    'cot': lambda a: Radian(math.cot(a)).easy(),
    
    'arcsin': lambda a: Radian(math.asin(a)).easy(),
    'arccos': lambda a: Radian(math.acos(a)).easy(),
    'arctan': lambda a: Radian(math.atan(a)).easy(),
    'arcsec': lambda a: Radian(math.asec(a)).easy(),
    'arccsc': lambda a: Radian(math.acsc(a)).easy(),
    'arccot': lambda a: Radian(math.acot(a)).easy(),

    'sinh': math.sinh,
    'cosh': math.cosh,
    'tanh': math.tanh,
    'sech': math.sech,
    'csch': math.csch,
    'coth': math.coth,

    'arsinh': math.asinh,
    'arcosh': math.acosh,
    'artanh': math.atanh,
    'arsech': math.asech,
    'arcsch': math.acsch,
    'arcoth': math.acoth,

    '∂sin': lambda a: Radian(math.cos(a)).easy(),
    '∂cos': lambda a: Radian(-math.sin(a)).easy(),
    '∂tan': lambda a: Radian(math.sec(a) ** 2).easy(),
    '∂sec': lambda a: Radian(math.tan(a) / math.cos(a)).easy(),
    '∂csc': lambda a: Radian(-math.csc(a) * math.cot(a)).easy(),
    '∂cot': lambda a: Radian(-math.csc(a) ** 2).easy(),

    '∂arcsin': lambda a: 1/(sqrt(1 - a ** 2)),
    '∂arccos': lambda a: -1/(sqrt(1 - a ** 2)),
    '∂arctan': lambda a: 1/(1 + a ** 2),
    '∂arcsec': lambda a: 1/(abs(a) * sqrt(x ** 2 - 1)),
    '∂arccsc': lambda a: -1/(abs(a) * sqrt(x ** 2 - 1)),
    '∂arccot': lambda a: -1/(1 + a ** 2),

    '∂sinh': math.cosh,
    '∂cosh': math.sinh,
    '∂tanh': lambda a: math.sech(a) ** 2,
    '∂sech': lambda a: -math.tanh(a) / math.cosh(a),
    '∂csch': lambda a: -math.csch(a) * math.coth(a),
    '∂coth': lambda a: -math.csch(a) ** 2,

    '∂arsinh': lambda a: 1/sqrt(a ** 2 + 1),
    '∂arcosh': lambda a: 1/sqrt(a ** 2 - 1),
    '∂artanh': lambda a: 1/(1 - a ** 2),
    '∂arsech': lambda a: 1/(a * sqrt(1 - a ** 2)),
    '∂arcsch': lambda a: 1/(abs(a) * sqrt(1 + a ** 2)),
    '∂arcoth': lambda a: 1/(1 - a ** 2),

    '∫sin': lambda a: Radian(-math.cos(a)).easy(),
    '∫cos': lambda a: Radian(math.sin(a)).easy(),
    '∫tan': lambda a: math.ln(Radian(math.sec(a)).easy()),
    '∫sec': lambda a: math.ln(Radian(math.sec(a)).easy() + Radian(math.tan(a)).easy()),
    '∫csc': lambda a: -math.ln(Radian(math.csc(a)).easy() + Radian(math.coth(a)).easy()),
    '∫cot': lambda a: math.ln(Radian(math.sin(a)).easy()),
    
    '∫arcsin': lambda a: a * Radian(math.asin(a)).easy() + sqrt(1 - a ** 2),
    '∫arccos': lambda a: a * Radian(math.acos(a)).easy() - sqrt(1 - a ** 2),
    '∫arctan': lambda a: a * Radian(math.atan(a)).easy() - math.ln(a ** 2 + 1) / 2,
    '∫arcsec': lambda a: a * Radian(math.asec(a)).easy() - math.acosh(a),
    '∫arccsc': lambda a: a * Radian(math.acsc(a)).easy() + math.acosh(a),
    '∫arccot': lambda a: a * Radian(math.acot(a)).easy() + math.ln(a ** 2 + 1) / 2,
    
    '∫sinh': math.cosh,
    '∫cosh': math.sinh,
    '∫tanh': lambda a: math.ln(math.cosh(a)),
    '∫sech': lambda a: Radian(math.atan(math.sinh(a))).easy(),
    '∫csch': lambda a: math.ln(math.tanh(a / 2)),
    '∫coth': lambda a: math.ln(math.sinh(a)),
    
    '∫arsinh': lambda a: a * math.acosh(a) - sqrt(a ** 2 - 1),
    '∫arcosh': lambda a: a * math.asinh(a) - sqrt(a ** 2 + 1),
    '∫artanh': lambda a: a * math.atanh(a) + math.ln(a ** 2 - 1) / 2,
    '∫arsech': lambda a: a * math.asech(a) + math.asin(a),
    '∫arcsch': lambda a: a * math.acsch(a) + math.asinh(a),
    '∫arcoth': lambda a: a * math.acoth(a) + math.ln(a ** 2 - 1) / 2,

    'cis': lambda a: math.cos(a) + 1j * math.sin(a),
    'exp': math.exp,
    'ln': math.log,
    'sgn': lambda a: math.copysign(1, a),
    'arg': lambda a: Radian(cmath.phase(a)).easy(),
    'adj': lambda a: a.adjugate(),
    'inv': lambda a: a.inverse(),
    
    'φ': lambda a: sum(map(lambda k: math.gcd(a, k) == 1, range(1, a+1))),
    'π': lambda a: sum(map(lambda k: prime(k), range(1, a+1))),
    'λ': lambda a: λ(a),
    'Ω': lambda a: Ω(a),
    'ω': lambda a: ω(a),
    'Γ': math.gamma,
    'δ': lambda a: δ(a),
    'Φ': lambda a: Φ(a),
    'ℒ': lambda a: laplace.transform(a),
    'ℱ': lambda a: farey(a),
    
    'μ': lambda a, b = None: ((Ω(a) == ω(a)) * λ(a) if b is None else μ(a, b)),

}

def execute(tokens, index = 0, left = None, right = None, args = None):

    def getvalue(value):
        if value == 'L':
            return left  if left  is not None else 0
        if value == 'R':
            return right if right is not None else 0

        if value == 'LAST':
            return execute(tokens, -1)
        
        return execute(tokens, int(value))
        
    if not tokens:
        return

    line = tokens[(index - 1) % len(tokens)]
    mode = line[0].count('>')

    if mode == 1:
        return tryeval(line[1])

    joined = ''.join(line)

    if OPERATOR.search(joined):
        line = line[1:]

        if line[0] == 'id':
            return left

        if line[0] in PREFIX:
            atom = PREFIX_ATOMS[line[0]]
            target = getvalue(line[1])
            return atom(target)

        if line[1] in POSTFIX:
            atom = POSTFIX_ATOMS[line[1]]
            target = getvalue(line[0])
            return atom(target)

        if line[1] in INFIX:
            larg, atom, rarg = line
            if atom == '∘':
                larg = int(larg)
                rarg = getvalue(rarg)
                return execute(tokens, larg, rarg)
            else:
                larg = getvalue(larg)
                rarg = getvalue(rarg)
                atom = INFIX_ATOMS[atom]
                return atom(larg, rarg)

        if line[0] in OPEN and line[2] in CLOSE:
            try:
                atom = SURROUND_ATOMS[line[0] + line[2]]
            except:
                atom = lambda a: a

            target = getvalue(line[1])
            return atom(target)

    if FUNCTION.search(joined):
        line = line[1:]
        func, target = line
        _, target, _ = target
        func = FUNCTIONS[func]
        target = getvalue(target)
        return func(target)

    if PATTERN.search(joined):
        tkns = re.split(r', ?', line[1])
        formula, *conds = tkns
        formula = mathparser.parse(formula)

        limits = []
        sets = {}
        for cond in conds:
            if re.search(r'^[a-z]∈[{}]$'.format(''.join(SETS.keys())), cond):
                sets[cond[0]] = cond[2]

            elif re.search(r'^(.*?)[=≠><≥≤⊆⊂⊄⊅⊃⊇∣∤≮≯≰≱⊈⊉](.*?)$', cond):
                leftarg, op, rightarg = re.search(r'^(.*?)([=≠><≥≤⊆⊂⊄⊅⊃⊇∣∤≮≯≰≱⊈⊉])(.*?)$', cond).groups()
                limits.append(mathparser.BinaryOp(left = mathparser.parse(leftarg), op = op, right = mathparser.parse(rightarg)))

        return search(left, limits, sets)
        
    if STREAM.search(joined):
        if line[1] == 'Output ':
            targets = ''.join(line[2:]).split()
            for target in targets:
                string = getvalue(target)
                output(string)
            return string

        if line[1].strip() == 'Error':
            output(execute(tokens, int(line[2])), -1)
            sys.exit()

    if LOOP.search(joined):
        loop = line[1]
        targets = line[2].split()

        if loop == 'While':
            cond, call, *_ = targets
            last = 0
            while execute(tokens, int(cond)):
                last = getvalue(call)
            return last
                
        if loop == 'DoWhile':
            cond, call, *_ = targets
            last = getvalue(call)
            while execute(tokens, int(cond)):
                last = getvalue(call)
            return last

        if loop == 'For':
            iters, call, *_ = targets
            for last in range(getvalue(iters)):
                last = getvalue(call)
            return last
            
        if loop == 'If':
            cond, true, *false = targets
            false = false[:1]
            if getvalue(cond):
                return getvalue(true)
            else:
                if false: return getvalue(false[0])
                else: return 0

        if loop == 'Each':
            call, *iters = targets
            result, final = [], []
            for tgt in iters:
                res = getvalue(tgt)
                result.append(res if hasattr(res, '__iter__') else [res])
                
            result = transpose(result)

            for args in result:
                while len(args) != 2:
                    args.append(args[-1])
                    
                final.append(execute(tokens, index = int(call), left = args[0], right = args[1]))

            if all(type(a) == str for a in final):
                return ''.join(final)

            return final

        if loop in ['Select∧', 'Select∨', 'Select∘']:
            mode = '∧∨∘'.find(loop[-1])
            *calls, array = targets
            array = getvalue(array)
            final = []

            for elem in array:
        
                if -1 < mode < 2:
                    ret = []
                    for call in calls:
                        ret.append(execute(tokens, index = int(call), left = elem, right = elem))

                    if mode == 0:
                        func = all
                    if mode == 1:
                        func = any

                    if func(ret):
                        final.append(elem)

                else:
                    ret = execute(tokens, index = int(calls[0]), left = elem, right = elem)
                    for call in calls[1:]:
                        ret = execute(tokens, index = int(call), left = ret, right = ret)
                    if ret:
                        final.append(elem)

            if all(type(a) == str for a in final):
                return ''.join(final)

            return final
                
        if loop in '∑∏…':
           start, end, *f = targets
           start = getvalue(start)
           end = getvalue(end) + 1

           if loop == '∑':
               total = 0
           if loop == '…':
               total = []
           if loop == '∏':
               total = 1

           for n in range(start, end):
               sub = 0
               for fn in f:
                   sub += execute(tokens, int(fn), left = n, right = sub)

               if loop == '∑':
                   total += sub
               if loop == '∏':
                   total *= sub
               if loop == '…':
                   total.append(sub)

           return total

        if loop in '∀∃∄':
            pred, *args = targets

            ret = []
            for arg in args:
                ret.append(execute(tokens, int(pred), left = getvalue(arg)))
            print(ret)
        
        if loop == 'Then':
            ret = []
            for ln in targets:
                ret.append(execute(tokens, int(ln)))
            return ret

        if loop == 'Call':
            line, val, *_ = targets
            expr = setvar(execute(tokens, int(line)), getvalue(val))
            return evaluate(expr)

        if loop == 'Solve':
            eq, *iv = targets

    if EXPR.search(joined):
        return mathparser.parse(line[1])
        
def conjugate_transpose(mat):
    return mat.map_op(lambda a: a.conjugate() if isinstance(a, complex) else a).transpose()

def deduplicate(array):
    final = []
    for value in array:
        if value not in final:
            final.append(value)
    return final

def evaluate(tree):
    if isinstance(tree, mathparser.BinaryOp):
        cmd = INFIX_ATOMS[tree.op]
        return cmd(evaluate(tree.left), evaluate(tree.right))

    if isinstance(tree, mathparser.UnaryOp):
        if tree.pos == 'Postfix':
            if mathparser.isexp(tree.op):
                cmd = lambda a: a ** mathparser.normalise(tree.op)
            else:
                cmd = POSTFIX_ATOMS[tree.op]

        if tree.pos == 'Prefix':
            if tree.op in FUNCTIONS:
                cmd = FUNCTIONS[tree.op]
            else:
                cmd = PREFIX_ATOMS[tree.op]

        return cmd(evaluate(tree.operand))

    if isinstance(tree, (mathparser.Variable, mathparser.Constant)):
        return tree.value

def farey_sequence(n):
    (a, b, c, d) = (0, 1, 1, n)
    yield a/b
    while c <= n:
        k = (n + b) // d
        (a, b, c, d) = (c, d, k * c - a, k * d - b)
        yield a/b

def farey(n):
    return list(farey_sequence(n))

def frombase(digits, base):
    total = 0
    for index, digit in enumerate(digits[: :-1]):
        total += digit * base ** index
    return total

def output(value, file = 1):
    if file < 0:
        file = sys.stderr
    elif file == 0:
        file = sys.stdin
    else:
        file = sys.stdout

    if type(value) in [set, frozenset]:
        print(end = '{', file = file)
        for v in list(value)[: -1]:
            if type(v) == frozenset:
                v = set(v)
            print(v, end = ', ', file = file)
        if value:
            out = list(value)[-1]
        else:
            out = ''
        if type(out) == frozenset:
            out = set(out)
        print(out, '}', sep = '', file = file)

    elif isinstance(value, InfSet):
        print(repr(value), file = file)
        
    else:
        print(value, file = file)

def prime(n):
    if not isinstance(n, int):
        return False
    for i in range(2, int(n)):
        if n%i == 0: return False
    return n > 1 and type(n) == int

def powerset(s):
    x = len(s)
    result = []
    for i in range(1 << x):
        result.append([s[j] for j in range(x) if (i & (1 << j))])
    return result

def search(value, limits, sets):
    pass

def setvar(branch, val):
    new = branch.copy()
    
    if isinstance(branch, mathparser.UnaryOp):
        new.operand = setvar(branch.operand, val)

    if isinstance(branch, mathparser.BinaryOp):
        new.left = setvar(branch.left, val)
        new.right = setvar(branch.right, val)

    if isinstance(branch, mathparser.Variable):
        if branch.name is not None:
            new.value = val

    return new

def sqrt(x):
    if isinstance(x, (int, float)) and x >= 0:
        return math.sqrt(x)
    return cmath.sqrt(x)

def subfactorial(n):
    if n == 1:
        return 0
    if n == 2:
        return 1
    if n == 3:
        return 2
    
    return (n-1) * (subfactorial(n-1) + subfactorial(n-2))

def tobase(value, base):
    digits = []
    while value:
        digits.append(value % base)
        value //= base
    return digits[::-1]

def tokenise(regex, string):
    result = findall(string, regex)
    if result[0] == '> ':
        return result[:2]
    return result

def tokenizer(code, stdin, debug = False):
    for line in stdin.split('\n'):
        if line:
            try: code = code.replace('> Input\n', '> {}\n'.format(tryeval(line)), 1)
            except: code = code.replace('> Input\n', '> "{}"\n'.format(line), 1)
        else:
            code = code.replace('> Input\n', '> 0\n', 1)

    code = code.split('\n')
    final = []

    for line in code:
        for name, regex in REGEXES.items():
            if debug: print(repr(line), '{}: {}'.format(name, regex.search(line)), file = sys.stderr)
            if regex.search(line):
                final.append(tokenise(regex, line))
        if debug: print()
    return final

def transpose(array):
    if isinstance(array, Matrix):
        return array.transpose()
    return list(map(list, zip(*array)))

def tryeval(value, stdin=True):
    try:
        return eval(value)
    except:
        if not stdin:
            return value
        
        if value == 'InputAll':
            return CONST_STDIN
        
        if re.search(r'^[A-Z]\(.*?\)', value):
            axis = list(map(lambda a: int(a[0]), re.findall(r'(-?[1-9]\d*|0)(\.\d+)?', value)))
            name = re.search(r'[A-Z]', value).group()
            return Coords(name, *axis)

        if re.search(r'^[A-Z]→[A-Z]\(.*?\)', value):
            axis = list(map(lambda a: int(a[0]), re.findall(r'(-?[1-9]\d*|0)(\.\d+)?', value)))
            start, end = re.findall(r'[A-Z]', value)
            return Vector(start, end, *axis)

        if re.search(r'^(-?([1-9]\d*|0)(\.\d+)?)[+-](([1-9]\d*|0)(\.\d+)?)i$', value):
            return eval(value.replace('i', 'j'))

    try:
        return NILAD_ATOMS[value]
    except:
        return value

def Φ(n):
    total = 0
    for k in range(1, n+1):
        total += sum(map(lambda a: math.gcd(k, a) == 1, range(1, k+1)))
    return total

def δ(x):
    if x == 0:
        return math.inf
    return 0

def λ(x):
    return (-1) ** Ω(x)

def μ(c, t):
    if t < c:
        return 0
    return 1

def Ω(x):
    total = 0
    p = 2
    while x > 1:
        if x % p == 0:
            total += 1
            x //= p
            p = 2
        else:
            p += 1
            while not prime(p):
                p += 1
    return total

def ω(x):
    if x < 2:
        return 0
    
    p = 2
    total = 0
    while p <= x:
        if x % p == 0:
            total += 1
        p += 1
        while not prime(p) and p <= x:
            p += 1
    return total

if __name__ == '__main__':

    with open('test files/expr', encoding = 'utf-8') as file:
        contents = file.read()

    execute(tokenizer(contents, CONST_STDIN))










