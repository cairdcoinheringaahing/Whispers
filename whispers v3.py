# Standard imports

import argparse
import cmath
import enum
import functools
import itertools
import math
import mpmath
import operator
import random
import re
import statistics
import sympy
import sys
import unicodedata

# Scipy imports

import scipy.integrate
import scipy.special

# module imports

import __types as types
from __types import Coords, InfSeq, InfSet, Matrix, Radian, Vector

import mathparser
import derivatives
import integrals
# import diffeqs
import simplify

sys.setrecursionlimit(1 << 30)

# Functions

if not hasattr(math, 'gcd'):
    math.gcd = lambda a, b: math.gcd(b, a % b) if b else a

sin     = lambda a: Radian(math.sin(a)).easy()
cos     = lambda a: Radian(math.cos(a)).easy()
tan     = lambda a: Radian(math.tan(a)).easy()
asin    = lambda a: Radian(math.asin(a)).easy()
acos    = lambda a: Radian(math.acos(a)).easy()
atan    = lambda a: Radian(math.atan(a)).easy()

sec     = lambda a: 1 / cos(a)
csc     = lambda a: 1 / sin(a)
cot     = lambda a: 1 / tan(a)
asec    = lambda a: acos(1 / a)
acsc    = lambda a: asin(1 / a)
acot    = lambda a: atan(1 / a)

sech    = lambda a: 1 / math.cosh(a)
csch    = lambda a: 1 / math.sinh(a)
coth    = lambda a: 1 / math.tanh(a)
asech   = lambda a: math.acosh(1 / a)
acsch   = lambda a: math.asinh(1 / a)
acoth   = lambda a: math.atanh(1 / a)

ln      = math.log

ζ       = scipy.special.zeta
Σ       = mpmath.nsum

Si      = lambda z: scipy.special.sici(z)[0],
Ci      = lambda z: scipy.special.sici(z)[1],
Shi     = lambda z: scipy.special.shichi(z)[0],
Chi     = lambda z: scipy.special.shichi(z)[1],

dawson  = lambda x: math.exp(x**2) * scipy.integrate.quad(lambda t: math.exp(-t ** 2), 0, x)[0],

product = functools.partial(functools.reduce, operator.mul)
findall = lambda string, regex: list(filter(None, regex.match(string).groups()))
square = lambda a: a ** 2

CHARACTERISTIC = 0

# String constants

# ₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹

INFIX   = '=≠><≥≤+-±⋅×÷*%∆∩∪⊆⊂⊄⊅⊃⊇∖∈∉«»∤∣⊓⊔∘⊤⊥…⍟ⁱⁿ‖ᶠᵗ∓∕∠≮≯≰≱∧∨⋇⊼⊽⊿j≪≫⊈⊉½→∥∦⟂⊾∡√CP?∔∸δMζΓγΒ⊞⋕⋚⋛⟌⧺⧻⨥⨧⨪⩱⩲'
INFIXES = ['∂Β']
PREFIX  = "∑∏#√?'Γ∤℘ℑℜ∁≺≻∪⍎R…∛\-!∂∫IZ⊦⨊"
POSTFIX = '!’#²³ᵀᴺ°ᴿ₁ᶜ?⊹'
POSTFIXES = ['!{2,}']
OPEN    = '|(\[⌈⌊{"‖'
CLOSE   = '|)\]⌉⌋}"‖'

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

NILADS  = list(
    '½∅'
    'CGW'
    'bcehij'
    'ΦΨΩ'
    'γμπτφ'
    '⊨⊭'
    '' + ''.join(SETS.keys())
) + \
[
    '1j', '√2', 'ln(2)', '√2ₛ', '1/π', '1/e',
    'ζ(2)', 'ζ(3)', 'β(2)',
    'B⁺[₀-₉]+', 'B⁻[₀-₉]+',
    'C₂', 'C₁₀', 'I₁', 'I₂', 'Nₐ', 'P₂', 'S₃',
    'mₑ',
    'γ₂', 'δₛ', 'ε₀', 'θₘ'
]

SEQS    = [
    
    'Bₙ',   # Bell numbers
    'Cₙ',   # Catalan numbers
    'Eₙ',   # Euler numbers
    'Fₙ',   # Fermat numbers
    'Lₙ',   # Lucas numbers
    'Mₙ',   # Mersenne primes
    'Pₙ',   # Pell numbers
    'Tₙ',   # Tetrahedral numbers
    
    'fₙ',   # Fibonacci numbers
    'sₙ',   # Sylvester's sequence
    'tₙ',   # Triangular numbers
    
    '!ₙ',   # Factorials
    
    'σₙ',   # Divisor function
    'φₙ',   # Totient function
    
    'ℙₙ',   # Primes
    
    'pₙ#',  # Primorials

    'πᵢ',   # Digits of π
    'τᵢ',   # Digits of τ
    'φᵢ',   # Digits of φ
    'eᵢ',   # Digits of e

    '1ⁿ',   # Powers of 1
    '2ⁿ',   # Powers of 2
    '3ⁿ',   # Powers of 3
    '4ⁿ',   # Powers of 4
    '5ⁿ',   # Powers of 5
    '6ⁿ',   # Powers of 6
    '7ⁿ',   # Powers of 7
    '8ⁿ',   # Powers of 8
    '9ⁿ',   # Powers of 9
    '10ⁿ',  # Powers of 10
    '-1ⁿ',  # Powers of -1
    'iⁿ',   # Powers of i

    'n²',   # Square numbers
    'n³',   # Cube numbers
    'nⁿ',   # nⁿ numbers
    '1/n',  # Reciprocal numbers

]

FUNCS   = [
    
    # Trig
    # Inverse trig
    # Hyperbolic
    # Inverse hyperbolic
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
    'arcsin', 'arccos', 'arctan', 'arcsec', 'arccsc', 'arccot',
    'cosh', 'sinh', 'tanh', 'sech', 'csch', 'coth',
    'arcosh', 'arsinh', 'artanh', 'arsech', 'arcsch', 'arcoth',

    # Trig derivatives
    # Inverse trig derivatives
    # Hyperbolic derivatives
    # Inverse hyperbolic derivatives
    '∂sin', '∂cos', '∂tan', '∂sec', '∂csc', '∂cot',
    '∂arcsin', '∂arccos', '∂arctan', '∂arcsec', '∂arccsc', '∂arccot',
    '∂cosh', '∂sinh', '∂tanh', '∂sech', '∂csch', '∂coth',
    '∂arcosh', '∂arsinh', '∂artanh', '∂arsech', '∂arcsch', '∂arcoth',
    
    # Trig integrals
    # Inverse trig integrals
    # Hyperbolic integrals
    # Inverse hyperbolic integrals
    '∫sin', '∫cos', '∫tan', '∫sec', '∫csc', '∫cot',
    '∫arcsin', '∫arccos', '∫arctan', '∫arcsec', '∫arccsc', '∫arccot',
    '∫cosh', '∫sinh', '∫tanh', '∫sech', '∫csch', '∫coth',
    '∫arcosh', '∫arsinh', '∫artanh', '∫arsech', '∫arcsch', '∫arcoth',

    # Expanded Trig functions
    'exsec', 'excosec',
    'versin', 'coversin', 'haversin', 'hacoversin',
    'vercos', 'covercos', 'havercos', 'hacovercos',
    'cis', 'sinc',

    # Inverse expanded trig functions
    'arcexsec', 'arcexcosec',
    'arcversin', 'arccoversin', 'archaversin', 'archacoversin',
    'arcvercos', 'arccovercos', 'archavercos', 'archacovercos',

    # Expanded trig integrals
    '∫exsec', '∫excosec',
    '∫versin', '∫coversin', '∫haversin', '∫hacoversin',
    '∫vercos', '∫covercos', '∫havercos', '∫hacovercos',

    # Inverse expanded trig integrals
    '∫arcexsec', '∫arcexcosec',
    '∫arcversin', '∫arccoversin', '∫archaversin', '∫archacoversin',
    '∫arcvercos', '∫arccovercos', '∫archavercos', '∫archacovercos',

    # Trig integral functons
    # Related Trig integral functions
    'Si', 'si', 'Ci', 'Cin',
    'Shi', 'Chi',

    # Trig integral derivatives
    # Related Trig integral derivatives
    '∂Ci', '∂Cin',
    '∂Shi', '∂Chi',
    
    # Trig integral integrals
    # Related Trig integral integrals
    '∫Si', '∫si', '∫Ci', '∫Cin',
    '∫Shi', '∫Chi',

    # Exponetial and related function
    # Error functions, Dawson functions
    # Log10, log2, ln, log integral, offset log integral, dilog,
    #     iterated log, iterated lg, iterated ln
    # Airy functions
    'exp', 'Ei', 'E₁',
    'erf', 'erfc', 'erfi', 'erfcx', 'D₊', 'D₋',
    'log', 'lg', 'ln', 'li', 'Li', 'Li₂', 'log*', 'lg*', 'ln*', 'Lc',
    'Ai', 'Bi',
    
    # argument, sign, Highly composite, largely composite, arithmetic-geometric mean
    # Matrix functions
    'arg', 'sgn', 'HCN', 'LCN', 'agm', 
    'mat', 'adj', 'inv',

    # Error function derivatives, Dawson derivatives
    # Log function derivatives
    # Airy derivatives
    '∂erf', '∂erfc', '∂erfi', '∂erfcx', '∂D₊', '∂D₋',
    '∂log', '∂lg', '∂ln', '∂li', '∂Li₂', '∂Lc',
    '∂Ai', '∂Bi', 

    # Error function integrals, Dawson integrals
    # Log function integrals
    # Airy integrals
    '∫erf', '∫erfc', '∫erfi',
    '∫log', '∫lg', '∫ln', '∫li', '∫Li₂', '∫Lc',
    '∫Ai', '∫Bi',

    # Fresnel cos integral, Complete elliptic integral (2nd kind),
    #    Barnes G-function, K-function, Fresnel sin integral, Lambert-W function
    # Complete elliptic integral (2st kind), Partition function,
    #    Aliquot sum, Faddeeva function
    # Farey function
    'C', 'E', 'F', 'G', 'K', 'S', 'W',
    'k', 'p', 's', 'w',
    'ℱ',

    # Barnes G-function derivative, K-function derivative
    # Faddeeva function derivative
    '∂G', '∂K',
    '∂w',
    
    # Beta function, Gamma function, Von Mangoldt, Riemann Xi,
    #    Totient summatory function, Prime omega function
    # Dirichlet beta, Dirac delta, Riemann Zeta, Dirichlet eta, Liouville,
    #    Möbius, Landau Xi, prime counter, sigma, Totient function, digamma,
    #    Prime omega
    'Γ', 'Λ', 'Ξ', 'Π', 'Φ', 'Ω',
    'β', 'δ', 'ζ', 'η', 'λ', 'μ', 'ξ', 'π', 'σ', 'ς', 'φ', 'ψ', 'ω',

    # Beta derivative, Gamma derivative, Riemann Xi derivative
    # Dirichlet beta derivative, Riemann zeta derivative,
    #    Dirichlet eta derivative, Landau Xi derivative, Trigamma
    '∂Γ', '∂Ξ', '∂Π',
    '∂β', '∂ζ', '∂η', '∂ξ', '∂ψ',

    # Dirichlet beta integral, Riemann zeta integral,
    #    Dirichlet eta integral, Landau Xi integral, log gamma
    '∫β', '∫ζ', '∫η', '∫ψ',
    
    # Bessel jinc function
    'jinc',
    
    # Derivative of Bessel jinc function
    '∂jinc',

    # Clausen cos function
    # Fermi-Dirac integral
    # Probabilists' Hermite polynomials
    # Physicists' Hermite polynomials
    # Hankel function of the first kind
    # Hankel function of the second kind
    # Modified Bessel function of the first kind
    # Bessel function of the first kind
    # Modified Bessel function of the second kind
    # Laguerre polynomials
    # Polylogarithm
    # Clausen sin function
    # Chebyshev polynomials (first kind)
    # Inverse tangent integral function
    # Chebyshev polynomials (second kind)
    # Bessel function of the second kind
    # Random fibonacci function
    # Spherical Bessel function of the first kind
    # Spherical Bessel function of the second kind
    # Modified spherical Bessel function of the first kind
    # Modified spherical Bessel function of the second kind
    # Super-logarithm
    # Multivariate gamma function
    # Jahnke-Emden lambda function
    # Heaviside
    # Divisor function
    # Legendre chi function
    # Polygamma
    # Multivariate digamma
    # Multivariate polygamma
    'C([₀-₉]+)',
    'F([₀-₉]+)',
    'H([₀-₉]+)',
    'He([₀-₉]+)',
    'H⁽¹⁾([₀-₉]+)',
    'H⁽²⁾([₀-₉]+)',
    'I([₀-₉]+)',
    'J([₀-₉]+)',
    'K([₀-₉]+)',
    'L([₀-₉]+)',
    'Li([₀-₉]+)',
    'S([₀-₉]+)',
    'T([₀-₉]+)',
    'Ti([₀-₉]+)',
    'U([₀-₉]+)',
    'Y([₀-₉]+)',
    'f([₀-₉]+)',
    'i([₀-₉]+)',
    'j([₀-₉]+)',
    'k([₀-₉]+)',
    'y([₀-₉]+)',
    'slog([₀-₉]+)',
    'Γ([₀-₉]+)',
    'Λ([₀-₉]+)',
    'μ([₀-₉]+)',
    'σ([₀-₉]+)',
    'χ([₀-₉]+)',
    'ψ([⁰¹²³⁴⁵⁶⁷⁸⁹]+)',
    'ψ([₀-₉]+)',
    'ψ([₀-₉]+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)',

    # Clausen cos derivative
    # Fermi-Dirac integral derivative
    # Probabilists' Hermite polynomials derivative
    # Physicists' Hermite polynomials derivative
    # Hankel function of the first kind derivative
    # Hankel function of the second kind derivative
    # Modified Bessel function of the first kind derivative
    # Bessel function of the first kind derivative
    # Modified Bessel function of the second kind derivative
    # Laguerre polynomials derivative
    # Polylogarithm derivative
    # Clausen sin derivative
    # Chebyshev polynomials (first kind) derivative
    # Inverse tangent integral derivative
    # Chebyshev polynomials (second kind) derivative
    # Bessel function of the second kind derivative
    # Spherical Bessel function of the first kind derivative
    # Spherical Bessel function of the second kind derivative
    # Modified spherical Bessel function of the first kind derivative
    # Modified spherical Bessel function of the second kind derivative
    # Multivariate gamma derivative
    # Jahnke-Emden lambda derivative
    # Legendre chi derivative
    # Multivariate trigamma
    '∂C([₀-₉]+)',
    '∂F([₀-₉]+)',
    '∂H([₀-₉]+)',
    '∂He([₀-₉]+)',
    '∂H⁽¹⁾([₀-₉]+)',
    '∂H⁽²⁾([₀-₉]+)',
    '∂I([₀-₉]+)',
    '∂J([₀-₉]+)',
    '∂K([₀-₉]+)',
    '∂L([₀-₉]+)',
    '∂Li([₀-₉]+)',
    '∂S([₀-₉]+)',
    '∂T([₀-₉]+)',
    '∂Ti([₀-₉]+)',
    '∂U([₀-₉]+)',
    '∂Y([₀-₉]+)',
    '∂i([₀-₉]+)',
    '∂j([₀-₉]+)',
    '∂k([₀-₉]+)',
    '∂y([₀-₉]+)',
    '∂Γ([₀-₉]+)',
    '∂Λ([₀-₉]+)',
    '∂χ([₀-₉]+)',
    '∂ψ([₀-₉]+)',

    # Clausen cos integral
    # Fermi-Dirac integral integral
    # Probabilists' Hermite polynomials integral
    # Physicists' Hermite polynomials integral
    # Laguerre polynomials integral
    # Polylogarithm integral
    # Clausen sin integral
    # Chebyshev polynomials (first kind) integral
    # Inverse tangent integral integral
    # Chebyshev polynomials (second kind) integral
    # Legendre chi integral
    # Log multivariate gamma
    '∫C([₀-₉]+)',
    '∫F([₀-₉]+)',
    '∫H([₀-₉]+)',
    '∫He([₀-₉]+)',
    '∫L([₀-₉]+)',
    '∫Li([₀-₉]+)',
    '∫S([₀-₉]+)',
    '∫T([₀-₉]+)',
    '∫Ti([₀-₉]+)',
    '∫U([₀-₉]+)',
    '∫χ([₀-₉]+)',
    '∫ψ([₀-₉]+)',
    
    # Arithmetic, geometric and harmonic means
    # Median, Mode
    # Population stddev, Sample stddev, Population variance, Sample variance
    'mean', 'GM', 'HM',
    'med', 'mode',
    'σ', 's', 'σ²', 's²',
    
]

# Regexes

OPERATOR = re.compile(r'''
	^
	(>>\ )
	(?:
		(id)
	|
		([1-9]\d*|[LR])([{}]|{})([1-9]\d*|[LR])
	|
                ([1-9]\d*|[LR])(↑+)([1-9]\d*|[LR])
	|
		([{}])([1-9]\d*|[LR])([{}])
	|
		([{}])([1-9]\d*|[LR])
	|
		([1-9]\d*|[LR])([{}])
	|
                ([1-9]\d*|[LR])(!{})
	)
	(?:
		\s*
		;
		.*
	)?
	$
	'''.format(INFIX, '|'.join(INFIXES), OPEN, CLOSE, PREFIX, POSTFIX, '{2,}'), re.VERBOSE)

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
			[\[{{]
			(
				(
					
					-?([1-9]\d*|0)
						(\.\d+)?
					,\ ?
				)*
				-?([1-9]\d*|0)
					(\.\d+)?
			)*
			[}}\]]
		)
	|
			
		(
			{NILADS}
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
        |
                (
                    [{AEGEAN}]+
                )
	)
	$
	'''.format(NILADS = 
            '|'.join(NILADS) \
            .replace('(', r'\(').replace(')', r'\)'),
                   AEGEAN = ''.join(map(chr, range(0x10107, 0x10133))),
                   
        ),
                  re.VERBOSE)

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
                Mat
            # |
            #     Solve
            |
                ∫
            # |
            #     Match
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
    (?:
        (?:
            (
                {}
            )
            (
                \((?:(?: \d+|[LR]),\ )*(?: \d+|[LR])\)
            )
        )
        |
        (?:
            (
                \((?: \d+|[LR])\)
            )
            ([₀-₉]+)
        )
        |
        (
            (\d+|[LR])
            ⁽
            ([⁰¹²³⁴⁵⁶⁷⁸⁹]+)
            ⁾
        )
        |
        (?:
            (₂F₁)\(
                (\d+|[LR]),\ 
                (\d+|[LR]);\ 
                (\d+|[LR]);\ 
                (\d+|[LR])
            \)
        )
    )
    $'''.format('|'.join(FUNCS)), re.VERBOSE)

EXPR = re.compile(r'''
    ^
    (>>>\ )
    (.*)
    $
    ''', re.VERBOSE)

"""

PATTERN = re.compile(r'''
    ^
    (>>>>\ )
    (.*?)
    ,\ ?
    (.*?)
    $
    ''', re.VERBOSE)

"""

SEQ = re.compile(r'''
    ^
    (>\ )
    ({})
    $
    '''.format('|'.join(SEQS).replace('#', '\#')), re.VERBOSE)

CHAR = re.compile(r'''
    ^
    (>\ )
    (?:
        (ℤ)
        ([₀-₉]+)
    )
    $
    ''', re.VERBOSE)

REGEXES = {
    'Function': FUNCTION,
    'Operator': OPERATOR,
    # 'Pattern': PATTERN,
    'Stream': STREAM,
    'Nilad': NILAD,
    'Loop': LOOP,
    'Expr': EXPR,
    'Seq': SEQ,
    'Char': CHAR,
}

# Gen functions

class InfGen(enum.Enum):

    def π():
        a = 10000
        c = 2800
        b = d = e = g = 0
        f = [a / 5] * (c + 1)
        while True:
            g = c * 2
            b = c
            d += f[b] * a
            g -= 1
            f[b] = d % g
            d //= g
            g -= 1
            b -= 1

            while b:
                d *= b
                d += f[b] * a
                g -= 1
                f[b] = d % g
                d //= g
                g -= 1
                b -= 1

            digits = '%.4d' % (e+d/a)
            for digit in digits:
                yield int(digit)
            e = d%a

    def τ():
        a = 10000
        c = 2800
        b = d = e = g = 0
        f = [a / 5] * (c + 1)
        while True:
            g = c * 2
            b = c
            d += f[b] * a
            g -= 1
            f[b] = d % g
            d //= g
            g -= 1
            b -= 1

            while b:
                d *= b
                d += f[b] * a
                g -= 1
                f[b] = d % g
                d //= g
                g -= 1
                b -= 1

            digits = '%.4d' % (2*(e+d/a))
            if len(digits) == 5:
                digits = digits[1:]
                
            for digit in digits:
                yield int(digit)
            e = d%a
        

    def φ():
        r = 11
        x = 400
        yield 1
        yield 6
        while True:
            d = 0
            while 20*r*d + d*d < x:
                d += 1
            d -= 1
            yield d
            x = 100* (x - (20*r+d) * d)
            r = 10 * r + d

    def e():
        n = 2
        yield 2
        yield 7
        yield 1
        while True:
            ret = -10 * math.floor(math.e * 10 ** (n - 2)) + math.floor(math.e * 10 ** (n + 1))
            yield tobase(ret, 10)[-1]
            n += 1

    def powers(base):
        def inner():
            x = 1
            while True:
                yield base ** x
                x += 1
        return inner

    def triangle():
        x = 0
        index = 1
        while True:
            yield x
            x += index
            index += 1

    def square():
        x = 1
        while True:
            yield x ** 2
            x += 1

    def cube():
        x = 1
        while True:
            yield x ** 3
            x += 1

    def self_power():
        x = 1
        while True:
            yield x ** x
            x += 1

    def reciprocals():
        index = 1
        while True:
            yield 1 / index
            index += 1

    def fibonacci():
        a = b = 1
        while True:
            yield a
            a, b = b, a+b

    def primes():
        num = 2
        while True:
            if prime(num):
                yield num
            num += 1

    def factorials():
        x = 1
        while True:
            yield math.factorial(x)
            x += 1

    def bell():
        def inner(n):
            if n < 2: return 1

            total = 0
            for k in range(n):
                total += math.comb(n-1, k) * inner(k)
            return total

        x = 0
        while True:
            yield inner(x)
            x += 1

    def catalan():
        n = 0
        while True:
            yield math.comb(2*n, n) / (n + 1)
            n += 1

    def euler_zigzag():
        def inner(n):
            if n < 4: return 1

            total = 0
            for k in range(n):
                total += math.comb(n-1, k) * inner(k) * inner(n-k-1)
            return total / 2

        x = 1
        while True:
            yield inner(x)
            x += 1

    def fermat():
        n = 0
        while True:
            yield 2 ** (2 ** n) + 1
            n += 1

    def lucas():
        a, b = 2, 1
        while True:
            yield a
            a, b = b, a+b

    def mersenne():
        p = 1
        while True:
            m = 2 ** p - 1
            if prime(m):
                yield m
            p += 1

    def pell():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, 2*a + b

    def tetrahedral():
        n = 1
        while True:
            gen = InfGen.triangle()
            yield sum(next(gen) for _ in range(n))
            n += 1

    def primorial():
        yield 1
        n = 1
        while True:
            gen = InfGen.primes()
            yield product(next(gen) for _ in range(n))
            n += 1

    def sylvester():
        def inner(n):
            if n == 0: return 2

            total = 1
            for i in range(n):
                total *= inner(i)
            return total + 1

        n = 0
        while True:
            yield inner(n)
            n += 1

    def σ():
        n = 1
        while True:
            yield σ(n)
            n += 1

    def totient():
        n = 1
        while True:
            yield φ(n)
            n += 1

# Math functions

def Γ(z, scripts = (None, None)):
    if hasattr(scripts, '__iter__'):
        sub, _ = scripts

        if sub is None:
            return math.gamma(z)

        p = mathparser.normalise(sub)
        return (π ** (p*(p-1)/4)) * G(z + 1/2) * G(z + 1) / (G(z + (1-p)/2) * G(z + 1 - p/2))

    else:
        s, x = z, scripts
        if s == 0:
            return math.exp(-x)
        s -= 1
        return s*Γ(s, x) + math.exp(-x) * (x ** s)

def dΓ(z, scripts = (None, None)):
    sub, _ = scripts

    if sub is None:
        return Γ(z) * ψ(z)

    p = mathparser.normalise(sub)
    return Γ(z, (p, None)) * ψ(z, (p, None))

def ψ(z, scripts = (None, None)):
    sub, sup = scripts

    if sub is sup is None:
        return scipy.special.psi(z)

    if sub is None:
        sup = mathparser.normalise(sup)
        return scipy.special.polygamma(sup, z)

    if sup is None:
        p = mathparser.normalise(sub)
        total = 0
        for i in range(1, p+1):
            total += ψ(z + (1 - i) / 2)
        return total

    p = mathparser.normalise(sub)
    n = mathparser.normalise(sup)
    total = 0
    for i in range(1, p+1):
        total += scipy.special.polygamma(n, z + (1 - i) / 2)
    return total

def dψ(z, scripts = (None, None)):
    sub, _ = scripts

    if sub is None:
        return scipy.special.polygamma(3, z)

    p = mathparser.normalise(sub)
    total = 0
    for i in range(1, p+1):
        total += scipy.special.polygamma(3, z + (1 - i) / 2)
    return total

def Iψ(z, scripts = (None, None)):
    sub, _ = scripts

    if sub is None:
        return ln(Γ(z))

    return ln(Γ(z, scripts))

## Latin character functions

def C(a, b = None):
    if b is None:
        return scipy.special.fresnel(a * sqrt(2 / π))[1]

    f = lambda k: math.cos(k * a) / k ** b
    return Σ(f, [1, math.inf])

def dC(θ, z):
    f = lambda k: math.sin(k * θ) / k ** (z - 1)
    return -Σ(f, [1, math.inf])

def IC(θ, z):
    f = lambda k: math.sin(k * θ) / k ** (z + 1)
    return Σ(f, [1, math.inf])

def F(x, j, sub = True):
    if j is None:
        return 2 ** (2 ** x) + 1

    if sub:
        return -Li(-math.exp(x), j+1)
    else:
        print(x, j)

def dF(x, j):
    return F(x, j-1)

def IF(x, j):
    return F(x, j+1)

def G(n):
    return Γ(n) ** (n-1) / K(n)

def dG(n):
    left = (Γ(n) ** (n - 1)) / K(n)
    right = (n * ψ(n) - scipy.special.polygamma(-1, n) - n + (1 + ln(τ)) / 2)
    return left * right

def H(x, n):
    if n == 0:
        return 1
    if n == 1:
        return 2*x

    return 2 * x * H(x, n-1) - 2 * (n-1) * H(x, n-2)

def dH(x, n):
    return 2 * n * H(x, n-1)

def IH(x, n):
    return H(x, n+1) / (2*n + 2)

def He(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x

    return x * He(x, n-1) - (n-1) * He(x, n-2)

def dHe(x, n):
    return n * He(x, n-1)

def IHe(x, n):
    return He(x, n+1) / (n+1)

def K(z, n = None):
    if n is None:
        return math.exp(scipy.special.polygamma(-2, z) + (z * (z - 1)) / 2 - z * ln(τ) / 2)
    return scipy.special.kv(n, z)

def dK(z, n = None):
    if n is None:
        return (scipy.special.polygamma(-1, z) + z - (1 + ln(τ)) / 2) * K(z)
    return scipy.special.kvp(n, z)

def L(x, k):
    if k < 0:
        return math.exp(x) * L(-x, 1-k)
    
    if k == 0:
        return 1
    if k == 1:
        return 1-x

    return ((2*k-1-x)*L(x, k-1) - (k-1)*L(x, k-2)) / k

def dL(x, k):
    if k < 0:
        return L(k, x) - math.exp(x) * dL(-x, 1-k)

    if k == 0:
        return 0
    if k == 1:
        return -1

    return ((2*k-1-x)*dL(x, k-1) - L(x, k-1) - (k-1)*dL(x, k-2)) / k

def IL(x, k):
    return L(x, k) - L(x, k+1)

def Lc(z):
    return z*ln(2*sin(z*π)) + S(τ*z, 2) / τ

def dLc(z):
    return ln(2*sin(z*π)) + z*π*cot(z*π) + dS(τ*z, 2)

def ILc(z):
    x = π*1j*z
    
    a = IS(τ*z, 2) / τ ** 2
    b = z ** 2 * (-3*ln(math.exp(2*x)-1) + 3*ln(math.exp(x) - math.exp(-x)) + π*1j) / 6
    c = Li(math.exp(2*x), 3) / τ ** 2
    d = 1j * z * Li(math.exp(τ*1j*z)) / τ

    return a + b - c + d

def li(z):
    return scipy.special.expi(ln(z))

def Li(z, s = None):
    if s is None:
        return li(z) - li(2)
    
    return Σ(lambda k: (z ** k) / (k ** s), [1, math.inf])

def dLi(z, s):
    f = lambda k: (z ** (k-1)) / (k ** (s-1))
    return Σ(f, [1, math.inf])

def ILi(z, s):
    if s == 0:
        return -z - ln(z-1)
    
    return z*Li(z, s-1) - ILi(z, s-1)

def S(a, b = None):
    if b is None:
        return scipy.special.fresnel(a * sqrt(2 / π))[0]

    f = lambda k: math.sin(k * a) / k ** b
    return Σ(f, [1, math.inf])

def dS(θ, z):
    f = lambda k: math.cos(k * θ) / k ** (z - 1)
    return Σ(f, [1, math.inf])

def IS(θ, z):
    f = lambda k: math.cos(k * θ) / k ** (z + 1)
    return -Σ(f, [1, math.inf])

def T(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x

    return 2*x*T(x, n-1) - T(x, n-2)

def dT(x, n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    return 2*T(x, n-1) + 2*x*dT(x, n-1) - dT(x, n-2)

def IT(x, n):
    return (n * T(x, n+1)) / (n ** 2 - 1) - (x * T(x, n)) / (n - 1)

def Ti(z, s):
    return (Li(1j*z, s) - Li(-1j*z, s)) / 2j

def dTi(z, s):
    return (dLi(1j*z, s) + dLi(-1j*z, s)) / 2

def ITi(z, s):
    return (ILi(z*1j) - ILi(-z*1j)) / 2j

def U(x, n):
    if n == 0:
        return 1
    if n == 1:
        return 2*x

    return 2*x*U(x, n-1) - U(x, n-2)

def dU(x, n):
    if n == 0:
        return 0
    if n == 1:
        return 2

    return 2*U(x, n-1) + 2*x*dU(x, n-1) - dU(x, n-2)

def IU(x, n):
    return T(x, n+1) / (n+1)

def k(z, n = None):
    if n is None:
        if isinstance(z, list):
            return scipy.special.ellipkinc(*a)
        return scipy.special.ellipk(a)
    return scipy.special.spherical_kn(n, z)

def p(n):
    def inner():
        total = 0
        for k in range(n):
            total += σ(n - k) * p(k)
        return total
    
    if n == 0: return 1
    if n == 1: return 1
    return inner() // n

def w(z):
    return scipy.special.erfcx(-z * 1j)

def dw(z):
    return (2j * math.exp(-2*z**2)) / sqrt(π) - 2*z*w(z)

## Greek character functions

def Β(x, y):
    return Γ(x) * Γ(y) / Γ(x + y)

def dΒ(x, y):
    return Β(x, y) * (ψ(x) - ψ(x + y))

def Λ(n, v = None):
    if v is None:
        for p in range(n + 1):
            if not prime(p): continue
            for k in range(1, n + 1):
                if n == p ** k:
                    return ln(p)
        return 0
    return scipy.special.lmbda(v, n)

def dΛ(z, v):
    J  = scipy.special.jv
    dJ = scipy.special.jvp
    return (2/z) ** v * Γ(v+1) * (dJ(v, z) - v*z*J(v, z))

def Ξ(z):
    return ξ(1/2 + z * 1j)

def dΞ(z):
    return 1j * dξ(1/2 + z * 1j)

def Π(z):
    return z*Γ(z)

def dΠ(z):
    return Γ(z) + z*dΓ(z)

def Φ(n):
    total = 0
    for k in range(1, n+1):
        total += sum(map(lambda a: math.gcd(k, a) == 1, range(1, k+1)))
    return total

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

def β(s):
    f = lambda n: (-1) ** n / (2*n + 1) ** s
    return Σ(f, [0, math.inf])

def dβ(s):
    f = lambda n: ln(2*n + 1) * (-1) ** (n + 1) / (2*n + 1) ** s
    return Σ(f, [1, math.inf])

def Iβ(s):
    f = lambda n: (-1) ** (n+1) / (ln(2*n+1) * (2*n+1) ** s)
    return Σ(f, [1, math.inf])

def δ(x):
    if x == 0:
        return math.inf
    return 0

def dζ(s):
    f = lambda k: ln(k) / k ** s
    return -Σ(f, [2, math.inf])

def Iζ(s):
    f = lambda k: k ** (-s) / ln(k)
    return -Σ(f, [2, math.inf])

def η(s):
    return (1 - 2 ** (1 - s)) * ζ(s)

def dη(s):
    return (2 ** (1 - s)) * ln(2) * ζ(s) + (1 - (2 ** (1 - s))) * dζ(s)

def Iη(s):
    f = lambda n: (-1) ** n / (n ** s * ln(n))
    return Σ(f, [1, math.inf])

def λ(x):
    return (-1) ** Ω(x)

def μ(c, t):
    if t < c:
        return 0
    return 1

def ξ(s):
    return (s * (s - 1) * (π ** (-s / 2)) * Γ(s / 2) *  ζ(s)) / 2

def dξ(z):
    f  = (z * (z - 1)) / 2
    df = z - 1/2
    g  = π ** (-z / 2)
    dg = -g * ln(sqrt(π))
    h  = Γ(z / 2) * ζ(z)
    dh = Γ(z / 2) * ((ψ(z / 2) * ζ(z)) / 2 + dζ(z))

    return df*g*h + f*dg*h + f*g*dh

def pi(n):
    return sum(map(lambda k: prime(k), range(1, n+1)))

def σ(n, x = 1):
    if hasattr(n, '__iter__'):
        return stddev(n, 'σ')
    total = 0
    for d in range(1, n + 1):
        if n % d: continue
        total += d ** x
    return total

def ς(n):
    total = 0
    for k in range(1, n + 1):
        total += 1 / φ(k)
    return total

def φ(n):
    return sum(map(lambda k: math.gcd(n, k) == 1, range(1, n+1)))

def χ(z, ν):
    return (Li(z, ν) - Li(-z, ν)) / 2

def dχ(z, ν):
    return (dLi(z, ν) + dLi(-z, ν)) / 2

def Iχ(z, ν):
    return (ILi(z, ν) + ILi(-z, ν)) / 2

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

## Multichar named functions

def Ierf(z):
    return z*scipy.special.erf(z) + math.exp(-z ** 2) / sqrt(π)

def farey(n):
    return list(farey_sequence(n))

def fib(n):
    if n == 0: return 0
    if n == 1: return 1

    a, b = 0, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return b

def HCN(z):
    for j in range(1, z):
        if σ(j, 0) >= σ(z, 0):
            return False
    return True

def iterlog(n, base = None):
    if n <= 1:
        return 0
    if base is None: f = ln
    else: f = lambda a: math.log(a, base)
    return 1 + iterlog(f(n), base)

def LCN(z):
    for j in range(1, z):
        if σ(j, 0) > σ(z, 0):
            return False
    return True

def random_fib(x, n):
    if random.choice([0, 1]):
        op = operator.add
    else:
        op = operator.sub

    if n in (1, 2):
        return 1

    return op(random_fib(x, n-1), random_fib(x, n-2))

def slog(z, b):
    if z == 1:
        return 0

    return slog(math.log(z, b)) + 1

def stddev(data, mode = 'σ'):
    if isinstance(data, str):
        data = list(map(ord, data))

    if mode == 'σ²': f = statistics.pvariance
    if mode == 'σ':  f = statistics.pstdev
    if mode == 's²': f = statistics.variance
    if mode == 's':  f = statistics.stdev

    return f(data)

# Exec functions

def assert_(val):
    assert val
        
def conjugate_transpose(mat):
    return mat.map_op(lambda a: a.conjugate() if isinstance(a, complex) else a).transpose()

def characteristic(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if CHARACTERISTIC:
            if isinstance(ret, int):
                return ret % CHARACTERISTIC
        return ret
    return wrapper

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

@characteristic
def execute(tokens, index = 0, left = None, right = None, args = None):
    global CHARACTERISTIC

    def getvalue(value):
        if not value:
            return None
        
        if value == 'L':
            return left  if left  is not None else 0
        if value == 'R':
            return right if right is not None else 0

        if value == 'LAST':
            return execute(tokens, -1)
        if value == 'FIRST':
            return execute(tokens, 0)
        if value == 'NEXT':
            return execute(tokens, index + 1)
        if value == 'PREVIOUS':
            return execute(tokens, index - 1)
        
        return execute(tokens, int(value))
        
    if not tokens:
        return

    #return print(tokens)

    line = tokens[(index - 1) % len(tokens)]
    const = (line[0].count('>') == 1) and (line[1] not in SEQS) and (len(line) == 2)

    if const:
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

        if line[-1] in POSTFIX:
            atom = POSTFIX_ATOMS[line[1]]
            target = getvalue(line[0])
            return atom(target)

        if line[1] in list(INFIX) + INFIXES:
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

        if re.search(r'^↑+$', line[1]):
            larg = getvalue(line[0])
            rarg = getvalue(line[2])
            return hyper(larg, rarg, line[1].count('↑') + 2)

        if re.search(r'^!+$', line[-1]):
            k = line[-1].count('!')
            n = getvalue(line[0])
            return scipy.special.factorialk(n, k)

    if FUNCTION.search(joined) or line[1][0] == 'ψ':
        line = line[1:]
        if re.search(r'[₀₁₂₃₄₅₆₇₈₉]+@[⁰¹²³⁴⁵⁶⁷⁸⁹]+', '@'.join(line[1:3])):
            func, sub, sup, target = line

        elif re.search(r'[₀₁₂₃₄₅₆₇₈₉]+', line[1]):
            func, subp, target = line
            if re.search('[₀₁₂₃₄₅₆₇₈₉]+', subp):
                sub = mathparser.normalise(subp)
                sup = None
            else:
                sup = mathparser.normalise(subp)
                sub = None            
        else:
            func, target = line
            sub = sup = None

        func = func.strip('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹')

        target = target.strip('()')
        f = FUNCTIONS[func]

        if ', ' in target:
            target = list(map(getvalue, map(eval, target.split(', '))))
        else:
            target = getvalue(target)

        if func in ('ψ', '∂ψ', 'Γ', '∂Γ'):
            return f(target, [sub, sup])

        if func in 'F':
            if sub is None:
                return f(*target, sub = False)
        
        if sub is None:
            if sup is None:
                return f(target)

        if sup is None:
            return f(target, sub)
        return f(target, sub, sup)

    '''

    if PATTERN.search(joined):
        conds = re.split(r', ?', line[2])
        formula = mathparser.parse(line[1])

        limits = []
        sets = {}
        for cond in conds:
            if re.search(r'^[a-z]∈[{}]$'.format(''.join(SETS.keys())), cond):
                sets[cond[0]] = cond[2]

            elif re.search(r'^(.*?)[=≠><≥≤⊆⊂⊄⊅⊃⊇∣∤≮≯≰≱⊈⊉](.*?)$', cond):
                leftarg, op, rightarg = re.search(r'^(.*?)([=≠><≥≤⊆⊂⊄⊅⊃⊇∣∤≮≯≰≱⊈⊉])(.*?)$', cond).groups()
                limits.append(mathparser.BinaryOp(left = mathparser.parse(leftarg), op = op, right = mathparser.parse(rightarg)))

        # return (left, limits, sets)

        return search(formula, left, limits, sets)

    '''
        
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

        if loop == 'Match':
            # Incomplete
            pred, val = targets
            val = getvalue(val)
            ret = execute(tokens, int(pred), left = val)
            print('#', ret)
            return ret
        
        if loop == 'Then':
            ret = []
            for ln in targets:
                ret.append(execute(tokens, int(ln)))
            return ret

        if loop == 'Call':
            line, val, *_ = targets
            expr = simplify.setvar(execute(tokens, int(line)), getvalue(val))
            return evaluate(expr)

        if loop == 'Mat':
            rows = [execute(tokens, int(line)) for line in targets]
            return Matrix(rows)

        '''

        if loop == 'Solve':
            line, *inits = targets
            inits = list(map(getvalue, inits))
            if not inits:
                inits = None

            return diffeqs.main(execute(tokens, int(line)), inits)

        '''

        if loop == '∫':
            expr, a, b, *_ = targets
            return integrals.main(expr, [a, b])

    if EXPR.search(joined):
        if left is not None or right is not None:
            expr = mathparser.parse(line[1])
            expr = simplify.setvar(expr, left, 'L')
            expr = simplify.setvar(expr, right, 'R')
            return evaluate(expr)
        
        return mathparser.parse(line[1])

    if SEQ.search(joined):
        seq = line[1]
        return SEQ_ATOMS[seq]

    if CHAR.search(joined):
        CHARACTERISTIC = mathparser.normalise(line[2])

    if line[1] == '₂F₁':
        a, b, c, z = map(getvalue, line[2:])
        return scipy.special.hyp2f1(a, b, c, z)

def farey_sequence(n):
    (a, b, c, d) = (0, 1, 1, n)
    yield a/b
    while c <= n:
        k = (n + b) // d
        (a, b, c, d) = (c, d, k * c - a, k * d - b)
        yield a/b

def frombase(digits, base):
    total = 0
    for index, digit in enumerate(digits[: :-1]):
        total += digit * base ** index
    return total

def gen_ints(mult = 1, start = 1):
    value = start
    while True:
        yield value * mult
        value += 1

def hyper(a, b, n):
    if n == 0:
        return b + 1
    
    if b == 0:
        if n == 1:
            return a
        if n == 2:
            return 0
        if n >= 3:
            return 1
        
    return hyper(a, hyper(a, b - 1, n), n - 1)

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

'''

def search(func, value, limits, sets):
    print('}', func, value, limits, sets, len(sets))

'''

def sqrt(x):
    if isinstance(x, (int, float)) and x >= 0:
        return math.sqrt(x)
    return cmath.sqrt(x)

def subfactorial(n):
    if n == 1:
        return 0
    if n == 2:
        return 1
    
    return (n-1) * (subfactorial(n-1) + subfactorial(n-2))

def tobase(value, base):
    digits = []
    while value:
        digits.append(value % base)
        value //= base
    return digits[::-1]

def tokenise(regex, string):
    result = findall(string, regex)
    if NILAD.search(string):
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
    if re.search(r'B[⁺⁻][₀-₉]+', value):
        n = mathparser.normalise(value[2:])
        if value[1] == '⁺' and n == 1: return 1/2
        if n % 2: return 0
        
        return ((-1) ** (n-1)) * n * ζ(1-n)
            
    if value in NILAD_ATOMS.keys():
        return NILAD_ATOMS[value]
        
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

    return value

# Atoms

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
    '×': lambda a, b: a.x(b) if isinstance(a, Vector) else itertools.product(a, b),
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
    '«': min,
    '»': max,
    '∣': lambda a, b: not (a % b),
    '∤': lambda a, b: bool(a % b),
    '⊓': math.gcd,
    '⊔': lambda a, b: a * b // math.gcd(a, b),
    '⊥': tobase,
    '⊤': frombase,
    '…': lambda a, b: set(range(a, b+1)),
    '⍟': math.log,
    'ⁱ': lambda a, b: list(a).index(b),
    'ⁿ': lambda a, b: a[b % len(a)],
    '‖': lambda a, b: (list(a) if hasattr(a, '__iter__') else [a]) + (list(b) if hasattr(a, '__iter__') else [b]),
    'ᶠ': lambda a, b: a[:b] if hasattr(a, '__iter__') else tobase(a, 10)[:b],
    'ᵗ': lambda a, b: a[b:] if hasattr(a, '__iter__') else tobase(a, 10)[b:],
    '∓': lambda a, b: [a-b, a+b],
    '∕': lambda a, b: int(a / b),
    '∠': math.atan2,
    '≮': lambda a, b: not(a < b),
    '≯': lambda a, b: not(a > b),
    '≰': lambda a, b: not(a <= b),
    '≱': lambda a, b: not(a >= b),
    '∧': lambda a, b: a and b,
    '∨': lambda a, b: a or b,
    '⋇': lambda a, b: [a * b, a // b],
    '⊼': lambda a, b: not(a and b),
    '⊽': lambda a, b: not(a or b),
    '⊿': math.hypot,
    'j': complex,
    '≪': lambda a, b: a << b,
    '≫': lambda a, b: a >> b,
    '⊈': lambda a, b: (not a.issubset(b)) and a != b,
    '⊉': lambda a, b: (not a.issuperset(b)) and a != b,
    '½': lambda a, b: Coords('M', *list(map(lambda a, b: (a + b) / 2, b.axis, a.axis))),
    '→': lambda a, b: Vector(a.name, b.name, *list(map(operator.sub, b.axis, a.axis))),
    '∥': lambda a, b: a.parallel(b),
    '∦': lambda a, b: not a.parallel(b),
    '⟂': lambda a, b: a.perpendicular(b) if isinstance(a, Vector) else (math.gcd(a, b) == 1),
    '⊾': lambda a, b: not a.perpendicular(b) if isinstance(a, Vector) else (math.gcd(a, b) != 1),
    '∡': lambda a, b: a.angle(b),
    '√': lambda a, b: a ** (1 / b),
    'C': scipy.special.comb,
    'P': scipy.special.perm,
    '?': random.randint,
    '∔': lambda a, b: [a*b, a+b],
    '∸': lambda a, b: [a*b, a-b],
    'δ': lambda a, b: a == b,
    'M': lambda a, b: Matrix(b, b).reshape(b, b, [a] * (b ** 2)),
    'ζ': ζ,
    'Γ': Γ,
    'γ': lambda a, b: Γ(a) - Γ(a, b),
    'Β': Β,
    '∂Β': dΒ,
    '⊞': lambda a, b: [[x, y] for x in a for y in b],
    '⋕': lambda a, b: a == b and a.parallel(b),
    '⋚': lambda a, b: a <= b or a > b,
    '⋛': lambda a, b: a >= b or a < b,
    '⟌': lambda a, b: b / a,
    '⧺': lambda a, b: a + b + b,
    '⧻': lambda a, b: a + b + b + b,
    '⨥': lambda a, b: [a+b, a*b],
    '⨧': lambda a, b: (a + b) % 2,
    '⨪': lambda a, b: [a-b, a*b],
    '⩱': lambda a, b: [a==b, a+b],
    '⩲': lambda a, b: [a+b, a==b],

}

PREFIX_ATOMS = {

    '∑': lambda a: sum(a, type(list(a)[0])()),
    '∏': product,
    '#': len,
    '√': sqrt,
    "'": chr,
    '?': ord,
    'Γ': Γ,
    '∤': lambda a: [i for i in range(1, a+1) if a%i == 0],
    '℘': powerset,
    'ℑ': lambda a: complex(a).imag,
    'ℜ': lambda a: complex(a).real,
    '∁': lambda a: complex(a).conjugate,
    '≺': lambda a: a - 1,
    '≻': lambda a: a + 1,
    '∪': deduplicate,
    '⍎': lambda a: eval(a) if type(a) == str else round(a),
    'R': lambda a: (abs(a), a.θ),
    '…': lambda a: tobase(a, 10),
    '∛': lambda a: a ** (1/3),
    '-': lambda a: a.neg() if isinstance(a, Matrix) else -a,
    '!': subfactorial,
    '∂': derivatives.main,
    '∫': integrals.main,
    'I': lambda a: Matrix(a, a),
    'Z': lambda a: Matrix(a, a, True),
    '⊦': assert_,
    '⨊': lambda a: sum(a) % 2,

}

POSTFIX_ATOMS = {

    '!': math.factorial,
    '’': prime,
    '#': lambda a: product([i for i in range(1, a+1) if prime(i)]),
    '²': lambda a: a ** 2,
    '³': lambda a: a ** 3,
    'ᵀ': transpose,
    'ᴺ': sorted,
    '°': math.degrees,
    'ᴿ': math.radians,
    '₁': lambda a: a.unit,
    'ᶜ': lambda a: Radian(math.radians(a)),
    '?': random.choice,
    '⊹': conjugate_transpose,

}

SURROUND_ATOMS = {

    '||': abs,
    '⌈⌉': math.ceil,
    '⌊⌋': math.floor,
    '⌈⌋': int,
    '[]': lambda a: set(range(a+1)) if type(a) == int else list(a),
    '[)': lambda a: set(range(a)),
    '(]': lambda a: set(range(1, a+1)),
    '()': lambda a: set(range(1, a)),
    '{}': set,
    '""': str,
    '‖‖': abs,

}

SEQ_ATOMS = {

    '!ₙ':   InfSeq(InfGen.factorials),
    
    '-1ⁿ':  InfSeq(InfGen.powers(-1), ordered = 0, uniques = [-1, 1]),
    
    '1/n':  InfSeq(InfGen.reciprocals, ordered = -1),
    
    '10ⁿ':  InfSeq(InfGen.powers(10)),
    '1ⁿ':   InfSeq(InfGen.powers(1)),
    '2ⁿ':   InfSeq(InfGen.powers(2)),
    '3ⁿ':   InfSeq(InfGen.powers(3)),
    '4ⁿ':   InfSeq(InfGen.powers(4)),
    '5ⁿ':   InfSeq(InfGen.powers(5)),
    '6ⁿ':   InfSeq(InfGen.powers(6)),
    '7ⁿ':   InfSeq(InfGen.powers(7)),
    '8ⁿ':   InfSeq(InfGen.powers(8)),
    '9ⁿ':   InfSeq(InfGen.powers(9)),
    
    'Bₙ':   InfSeq(InfGen.bell),
    'Cₙ':   InfSeq(InfGen.catalan),
    'Eₙ':   InfSeq(InfGen.euler_zigzag),
    'Fₙ':   InfSeq(InfGen.fermat),
    'Lₙ':   InfSeq(InfGen.lucas),
    'Mₙ':   InfSeq(InfGen.mersenne),
    'Pₙ':   InfSeq(InfGen.pell),
    'Tₙ':   InfSeq(InfGen.tetrahedral),

    'fₙ':   InfSeq(InfGen.fibonacci),
    
    'iⁿ':   InfSeq(InfGen.powers(1j), ordered = 0, uniques = [1j, -1, -1j, 1]),
    
    'n²':   InfSeq(InfGen.square),
    'n³':   InfSeq(InfGen.cube),
    'nⁿ':   InfSeq(InfGen.self_power),
    'pₙ#':  InfSeq(InfGen.primorial),
    'sₙ':   InfSeq(InfGen.sylvester),
    'tₙ':   InfSeq(InfGen.triangle),
    
    'eᵢ':   InfSeq(InfGen.e, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    'πᵢ':   InfSeq(InfGen.π, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    'τᵢ':   InfSeq(InfGen.τ, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    'φᵢ':   InfSeq(InfGen.φ, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    
    'σₙ':   InfSeq(InfGen.σ),
    'φₙ':   InfSeq(InfGen.totient),
    'ℙₙ':   InfSeq(InfGen.primes),

}

# Constants

γ   = mpmath.euler
π   = math.pi
τ   = 2*π

C10 = '0.'
for k in range(1 << 16):
    C10 += str(k + 1)
C10 = eval(C10)

Ω_ = 0
for _ in range(1 << 16):
    Ω_ = (1 + Ω_) / (1 + math.exp(Ω_))

C_f  = lambda n: (-1) ** n / (2*n + 1) ** 2
I1_f = lambda n: (-1) ** (n+1) / (n ** n)
I2_f = lambda n: 1 / (n ** n)
Ψ_f  = lambda n: 1 / fib(n)

NILAD_ATOMS = {

    '1/e':  1 / math.e,
    'C':    Σ(C_f,  [0, math.inf]),
    'C₁₀':  C10,
    'C₂':   0.660161815846869573927812110014,
    'G':    Β(1/4, 1/2) / τ,
    'I₁':   Σ(I1_f, [1, math.inf]),
    'I₂':   Σ(I2_f, [1, math.inf]),
    'Nₐ':   6.02214076 * 10 ** 23,
    'P₂':   math.asinh(1) + sqrt(2),
    'S₃':   (sqrt(13) + 3) / 2,
    'W':    ((45 - sqrt(1929)) / 18) ** (1/3) + ((45 + sqrt(1929)) / 18) ** (1/3),
    'b':    (3 - sqrt(5)) * π,
    'c':    299792458,
    'e':    math.e,
    'h':    6.62607015 * 10 ** -34,
    'i':    1j,
    'j':    1j,
    'mₑ':   9.1093837015 * 10 ** -31,
    '½':    0.5,
    'Φ':    (sqrt(5) - 1) / 2,
    'Ψ':    Σ(Ψ_f,  [1, math.inf]) + 1,
    'Ω':    Ω_,
    'γ₂':   2 / sqrt(3),
    'δₛ':   sqrt(2) + 1,
    'ε₀':   8.8541878128 * 10 ** -12,
    'θₘ':   atan(sqrt(2)),
    'μ':    1.45136923488338105028396848589202744949303228,
    'φ':    (sqrt(5) + 1) / 2,
    '∅':    set(),
    '√2':   sqrt(2),
    '√2ₛ':  1.559610469,

    '⊨': True,
    '⊭': False,
    
    chr(0x1d539): {0, 1},                                                                           # B
    chr(0x2102) : InfSet(chr(0x2102),  lambda a: isinstance(a, complex)),                           # C
    chr(0x1d53c): InfSet(chr(0x1d53c), lambda a: a % 2 == 0, True, k = 0),                          # E
    chr(0x1d541): InfSet(chr(0x1d541), lambda a: isinstance(a, float) and not a.is_integer()),      # J
    chr(0x1d544): InfSet(chr(0x1d544), lambda a: isinstance(a, Matrix)),                            # M
    chr(0x2115) : InfSet(chr(0x2115),  lambda a: isinstance(a, int) and a > 0, True, k = 1),        # N
    chr(0x1d546): InfSet(chr(0x1d546), lambda a: a % 2 == 1, True, k = 1),                          # O
    chr(0x2119) : InfSet(chr(0x2119),  lambda a: isprime(a), True, k = 2),                          # P
    chr(0x211d) : InfSet(chr(0x211d),  lambda a: isinstance(a, (int, float))),                      # R
    chr(0x1d54a): InfSet(chr(0x1d54a), lambda a: cmath.sqrt(a).is_integer()),                       # S
    chr(0x1d54c): InfSet(chr(0x1d54c), lambda a: True),                                             # U
    chr(0x1d54d): InfSet(chr(0x1d54d), lambda a: isinstance(a, Vector)),                            # V
    chr(0x1d54e): InfSet(chr(0x1d54e), lambda a: isinstance(a, int) and a >= 0, True, k = 0),       # W
    chr(0x2124) : InfSet(chr(0x2124),  lambda a: isinstance(a, int)),                               # Z

}

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

    'sin':              sin,
    'cos':              cos,
    'tan':              tan,
    'sec':              sec,
    'csc':              csc,
    'cot':              cot,
    
    'arcsin':           asin,
    'arccos':           acos,
    'arctan':           atan,
    'arcsec':           asec,
    'arccsc':           acsc,
    'arccot':           acot,

    'sinh':             math.sinh,
    'cosh':             math.cosh,
    'tanh':             math.tanh,
    'sech':             sech,
    'csch':             csch,
    'coth':             coth,

    'arsinh':           math.asinh,
    'arcosh':           math.acosh,
    'artanh':           math.atanh,
    'arsech':           asech,
    'arcsch':           acsch,
    'arcoth':           acoth,

    '∂sin':             cos,
    '∂cos':             lambda a: -sin(a),
    '∂tan':             lambda a: sec(a) ** 2,
    '∂sec':             lambda a: tan(a) / cos(a),
    '∂csc':             lambda a: -csc(a) * cot(a),
    '∂cot':             lambda a: -csc(a) ** 2,

    '∂arcsin':          lambda a: 1/(sqrt(1 - a ** 2)),
    '∂arccos':          lambda a: -1/(sqrt(1 - a ** 2)),
    '∂arctan':          lambda a: 1/(1 + a ** 2),
    '∂arcsec':          lambda a: 1/(abs(a) * sqrt(x ** 2 - 1)),
    '∂arccsc':          lambda a: -1/(abs(a) * sqrt(x ** 2 - 1)),
    '∂arccot':          lambda a: -1/(1 + a ** 2),

    '∂sinh':            math.cosh,
    '∂cosh':            math.sinh,
    '∂tanh':            lambda a: math.sech(a) ** 2,
    '∂sech':            lambda a: -math.tanh(a) / math.cosh(a),
    '∂csch':            lambda a: -math.csch(a) * math.coth(a),
    '∂coth':            lambda a: -math.csch(a) ** 2,

    '∂arsinh':          lambda a: 1/sqrt(a ** 2 + 1),
    '∂arcosh':          lambda a: 1/sqrt(a ** 2 - 1),
    '∂artanh':          lambda a: 1/(1 - a ** 2),
    '∂arsech':          lambda a: 1/(a * sqrt(1 - a ** 2)),
    '∂arcsch':          lambda a: 1/(abs(a) * sqrt(1 + a ** 2)),
    '∂arcoth':          lambda a: 1/(1 - a ** 2),

    '∫sin':             lambda a: -cos(a),
    '∫cos':             sin,
    '∫tan':             lambda a: ln(sec(a)),
    '∫sec':             lambda a: ln(sec(a) + tan(a)),
    '∫csc':             lambda a: -ln(csc(a) + coth(a)),
    '∫cot':             lambda a: ln(sin(a)),
    
    '∫arcsin':          lambda a: a * asin(a) + sqrt(1 - a ** 2),
    '∫arccos':          lambda a: a * acos(a) - sqrt(1 - a ** 2),
    '∫arctan':          lambda a: a * atan(a) - ln(a ** 2 + 1) / 2,
    '∫arcsec':          lambda a: a * asec(a) - math.acosh(a),
    '∫arccsc':          lambda a: a * acsc(a) + math.acosh(a),
    '∫arccot':          lambda a: a * acot(a) + ln(a ** 2 + 1) / 2,
    
    '∫sinh':            math.cosh,
    '∫cosh':            math.sinh,
    '∫tanh':            lambda a: ln(math.cosh(a)),
    '∫sech':            lambda a: atan(math.sinh(a)),
    '∫csch':            lambda a: ln(math.tanh(a / 2)),
    '∫coth':            lambda a: ln(math.sinh(a)),
    
    '∫arsinh':          lambda a: a * math.acosh(a) - sqrt(a ** 2 - 1),
    '∫arcosh':          lambda a: a * math.asinh(a) - sqrt(a ** 2 + 1),
    '∫artanh':          lambda a: a * math.atanh(a) + ln(a ** 2 - 1) / 2,
    '∫arsech':          lambda a: a * math.asech(a) + math.asin(a),
    '∫arcsch':          lambda a: a * math.acsch(a) + math.asinh(a),
    '∫arcoth':          lambda a: a * math.acoth(a) + ln(a ** 2 - 1) / 2,

    'exsec':            lambda a: sec(a) - 1,
    'excosec':          lambda a: csc(a) - 1,
    
    'versin':           lambda a: 1 - cos(a),
    'coversin':         lambda a: 1 - sin(a),
    'haversin':         lambda a: (1 - cos(a)) / 2,
    'hacoversin':       lambda a: (1 - sin(a)) / 2,
    
    'vercos':           lambda a: 1 + cos(a),
    'covercos':         lambda a: 1 + sin(a),
    'havercos':         lambda a: (1 + cos(a)) / 2,
    'hacovercos':       lambda a: (1 + sin(a)) / 2,
    
    'cis':              lambda a: cos(a) + 1j * sin(a),
    'sinc':             lambda a: sin(a) / a,

    'arcexsec':         lambda a: asec(a + 1),
    'arcexcosec':       lambda a: acsc(a + 1),
    
    'arcversin':        lambda a: acos(1 - a),
    'arccoversin':      lambda a: asin(1 - a),
    'archaversin':      lambda a: acos(1 - 2*a),
    'archacoversin':    lambda a: asin(1 - 2*a),
    
    'arcvercos':        lambda a: acos(a - 1),
    'arccovercos':      lambda a: asin(a - 1),
    'archavercos':      lambda a: acos(2*a - 1),
    'archacovercos':    lambda a: asin(2*a - 1),
    
    '∫exsec':           lambda a: ln(sec(a) + tan(a)) - a,
    '∫excosec':         lambda a: -ln(csc(a) + coth(a)) - a,
    
    '∫versin':          lambda a: a - sin(a),
    '∫coversin':        lambda a: a + cos(a),
    '∫haversin':        lambda a: (a - sin(a)) / 2,
    '∫hacoversin':      lambda a: (a + cos(a)) / 2,
    
    '∫vercos':          lambda a: a + sin(a),
    '∫covercos':        lambda a: a - cos(a),
    '∫havercos':        lambda a: (a + sin(a)) / 2,
    '∫hacovercos':      lambda a: (a - cos(a)) / 2,

    '∫arcexsec':        lambda a: (a+1) * asec(a+1) - math.acosh(a+1),
    '∫arcexcosec':      lambda a: (a+1) * acsc(a+1) + math.acosh(a+1),
    
    '∫arcversin':       lambda a: sqrt(2*a - a ** 2) + (a-1)*acos(1-a),
    '∫arccoversin':     lambda a: (a-1)*asin(1-a) - sqrt(2*a - a ** 2),
    '∫archaversin':     lambda a: (sqrt(4*a - a ** 2) - (1 - 2*a)*acos(1-2*a)) / 2,
    '∫archacoversin':   lambda a: ((2*a-1)*asin(1-2*a) - sqrt(4*a - a ** 2)) / 2,
    
    '∫arcvercos':       lambda a: (a-1)*acos(a-1) - sqrt(1 - (a-1) ** 2),
    '∫arccovercos':     lambda a: (a-1)*asin(a-1) + sqrt(1 - (a-1) ** 2),
    '∫archavercos':     lambda a: ((2*a-1)*acos(2*a-1) - sqrt(1 - (2*a-1) ** 2)) / 2,
    '∫archacovercos':   lambda a: ((2*a-1)*asin(2*a-1) + sqrt(1 - (2*a-1) ** 2)) / 2,

    'Si':               Si,
    'si':               lambda a: π/2 + Si(a),
    'Ci':               Ci,
    'Cin':              lambda a: γ + ln(a) - Ci(a),
    'Shi':              Shi,
    'Chi':              Chi,

    '∂Ci':              lambda a: cos(a) / a,
    '∂Cin':             lambda a: (1 - cos(a)) / a,
    
    '∂Shi':             lambda a: math.sinh(a) / a,
    '∂Chi':             lambda a: math.cosh(a) / a,
    
    '∫Si':              lambda a: a*Si(a) + cos(a),
    '∫si':              lambda a: a*Si(a) + cos(a) + a*π/2,
    '∫Ci':              lambda a: a*Ci(a) - sin(a),
    '∫Cin':             lambda a: a*γ + a*ln(a) + a - a*Ci(a) + sin(a),
    
    '∫Shi':             lambda a: a*Shi(a) - math.cosh(a),
    '∫Chi':             lambda a: a*Chi(a) - math.sinh(a),
    
    'exp':              math.exp,
    'Ei':               scipy.special.expi,
    'E₁':               scipy.special.exp1,
    
    'erf':              scipy.special.erf,
    'erfc':             scipy.special.erfc,
    'erfi':             scipy.special.erfi,
    'erfcx':            scipy.special.erfcx,
    'D₊':               scipy.special.dawsn,
    'D₋':               dawson,
    
    'log':              lambda a: math.log(a, 10),
    'lg':               lambda a: math.log(a, 2),
    'ln':               ln,
    'li':               li,
    'Li':               Li,
    'Li₂':              lambda a: Li(a, 2),
    'log*':             lambda a: iterlog(a, 10),
    'lg*':              lambda a: iterlog(a, 2),
    'ln*':              iterlog,
    'Lc':               Lc,
    
    'Ai':               lambda a: scipy.special.airy(a)[0],
    'Bi':               lambda a: scipy.special.airy(a)[2],

    'arg':              lambda a: Radian(cmath.phase(a)).easy(),
    'sgn':              lambda a: math.copysign(1, a),
    'HCN':              HCN,
    'LCN':              LCN,
    'agm':              scipy.special.agm,

    'mat':              Matrix,
    'adj':              lambda a: a.adjugate(),
    'inv':              lambda a: a.inverse(),
    
    '∂erf':             lambda a: 2 * math.exp(-a ** 2) / sqrt(π),
    '∂erfc':            lambda a: -2 * math.exp(-a ** 2) / sqrt(π),
    '∂erfi':            lambda a: 2 * math.exp(a ** 2) / sqrt(π),
    '∂erfcx':           lambda a: (4 * a * scipy.special.erfcx(a)) / sqrt(π),
    '∂D₊':              lambda a: 1 - 2 * a * scipy.special.dawsn(a),
    '∂D₋':              lambda a: 1 + 2 * a * dawson(a),
    
    '∂log':             lambda a: 1 / (a * ln(10)),
    '∂lg':              lambda a: 1 / (a * ln(2)),
    '∂ln':              lambda a: 1 / a,
    '∂li':              lambda a: 1 / ln(a),
    '∂Li₂':             lambda a: Σ(lambda k: z ** (k - 1) / k, [1, math.inf]),
    '∂Lc':              dLc,
    
    '∂Ai':              lambda a: scipy.special.airy(a)[1],
    '∂Bi':              lambda a: scipy.special.airy(a)[3],

    '∫erf':             Ierf,
    '∫erfc':            lambda a: a - Ierf(a),
    '∫erfi':            lambda a: scipy.special.erfi(a) - math.exp(a ** 2) / sqrt(π),
    
    '∫log':             lambda a: (a * ln(a) - a) / ln(10),
    '∫lg':              lambda a: (a * ln(a) - a) / ln(2),
    '∫ln':              lambda a: a * ln(a) - a,
    '∫li':              lambda a: a * li(a) - scipy.special.expi(2*ln(a)),
    '∫Li₂':             lambda a: a*Li(a, 2) + (a-1)*(ln(1-a)-1),
    '∫Lc':              ILc,
    
    '∫Ai':              lambda a: scipy.special.itairy(a)[0],
    '∫Bi':              lambda a: scipy.special.itairy(a)[1],

    'C':                C,
    'E':                lambda a: scipy.special.ellipeinc(*a) if isinstance(a, list) else scipy.special.ellipe(a),
    'F':                F,
    'G':                G,
    'H':                H,
    'He':               He,
    'H⁽¹⁾':             lambda a, b: scipy.special.hankel1(b, a),
    'H⁽²⁾':             lambda a, b: scipy.special.hankel2(b, a),
    'I':                lambda a, b: scipy.special.iv(b, a),
    'J':                lambda a, b: scipy.special.jv(b, a),
    'K':                K,
    'L':                L,
    'Li':               Li,
    'S':                S,
    'T':                T,
    'Ti':               Ti,
    'U':                U,
    'W':                scipy.special.lambertw,
    'Y':                lambda a, b: scipy.special.yv(b, a),
    
    'slog':             slog,
    
    'f':                random_fib,
    'i':                lambda a, b: scipy.special.spherical_in(b, a),
    'j':                lambda a, b: scipy.special.spherical_jn(b, a),
    'k':                k,
    'p':                p,
    's':                lambda a: σ(a) - a if not hasattr(a, '__iter__') else stddev(a, 's'),
    's²':               lambda a: stddev(a, 's²'),
    'w':                w,
    'y':                lambda a, b: scipy.special.spherical_yn(b, a),
    
    'ℱ':                farey,

    '∂C':               dC,
    '∂F':               dF,
    '∂G':               dG,
    '∂H':               dH,
    '∂He':              dHe,
    '∂H⁽¹⁾':            lambda a, b: scipy.special.h1vp(b, a),
    '∂H⁽²⁾':            lambda a, b: scipy.special.h2vp(b, a),
    '∂I':               lambda a, b: scipy.special.ivp(b, a),
    '∂J':               lambda a, b: scipy.special.jvp(b, a),
    '∂K':               dK,
    '∂L':               dL,
    '∂Li':              dLi,
    '∂S':               dS,
    '∂T':               dT,
    '∂Ti':              dTi,
    '∂U':               dU,
    '∂Y':               lambda a, b: scipy.special.yvp(b, a),
    
    '∂i':               lambda a, b: scipy.special.spherical_in(b, a, True),
    '∂j':               lambda a, b: scipy.special.spherical_jn(b, a, True),
    '∂k':               lambda a, b: scipy.special.spherical_kn(b, a, True),
    '∂y':               lambda a, b: scipy.special.spherical_yn(b, a, True),

    '∫C':               IC,
    '∫F':               IF,
    '∫H':               IH,
    '∫He':              IHe,
    '∫L':               IL,
    '∫Li':              ILi,
    '∫S':               IS,
    '∫T':               IT,
    '∫Ti':              ITi,
    '∫U':               IU,
    
    '∂w':               dw,
    
    'Γ':                Γ,
    'Λ':                Λ,
    'Ξ':                Ξ,
    'Π':                Π,
    'Φ':                Φ,
    'Ω':                Ω,
    
    'β':                β,
    'δ':                δ,
    'ζ':                ζ,
    'η':                η,
    'λ':                λ,
    'μ':                lambda a, b = None: ((Ω(a) == ω(a)) * λ(a) if b is None else μ(a, b)),
    'ξ':                ξ,
    'π':                pi,
    'σ':                σ,
    'σ²':               lambda a: stddev(a, 'σ²'),
    'ς':                ς,
    'φ':                φ,
    'χ':                χ,
    'ψ':                ψ,
    'ω':                ω,
    
    '∂Γ':               dΓ,
    '∂Λ':               dΛ,
    '∂Ξ':               dΞ,
    '∂Π':               dΠ,
    
    '∂β':               dβ,
    '∂ζ':               dζ,
    '∂η':               dη,
    '∂ξ':               dξ,
    '∂χ':               dχ,
    '∂ψ':               dψ,
    
    '∫β':               Iβ,
    '∫ζ':               Iζ,
    '∫η':               Iη,
    '∫χ':               Iχ,
    '∫ψ':               Iψ,

    'jinc':             lambda a: scipy.special.jv(1, a) / a,
    '∂jinc':            lambda a: - scipy.special.jv(2, a) / a,

    'mean':             statistics.mean,
    'GM':               statistics.geometric_mean,
    'HM':               statistics.harmonic_mean,

    'med':              statistics.median,
    'mode':             statistics.mode,

}

CONST_STDIN = sys.stdin.read()

if __name__ == '__main__':
    try:
        program = sys.argv[1]
    except IndexError:
        sys.exit(1)

    flags = ['--tokens', '-t', '--Tokens', '-T', '--parser', '-p']
    flag = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in flags else False

    argv = sys.argv[3:]
    while len(argv) < 2:
        argv.append(0)

    try:
        program = open(program, 'r', encoding = 'utf-8').read()
    except:
        pass

    if flag in ['--tokens', '-t']:
        print(tokenizer(program, CONST_STDIN), file = sys.stderr)
    if flag in ['--Tokens', '-T']:
        print(*tokenizer(program, CONST_STDIN), sep = '\n', file = sys.stderr)
        
    try:
        execute(tokenizer(program, CONST_STDIN, debug = flag in ['--parser', '-p']), left = argv[0], right = argv[1])
        print('⊤', file = sys.stderr)
    except AssertionError:
        print('⊥', file = sys.stderr)
