import functools
import itertools
import math
import operator
import re
import sys
import unicodedata

math.gcd = lambda a, b: math.gcd(b, a % b) if b else a

sys.setrecursionlimit(1 << 16)

B = chr(120121)
U = chr(120140)

φ = (1 + 5 ** 0.5) / 2
π = math.pi
e = math.e

product = functools.partial(functools.reduce, operator.mul)
findall = lambda string, regex: list(filter(None, regex.match(string).groups()))
square = lambda a: a ** 2

class Vector:
    def __init__(self, start, *components):
        if isinstance(components[0], str):
            end, *components = components
            self.named = False
        else:
            self.named = True
            end = None
            
        self.start = start
        self.end = end
        self.parts = components

    @property
    def θ(self):
        parts = list(self.parts)
        numer = parts.pop()
        denom = abs(Vector('', *parts))
        return math.degrees(math.atan(numer / denom))

    @property
    def unit(self):
        div = abs(self)
        return Vector('i', *list(map(lambda a: a / div, self.parts)))

    def __repr__(self):
        if self.named:
            return '{}=({})'.format(self.start, ' '.join(map(str, self.parts)))
        return '{}→{}({})'.format(self.start, self.end, ' '.join(map(str, self.parts)))

    def __abs__(self):
        return math.sqrt(sum(map(square, self.parts)))

    def __mul__(self, other):
        return sum(map(operator.mul, self.parts, other.parts))

    def __add__(self, other):
        return Vector('A', 'B', *list(map(operator.add, self.parts, other.parts)))

    def __sub__(self, other):
        return Vector('A', 'B', *list(map(operator.sub, self.parts, other.parts)))

    def __pow__(self, power):
        return abs(self) ** power

    def angle(self, other):
        return math.degrees(math.acos((self * other) / (abs(self) * abs(other))))

    def parallel(self, other):
        return Vector.angle(self, other) == 0

    def perpendicular(self, other):
        return Vector.angle(self, other) == 90

    def x(self, other):
        print('NotYetImplemented')
        return self

class Coords:
    def __init__(self, name, *axis):
        self.name = name
        self.axis = axis

    def __repr__(self):
        return '{}({})'.format(self.name, ', '.join(map(str, self.axis)))

class InfSet:
    def __init__(self, bold, condition):
        self.bold = bold
        self.cond = condition

    def __repr__(self):
        return 'x ∈ {}'.format(self.bold)

    def __contains__(self, other):
        return self.cond(other)

    def __add__(self, other):
        return InfSet('{}∪{}'.format(self.bold, other.bold), lambda a: self.cond(a) or other.cond(a))

    def __mul__(self, other):
        return InfSet('{}∩{}'.format(self.bold, other.bold), lambda a: self.cond(a) and other.cond(a))

    __or__   = __add__
    __ror__  = __add__
    __radd__ = __add__
    __and__  = __mul__
    __rand__ = __mul__
    __rmul__ = __mul__

PRED    = B + 'ℂℕℙℚℝ' + U + 'ℤ¬⊤⊥'
INFIX   = '=≠><≥≤+-±⋅×÷*%∆∩∪⊆⊂⊄⊅⊃⊇∖∈∉«»∤∣⊓⊔∘⊤⊥…⍟ⁱⁿ‖ᶠᵗ∓∕∠≮≯≰≱∧∨⋇⊼⊽∢⊿j≪≫⊈⊉½→∥∦⟂⊾∡√'
PREFIX  = "∑∏#√?'Γ∤℘ℑℜ∁≺≻∪⍎R₁"
POSTFIX = '!’#²³ᵀᴺ°ᴿ₁'
OPEN    = '|(\[⌈⌊{"'
CLOSE   = '|)\]⌉⌋}"'
NILADS  = '½∅' + B + 'ℂℕℙℝ' + U + 'ℤ'

# RIP ℚ

PREDICATE = re.compile(r'''
	^
	(>>>\ )
	([∀∃∄⊤⊥∑∏#≻])
	(
		(?:\d|[{}])+
	)$
	'''.format(PRED + '∘∧∨⊕' + INFIX + PREFIX + POSTFIX), re.VERBOSE)

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
			(?:\d+|[LR]|LAST)
		\ )*)
		(\d+|[LR]|LAST)
	|
		(Input
			(?:All)?
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
			(-?[1-9]\d*|0)\.\d+
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
					
					(-?[1-9]\d*|0)
						(\.\d+)?
					,\ ?
				)*
				(-?[1-9]\d*|0)
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
		[
			∑
			∏
			…
		]
	)
	(
		(?:\ \d+|\ [LR])+
	)
	$
	''', re.VERBOSE)

REGEXES = [PREDICATE, OPERATOR, STREAM, NILAD, LOOP]
CONST_STDIN = sys.stdin.read()

INFIX_ATOMS = {

    '=':lambda a, b: a == b,
    '≠':lambda a, b: a != b,
    '>':lambda a, b: a > b,
    '<':lambda a, b: a < b,
    '≥':lambda a, b: a >= b,
    '≤':lambda a, b: a <= b,
    '+':lambda a, b: a + b,
    '-':lambda a, b: a - b,
    '±':lambda a, b: [a+b, a-b],
    '⋅':lambda a, b: a * b,
    '×':lambda a, b: a.x(b) if isinstance(a, Vector) else set(map(frozenset, itertools.product(a, b))),
    '÷':lambda a, b: a / b,
    '*':lambda a, b: a ** b,
    '%':lambda a, b: a % b,
    '∆':lambda a, b: a ^ b,
    '∩':lambda a, b: a & b,
    '∪':lambda a, b: a | b,
    '⊆':lambda a, b: a.issubset(b),
    '⊂':lambda a, b: a != b and a.issubset(b),
    '⊄':lambda a, b: not a.issubset(b),
    '⊅':lambda a, b: not a.issuperset(b),
    '⊃':lambda a, b: a != b and a.issuperset(b),
    '⊇':lambda a, b: a.issuperset(b),
    '∖':lambda a, b: set(i for i in a if i not in b),
    '∈':lambda a, b: a in b,
    '∉':lambda a, b: a not in b,
    '«':lambda a, b: min(a, b),
    '»':lambda a, b: max(a, b),
    '∣':lambda a, b: not (a % b),
    '∤':lambda a, b: bool(a % b),
    '⊓':lambda a, b: math.gcd(a, b),
    '⊔':lambda a, b: a * b // math.gcd(a, b),
    '⊥':lambda a, b: tobase(a, b),
    '⊤':lambda a, b: frombase(a, b),
    '…':lambda a, b: set(range(a, b+1)),
    '⍟':lambda a, b: math.log(a, b),
    'ⁱ':lambda a, b: list(a).index(b),
    'ⁿ':lambda a, b: a[b % len(a)],
    '‖':lambda a, b: (list(a) if isinstance(a, (list, set)) else [a]) + (list(b) if isinstance(b, (list, set)) else [b]),
    'ᶠ':lambda a, b: a[:b],
    'ᵗ':lambda a, b: a[b:],
    '∓':lambda a, b: [a-b, a+b],
    '∕':lambda a, b: int(a / b),
    '∠':lambda a, b: [math.sin, math.cos, math.tan, math.asin, math.acos, math.atan][b%6](a),
    '≮':lambda a, b: not(a < b),
    '≯':lambda a, b: not(a > b),
    '≰':lambda a, b: not(a <= b),
    '≱':lambda a, b: not(a >= b),
    '∧':lambda a, b: a and b,
    '∨':lambda a, b: a or b,
    '⋇':lambda a, b: [a * b, a // b],
    '⊼':lambda a, b: not(a and b),
    '⊽':lambda a, b: not(a or b),
    '∢':lambda a, b: [math.sinh, math.cosh, math.tanh, math.asinh, math.acosh, math.atanh][b%6](a),
    '⊿':lambda a, b: math.hypot(a, b),
    'j':lambda a, b: complex(a, b),
    '≪':lambda a, b: a << b,
    '≫':lambda a, b: a >> b,
    '⊈':lambda a, b: a.issubset(b) and a != b,
    '⊉':lambda a, b: a.issuperset(b) and a != b,
    '½':lambda a, b: Coords('M', *list(map(lambda a, b: (a + b) / 2, b.axis, a.axis))),
    '→':lambda a, b: Vector(a.name, b.name, *list(map(operator.sub, b.axis, a.axis))),
    '∥':lambda a, b: a.parallel(b),
    '∦':lambda a, b: not a.parallel(b),
    '⟂':lambda a, b: a.perpendicular(b) if isinstance(a, Vector) else (math.gcd(a, b) == 1),
    '⊾':lambda a, b: not a.perpendicular(b) if isinstance(a, Vector) else (math.gcd(a, b) != 1),
    '∡':lambda a, b: a.angle(b),
    '√':lambda a, b: a ** (1 / b),

}

PREFIX_ATOMS = {

    '∑':lambda a: sum(a),
    '∏':lambda a: product(a),
    '#':lambda a: len(a),
    '√':lambda a: math.sqrt(a),
    "'":lambda a: chr(a),
    '?':lambda a: ord(a),
    'Γ':lambda a: math.gamma(a),
    '∤':lambda a: [i for i in range(1, a+1) if a%i == 0],
    '℘':lambda a: set(map(frozenset, powerset(a))),
    'ℑ':lambda a: complex(a).imag,
    'ℜ':lambda a: complex(a).real,
    '∁':lambda a: complex(a).conjugate,
    '≺':lambda a: a - 1,
    '≻':lambda a: a + 1,
    '∪':lambda a: deduplicate(a),
    '⍎':lambda a: eval(a) if type(a) == str else round(a),
    'R':lambda a: (abs(a), a.θ),

}

POSTFIX_ATOMS = {

    '!':lambda a: math.factorial(a),
    '’':lambda a: prime(a),
    '#':lambda a: product([i for i in range(1, a+1) if prime(i)]),
    '²':lambda a: a ** 2,
    '³':lambda a: a ** 3,
    'ᵀ':lambda a: transpose(a),
    'ᴺ':lambda a: sorted(a),
    '°':lambda a: math.degrees(a),
    'ᴿ':lambda a: math.radians(a),
    '₁':lambda a: a.unit,

}

SURROUND_ATOMS = {

    '||':lambda a: abs(a),
    '⌈⌉':lambda a: math.ceil(a),
    '⌊⌋':lambda a: math.floor(a),
    '⌈⌋':lambda a: int(a),
    '[]':lambda a: set(range(a+1)) if type(a) == int else list(a),
    '[)':lambda a: set(range(a)),
    '(]':lambda a: set(range(1, a+1)),
    '()':lambda a: set(range(1, a)),
    '{}':lambda a: set(a),
    '""':lambda a: str(a),
    '‖‖':lambda a: abs(a),

}

PREDICATE_ATOMS = {

     B :lambda a: a in [True, False],
    'ℂ':lambda a: type(a) == complex,
    'ℕ':lambda a: type(a) == int and a > 0,
    'ℙ':lambda a: prime(a),
    'ℝ':lambda a: type(a) in [int, float],
     U :lambda a: type(a) in [int, float, complex] or a in [True, False],
    'ℤ':lambda a: type(a) == int,
    '¬':lambda a: not a,
    '⊤':lambda a: True,
    '⊥':lambda a: False,

}

JUNCTION_ATOMS = {

    '∘':lambda a, b: a == b,
    '∧':lambda a, b: a & b,
    '∨':lambda a, b: a | b,
    '⊕':lambda a, b: a ^ b,

}

NILAD_ATOMS = {

    '½':0.5,
    '∅':set(),
     B :set([0, 1]),
    'ℂ':InfSet('ℂ', lambda a: isinstance(a, complex)),
    'ℕ':InfSet('ℕ', lambda a: int(a) == a > 0),
    'ℙ':InfSet('ℙ', lambda a: prime(a)),
    'ℝ':InfSet('ℝ', lambda a: isinstance(a, (float, int))),
     U :InfSet(U, lambda a: isinstance(a, (bool, complex, float, int))),
    'ℤ':InfSet('ℤ', lambda a: isinstance(a, int)),

}

def runpredicate(code, value):
    segments = re.split(r'[∘∧∨⊕]', code)
    junctions = re.sub(r'[^∘∧∨⊕]', '', code)
    index = 0
    results = [True] * len(segments)

    for index, segment in enumerate(segments):
        cin = 0
        called = False
        while cin < len(segment):
            
            char = segment[cin]
            called = False
                
            if char in INFIX:
                try: right = eval(segment[cin + 1])
                except: right = value
                results[index] = INFIX_ATOMS[char](value, right)
                cin += 1
            if char in PREFIX:
                results[index] = PREFIX_ATOMS[char](value)
            if char in POSTFIX:
                results[index] = POSTFIX_ATOMS[char](value)

            if char in PREDICATE_ATOMS.keys():
                if called:
                    results[index] = PREDICATE_ATOMS[char](results[index])
                else:
                    results[index] = PREDICATE_ATOMS[char](value)

            cin += 1

    final = results.pop(0)
    jindex = 0
    while results:
        cmd = JUNCTION_ATOMS[junctions[jindex]]
        final = cmd(final, results.pop(0))
        
    return '⊤' if final else '⊥'

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

    if PREDICATE.search(joined):
        line = line[1:]

        mode, pred = line 
        if mode == '≻':
            if isinstance(left, (set, list)):
                for value in list(left):
                    if runpredicate(pred, value) == '⊤':
                        return value
                return []
            else:
                while runpredicate(pred, left) == '⊥':
                    left += 1
                return left
            
        if mode == '∑':
            return sum(list(filter(lambda v: runpredicate(pred, v) == '⊤', left)))
        if mode == '∏':
            return product(list(filter(lambda v: runpredicate(pred, v) == '⊤', left)))
        if mode == '#':
            return len(list(filter(lambda v: runpredicate(pred, v) == '⊤', left)))

        if mode == '∀':
            ret = all(runpredicate(pred, value) == '⊤' for value in left)
        if mode == '∃':
            ret = any(runpredicate(pred, value) == '⊤' for value in left)
        if mode == '∄':
            ret = not any(runpredicate(pred, value) == '⊤' for value in left)
        if mode == '⊤':
            ret = runpredicate(pred, left) == '⊤'
        if mode == '⊥':
            ret = runpredicate(pred, left) == '⊥'
        
        assert ret
        return ('⊤' if ret else '⊥')

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
                   sub += execute(tokens, fn, left = n, right = sub)

               if loop == '∑':
                   total += sub
               if loop == '∏':
                   total *= sub
               if loop == '…':
                   total.append(sub)

           return total
        
        if loop == 'Then':
            ret = []
            for ln in targets:
                ret.append(execute(tokens, int(ln)))
            return ret

def deduplicate(array):
    final = []
    for value in array:
        if value not in final:
            final.append(value)
    return final

def frombase(digits, base):
    total = 0
    for index, digit in enumerate(digits[::-1]):
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
        for v in list(value)[:-1]:
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
        try: code = code.replace('> Input\n', '> {}\n'.format(eval(line)), 1)
        except: code = code.replace('> Input\n', '> "{}"\n'.format(line), 1)

    code = code.split('\n')
    final = []

    for line in code:
        for regex in REGEXES:
            if debug: print(repr(line), regex.search(line))
            if regex.search(line):
                final.append(tokenise(regex, line))
        if debug: print()
    return final

def transpose(array):
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

    try:
        return NILAD_ATOMS[value]
    except:
        return value

if __name__ == '__main__':
    try:
        program = sys.argv[1]
    except IndexError:
        sys.exit(0)

    flags = ['--tokens', '-t', '--Tokens', '-T', '--parser', '-p']

    flag = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in flags else False

    try:
        program = open(program, 'r', encoding='utf-8').read()
    except:
        pass

    if flag in ['--tokens', '-t']:
        print(tokenizer(program, CONST_STDIN))
    elif flag in ['--Tokens', '-T']:
        print(*tokenizer(program, CONST_STDIN), sep = '\n')
    elif flag in ['--parser', '-p']:
        tokenizer(program, CONST_STDIN, debug = True)
    else:
        try:
            execute(tokenizer(program, CONST_STDIN))
            if re.search(r'^>>> ', program, re.MULTILINE) and not re.search(r'^>> Output', program, re.MULTILINE):
                output_file = sys.stdout
            else:
                output_file = sys.stderr
                
            print('⊤', file = output_file)
        except AssertionError:
            print('⊥')
