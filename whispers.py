import functools
import itertools
import math
import operator
import re
import sys

sys.setrecursionlimit(1 << 16)

B = chr(120121)
U = chr(120140)

φ = (1 + 5 ** 0.5) / 2
π = math.pi
e = math.e

PRED    = B + 'ℂℕℙℝ' + U + 'ℤ¬⊤⊥'
INFIX   = '=≠><≥≤+-±⋅×÷*%∆∩∪⊆⊂⊄⊅⊃⊇∖∈∉«»∤∣⊓⊔∘⊤⊥…⍟ᵢₙ'
PREFIX  = "∑∏#√?'Γ∤℘ℑℜ∁≺≻"
POSTFIX = '!’#²³'
OPEN    = '|(\[⌈⌊{'
CLOSE   = '|)\]⌉⌋}'

PREDICATE = re.compile(r'''
	^
	(>>>\ )
	([∀∃∄⊤⊥∑#≻])
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
		([1-9]\d*|[LR])([{}])([1-9]\d*|[LR])
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
		(
			(?:\d+|[LR])
		\ )*
		(\d+|[LR])
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
			-?[1-9]\d*\.\d+
		|
			-?[1-9]\d*
		)
	|
		(
			[\[{]
			(
				(
					-?[1-9]\d*
						(\.\d+)?
					,\ ?
				)*
				-?[1-9]\d*
					(\.\d+)?
			)*
			[}\]]
		)
	|
		(
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
		)
	)
	$
	''', re.VERBOSE)

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
		(?:\ \d+|[LR])+
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
    '×':lambda a, b: set(map(frozenset, itertools.product(a, b))),
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
    '⊔':lambda a, b: a*b//math.gcd(a, b),
    '⊥':lambda a, b: tobase(a, b),
    '⊤':lambda a, b: frombase(a, b),
    '…':lambda a, b: set(range(a, b+1)),
    '⍟':lambda a, b: math.log(a, b),
    'ᵢ':lambda a, b: list(a).index(b),
    'ₙ':lambda a, b: a[b],

}

PREFIX_ATOMS = {

    '∑':lambda a: sum(a),
    '∏':lambda a: functools.reduce(operator.mul, a),
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

}

POSTFIX_ATOMS = {

    '!':lambda a: math.factorial(a),
    '’':lambda a: prime(a),
    '#':lambda a: functools.reduce(operator.mul, [i for i in range(1, a+1) if prime(i)]),
    '²':lambda a: a ** 2,
    '³':lambda a: a ** 3,

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

def execute(tokens, index=-1, left=None, right=None):
    if not tokens:
        return
    line = tokens[index]
    mode = line[0].count('>')

    if mode == 1:
        return tryeval(line[1])

    joined = ''.join(line)

    if PREDICATE.search(joined):
        line = line[1:]

        mode, pred = line 
        if mode == '≻':
            if isinstance(left, (set, list)):
                for value in left:
                    if runpredicate(pred, value) == '⊤':
                        return value
            else:
                while runpredicate(pred, left) == '⊥':
                    left += 1
                return left
            
        if mode == '∑':
            return sum(runpredicate(pred, value) == '⊤' for value in left)
        if mode == '#':
            return list(filter(lambda v: runpredicate(pred, v) == '⊤', left))

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
            target = left if line[1] == 'L' else execute(tokens, int(line[1])-1)
            return atom(target)

        if line[1] in POSTFIX:
            atom = POSTFIX_ATOMS[line[1]]
            target = left if line[0] == 'L' else execute(tokens, int(line[0])-1)
            return atom(target)

        if line[1] in INFIX:
            larg, atom, rarg = line
            if atom == '∘':
                larg = int(larg)-1
                rarg = execute(tokens, int(rarg)-1)
                return execute(tokens, larg, rarg)
            else:
                larg = left if larg == 'L' else execute(tokens, int(larg)-1)
                rarg = right if rarg == 'R' else execute(tokens, int(rarg)-1)
                atom = INFIX_ATOMS[atom]
                return atom(larg, rarg)

        if line[0] in OPEN and line[2] in CLOSE:
            try:
                atom = SURROUND_ATOMS[line[0] + line[2]]
            except:
                atom = lambda a: a

            target = int(line[1])-1
            return atom(execute(tokens, target))

    if STREAM.search(joined):
        if line[1] == 'Output ':
            targets = line[2:]
            for target in targets:
                output(execute(tokens, int(target)-1))
        if line[1].strip() == 'Error':
            output(execute(tokens, int(line[2])-1), -1)
            sys.exit()

    if LOOP.search(joined):
        loop = line[1]
        targets = list(map(lambda a: int(a)-1, line[2].split()))

        if loop == 'While':
            cond, call, *_ = targets
            while execute(tokens, cond):
                execute(tokens, call)
                
        if loop == 'DoWhile':
            cond, call, *_ = targets
            execute(tokens, call)
            while execute(tokens, cond):
                execute(tokens, call)

        if loop == 'For':
            iters, call, *_ = targets
            for _ in range(execute(tokens, iters)):
                execute(tokens, call)
            
        if loop == 'If':
            cond, true, *false = targets
            false = false[:1]
            if execute(tokens, cond):
                execute(tokens, true)
            else:
                if false: execute(tokens, false[0])
                else: return 0

        if loop == 'Each':
            call, *iters = targets
            result, final = [], []
            for tgt in iters:
                res = execute(tokens, tgt)
                result.append(res if hasattr(res, '__iter__') else [res])
            result = list(map(list, zip(*result)))
            for args in result:
                while len(args) != 2: args.append(None)
                argd = {'index':call, 'left':args[0], 'right':args[1]}
                final.append(execute(tokens, **argd))
            if all(type(a) == str for a in final):
                return ''.join(final)
            if len(final) == 1:
                return final[0]
            return final

        if loop in '∑∏…':
           start, end, *f = targets
           start = execute(tokens, start)
           end = execute(tokens, end) + 1

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
            for ln in targets:
                yield execute(tokens, ln)

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

def frombase(digits, base):
    total = 0
    for index, digit in enumerate(digits[::-1]):
        total += digit * base ** index
    return total

def tokenise(regex, string):
    result = list(filter(None, regex.match(string).groups()))
    if result[0] == '> ':
        return result[:2]
    return result

def tokenizer(code, stdin):
    for line in stdin.split('\n'):
        try: code = code.replace('> Input\n', '> {}\n'.format(eval(line)), 1)
        except: code = code.replace('> Input\n', '> "{}"\n'.format(line), 1)
    code = code.split('\n')
    final = []
    for line in code:
        for regex in REGEXES:
            if regex.search(line):
                final.append(tokenise(regex, line))
    return final

def tryeval(value, stdin=True):
    try:
        return eval(value)
    except:
        if not stdin:
            return value
        if value == 'InputAll':
            return CONST_STDIN
    return value

if __name__ == '__main__':
    program = sys.argv[1]
    flag = sys.argv[2] in ['--tokens', '-t'] if len(sys.argv) > 2 else False

    try:
        program = open(program, 'r', encoding='utf-8').read()
    except:
        pass

    if flag:
        print(tokenizer(program, CONST_STDIN))
    else:
        try:
            execute(tokenizer(program, CONST_STDIN))
            if re.search(r'^>>> ', program, re.MULTILINE) and not re.search(r'^>> Output', program, re.MULTILINE):
                print('⊤')
        except AssertionError:
            print('⊥')
