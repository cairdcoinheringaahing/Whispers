import functools
import itertools
import math
import operator
import re
import sys
import unicodedata

sys.setrecursionlimit(1 << 16)

B = chr(120121)
U = chr(120140)

φ = (1 + 5 ** 0.5) / 2
π = math.pi
e = math.e

product = functools.partial(functools.reduce, operator.mul)
findall = lambda string, regex: list(filter(None, regex.match(string).groups()))
normalise = lambda string: int(frombase(list(map(unicodedata.numeric, string)), 10))

PRED    = B + 'ℂℕℙℝ' + U + 'ℤ¬⊤⊥'
INFIX   = '=≠><≥≤+-±⋅×÷*%∆∩∪⊆⊂⊄⊅⊃⊇∖∈∉«»∤∣⊓⊔∘⊤⊥…⍟ⁱⁿ‖ᶠᵗ∓∕∠≮≯≰≱∧∨⋇⊼⊽∢⊿j≪≫'
PREFIX  = "∑∏#√?'Γ∤℘ℑℜ∁≺≻∪⍎"
POSTFIX = '!’#²³ᵀᴺ°ᴿ'
OPEN    = '|(\[⌈⌊{"'
CLOSE   = '|)\]⌉⌋}"'
LAMB    = 'λᶿᵝᵠ⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉₍₎ᵦᵩ'

GROUPS = re.compile(r'''
	(?:
		({0})
		(\d+)
		({0})
	)
	|
	(?:
		({0})
		(\d+)
	)
	|
	(?:
		(\d+)
		({0})
	)
	|
	(?:
		(\d+)
		(ᶿ)
		({1})
	)'''.format('[ᵝᵠ⁰¹²³⁴⁵⁶⁷⁸⁹]+', '[₀₁₂₃₄₅₆₇₈₉₍₎ᵦᵩ]+'), re.VERBOSE)

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

LAMBDA = re.compile(r'''
	^
	(>>\ )
	(λ)
	((?:
		(?:
			(?:
				\ 
				\d+
				ᶿ
				[₀₁₂₃₄₅₆₇₈₉₍₎ᵦᵩ]+
			)
			|
			(?:
				\ 
				[⁰¹²³⁴⁵⁶⁷⁸⁹ᵝᵠ]+
				\d+
				[⁰¹²³⁴⁵⁶⁷⁸⁹ᵝᵠ]+
			)
			|
			(?:
				\ 
				[⁰¹²³⁴⁵⁶⁷⁸⁹ᵝᵠ]+
				\d+
			)
			|
			(?:
				\ 
				\d+
				[⁰¹²³⁴⁵⁶⁷⁸⁹ᵝᵠ]+
			)
		)+
	)
	|
	(?:
		(?:\ \d+)+
	)
	)$''', re.VERBOSE)

STREAM = re.compile(r'''
	^(>>?\ )
	(?:
		(Output\ )
		((?:
			(?:\d+|[LR])
		\ )*)
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
			(-?[1-9]\d*|0)\.\d+
		|
			(-?[1-9]\d*|0)
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
		(?:\ \d+|\ [LR])+
	)
	$
	''', re.VERBOSE)

REGEXES = [PREDICATE, OPERATOR, LAMBDA, STREAM, NILAD, LOOP]
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
	'⋇':lambda a, b: [a * b, a / b],
	'⊼':lambda a, b: not(a and b),
	'⊽':lambda a, b: not(a or b),
	'∢':lambda a, b: [math.sinh, math.cosh, math.tanh, math.asinh, math.acosh, math.atanh][b%6](a),
	'⊿':lambda a, b: math.hypot(a, b),
	'j':lambda a, b: complex(a, b),
	'≪':lambda a, b: a << b,
	'≫':lambda a, b: a >> b,

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

def execute(tokens, index = 0, left = None, right = None, args = None):

    def getvalue(value):
        if value == 'L':
            return left  if left  is not None else 0
        if value == 'R':
            return right if right is not None else 0

        if all(c in '⁰¹²³⁴⁵⁶⁷⁸⁹' for c in value):
            return args[normalise(value) - 1] if args is not None else 0
        
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
                print(tokens, larg, rarg)
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

    if LAMBDA.search(joined):
        line = line[2]

        if re.search(r'^(\ \d+)+$', line):
            target, *argtarget = line.strip().split()
            target = int(target)
            argtarget = list(map(getvalue, argtarget))
            return execute(tokens, target, args = argtarget)

        else:
            statements = line.split()
            results = []
            for index, state in enumerate(statements):

                groups = findall(state, GROUPS)

                if len(groups) == 3:

                    if groups[1] == 'ᶿ':
                        indexes = re.findall('(₍[₀₁₂₃₄₅₆₇₈₉]+₎)|([₀₁₂₃₄₅₆₇₈₉ᵦᵩ])', groups[2])
                        indexes = list(filter(None, sum(indexes, tuple())))
                        indexes = list(map(lambda a: normalise(a.strip('₍₎')), indexes))

                        subret = list(map(lambda i: results[i - 1], indexes))
                        target = int(groups[0])
                        results.append(execute(tokens, target, left = subret))
                        continue

                    if groups[0] not in 'ᵝᵠ':
                        larg = getvalue(groups[0])
                    elif groups[0] == 'ᵝ':
                        try: larg = results[index - 1]
                        except: larg = getvalue('¹')
                    elif groups[0] == 'ᵠ':
                        try: larg = results[index + 1]
                        except: larg = getvalue('²')
                    
                    if groups[2] not in 'ᵝᵠ':
                        rarg = getvalue(groups[2])
                    elif groups[2] == 'ᵝ':
                        try: rarg = results[index - 1]
                        except: rarg = getvalue('¹')
                    elif groups[2] == 'ᵠ':
                        try: rarg = results[index + 1]
                        except: rarg = getvalue('²')

                    target = int(groups[1])
                    results.append(execute(tokens, target, left = larg, right = rarg))

                if len(groups) == 2:
                    if groups[0].isdecimal():
                        target = int(groups[0])

                        if groups[1] not in 'ᵝᵠ':
                            larg = getvalue(groups[1])
                        elif groups[1] == 'ᵝ':
                            try: larg = results[index - 1]
                            except: larg = getvalue('¹')
                        elif groups[1] == 'ᵠ':
                            try: larg = results[index + 1]
                            except: larg = getvalue('²')
                       
                        results.append(execute(tokens, target, left = larg))

                    else:
                        target = int(groups[1])

                        if groups[0] not in 'ᵝᵠ':
                            larg = getvalue(groups[0])
                        elif groups[0] == 'ᵝ':
                            try: larg = results[index - 1]
                            except: larg = getvalue('¹')
                        elif groups[0] == 'ᵠ':
                            try: larg = results[index + 1]
                            except: larg = getvalue('²')
                       
                        results.append(execute(tokens, target, right = larg))

                if len(groups) == 1:
                    target = int(groups[0])
                    results.append(execute(tokens, target))

            return results.pop()

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
                print('⊤')
        except AssertionError:
            print('⊥')
