import functools
import math
import operator
import re
import sys

sys.setrecursionlimit(1 << 16)

INFIX = '=≠><≥≤+-±×÷*%∆∩∪⊆⊂⊄⊅⊃⊇\∈∉«»'
PREFIX = "∑∏#√?'"
POSTFIX = '!’'
SURROUND = ['||', '⌈⌉', '⌊⌋']
EXTENSION = ['Reduce', 'Cumulate', 'Range']

OPERATOR = re.compile(r'''^(>> )(?:(\d+|[LR])([{}])(\d+|[LR])|((\|)|(⌈)|(⌊))(\d+|[LR])((?(2)\||(?(3)⌉|⌋)))|([{}])(\d+|[LR])|(\d+|[LR])([{}]))$'''.format(INFIX, PREFIX, POSTFIX))
STREAM = re.compile(r'''^(>>? )(?:(Output )((?:\d+|[LR]) )*(\d+|[LR])|(Input(?:All)?)|(Error ?)(\d+|[LR])?)$''')
NILAD = re.compile(r'''^(> )((((")|('))(?(5)[^"]|[^'])*(?(5)"|'))|(-?\d+\.\d+|-?\d+)|([[{]((-?\d+(\.\d+)?, ?)*-?\d+(\.\d+)?)*[}\]]))$''')
LOOP = re.compile(r'''^(>> )(While|For|If|Each|Cycle)((?: \d+|[LR])+)$''')
EXT = re.compile(r'''^(>> )(E:(?:{}))((?: \d+|[LR])+)$'''.format('|'.join(EXTENSION)))
REGEXES = [OPERATOR, STREAM, NILAD, LOOP, EXT]
CONST_STDIN = sys.stdin.read()

EXTENSION_ATOMS = {

    'E:Range':lambda a: list(range(1, a+1)),

}

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
    '×':lambda a, b: a * b,
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
    '\\':lambda a, b: set(i for i in a if i not in b),
    '∈':lambda a, b: a in b,
    '∉':lambda a, b: a not in b,
    '«':lambda a, b: min(a, b),
    '»':lambda a, b: max(a, b),

}

PREFIX_ATOMS = {

    '∑':lambda a: sum(a),
    '∏':lambda a: functools.reduce(operator.mul, a),
    '#':lambda a: len(a),
    '√':lambda a: math.sqrt(a),
    "'":lambda a: chr(a),
    '?':lambda a: ord(a),

}

POSTFIX_ATOMS = {

    '!':lambda a: math.factorial(a),
    '’':lambda a: prime(a),

}

SURROUND_ATOMS = {

    '||':lambda a: abs(a),
    '⌈⌉':lambda a: math.ceil(a),
    '⌊⌋':lambda a: math.floor(a),

}

def deduplicate(array):
    final = []
    for element in array:
        if element not in final:
            final.append(element)
    return final

def execute(tokens, index=-1, left=None, right=None):
    if not tokens:
        return
    line = tokens[index]
    mode = line[0].count('>')

    if mode == 1:
        return tryeval(line[1])

    joined = ''.join(line)

    if OPERATOR.search(joined):
        line = line[1:]

        if line[0] in PREFIX:
            assert len(line) == 2
            atom = PREFIX_ATOMS[line[0]]
            target = left if line[1] == 'L' else int(line[1])-1
            return atom(execute(tokens, target))

        if line[1] in POSTFIX:
            assert len(line) == 2
            atom = POSTFIX_ATOMS[line[1]]
            target = left if line[0] == 'L' else int(line[0])-1
            return atom(execute(tokens, target))

        if line[1] in INFIX:
            assert len(line) == 3
            larg, atom, rarg = line
            larg = left if larg == 'L' else execute(tokens, int(larg)-1)
            rarg = right if rarg == 'R' else execute(tokens, int(rarg)-1)
            atom = INFIX_ATOMS[atom]
            return atom(larg, rarg)

        if line[0] + line[2] in SURROUND:
            atom = SURROUND_ATOMS[line[0] + line[2]]
            target = int(line[1])-1
            return atom(execute(tokens, target))

    if STREAM.search(joined):
        if line[1] == 'Output ':
            targets = line[2:]
            for target in targets:
                print(execute(tokens, int(target)-1), end='')
        if line[1].strip() == 'Error':
            print(execute(tokens, int(line[2])-1), file=sys.stderr)
            sys.exit()

    if LOOP.search(joined):
        loop = line[1]
        targets = list(map(lambda a: int(a)-1, line[2].split()))

        if loop == 'While':
            cond, call, *_ = targets
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

    if EXT.search(joined):
        target = list(map(lambda a: int(a)-1, line[2].split()))[0]
        atom = EXTENSION_ATOMS[line[1]]
        return atom(execute(tokens, target))

def prime(n):
    for i in range(2, int(n)):
        if n%i == 0: return False
    return n > 1 and type(n) == int

def tokenise(regex, string):
    result = list(filter(None, regex.match(string).groups()))
    if result[0] == '> ':
        return result[:2]
    return deduplicate(result)

def tokenizer(code, stdin):
    for line in stdin.split('\n'):
        try: code = code.replace('> Input', '> {}'.format(eval(line)), 1)
        except: code = code.replace('> Input', '> "{}"'.format(line), 1)
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
    if flag: print(tokenizer(program, CONST_STDIN))
    else: execute(tokenizer(program, CONST_STDIN))
