import functools
import math
import operator
import re
import sys

OPERATOR = re.compile(r'''^(>> )(?:(\d+)([=≠><≥≤+−±×÷^%∆∩∪⊆⊂⊄⊅⊃⊇\\∈∉])(\d+)|((\|)|(⌈)|(⌊))(\d+)((?(2)\||(?(3)⌉|⌋)))|([√∑∏#])(\d+)|(\d+)([!’]))$''')
STREAM = re.compile(r'''^(>>? )(?:(Output )(\d+ )*(\d+)|(Input(?:All)?)|(Error ?)(\d+)?)$''')
NILAD = re.compile(r'''^(> )((((")|('))(?(5)[^"]|[^'])*(?(5)"|'))|(-?\d+\.\d+|-?\d+)|([[{]((-?\d+(\.\d+)?, ?)*-?\d+(\.\d+)?)*[}\]]))$''')
LOOP = re.compile(r'''^(>> )(?:(While|For|If)( \d+){2})|(?:(Each )(\d+))$''')
INFIX = '=≠><≥≤+−±×÷*%∆∩∪⊆⊂⊄⊅⊃⊇\∈∉∧∨⊕'
PREFIX = '∑∏#√'
POSTFIX = '!’'
SURROUND = ['||', '⌈⌉', '⌊⌋']
REGEXES = [OPERATOR, STREAM, NILAD, LOOP]
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

}

PREFIX_ATOMS = {

    '∑':lambda a: sum(a),
    '∏':lambda a: functools.reduce(operator.mul, a),
    '#':lambda a: len(a),
    '√':lambda a: math.sqrt(a),

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

def execute(tokens, index=-1):
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
            target = int(line[1])-1
            return atom(execute(tokens, target))

        if line[1] in POSTFIX:
            assert len(line) == 2
            atom = POSTFIX_ATOMS[line[1]]
            target = int(line[0])-1
            return atom(execute(tokens, target))

        if line[1] in INFIX:
            assert len(line) == 3
            left, atom, right = line
            left, right = map(lambda a: execute(tokens, int(a)-1), [left, right])
            atom = INFIX_ATOMS[atom]
            return atom(left, right)

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

def prime(n):
    for i in range(2, int(n)):
        if n%i == 0: return False
    return n > 1 and type(n) == int

def tokenise(regex, string):
    result = list(filter(None, regex.match(string).groups()))
    if result[0] == '> ':
        return result[:2]
    return deduplicate(result)

def tokenizer(code):
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
        if value == 'Input':
            return next(STDIN)
        if value == 'InputAll':
            return CONST_STDIN
    return value

STDIN = iter(list(map(lambda a: tryeval(a, stdin=False), CONST_STDIN.split('\n'))) + [0])

if __name__ == '__main__':
    program = sys.argv[1]
    try:
        program = open(program, 'r', encoding='utf-8').read()
    except:
        pass
    execute(tokenizer(program))
