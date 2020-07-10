import functools
import unicodedata

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

BINARY = [

    '+', '⋅', '÷', '-', '*', '%', '/', '⍟',
    # '∘',

]

FUNCS = [

    'sin', 'cos', 'tan', 'csc', 'sec', 'cot',
    'sinh', 'cosh', 'tanh', 'csch', 'sech', 'coth',
    'arcsin', 'arccos', 'arctan', 'arccsc', 'arcsec', 'arccot',
    'arcsinh', 'arccosh', 'arctanh', 'arccsch', 'arcsech', 'arccoth',
    'exp', 'ln', '∂', '∫', 'Γ', 'ℑ', 'ℜ',

]

PREFIX = [

    '√', '∛',

]

POSTFIX = [

    '!', '°', 'ᴿ',

]

PRIORITY = {

    # '∘': 5,

    '!': 5,
    
    '*': 4,
    '⍟': 4,
    '√': 4,
    '∛': 4,

    '⋅': 3,
    '÷': 2,
    '/': 2,
    '%': 2,

    '+': 1,
    '-': 1,

}

NUMBER = (int, float, complex)

class BinaryOp:

    def __init__(self, left = None, op = None, right = None, pri = 0):
        self.left = left
        self.op = op
        self.right = right
        self.pri = pri
        
    def command(self, dictionary):
        self.COMMANDS = dictionary

    def copy(self):
        return BinaryOp(self.left, self.op, self.right, self.pri)

    @property
    def arity(self):
        return (self.left is None) + (self.right is None)

    def __eq__(self, other):
        return hasattr(other, 'op') and self.op == other.op

    def __call__(self, argument):
        try: self.COMMANDS
        except: assert False
            
        cmd = self.COMMANDS[2][self.op]
        if self.op == '∘':
            ret = cmd(self.left, self.right)(argument)
        else:
            left = self.left(argument)
            right = self.right(argument)
            ret = cmd(left, right)

        if type(ret) == float and ret.is_integer():
            return int(ret)
        if type(ret) == complex and ret.imag == 0j:
            return ret.real
        return ret

    def __repr1__(self):
        if self.left is None:
            if self.right is None:
                return self.op
            return '{}{}'.format(self.op, self.right)
        
        if self.right is None:
            return '{}{}'.format(self.left, self.op)
        
        return '{}{}{}'.format(self.left, self.op, self.right)

    def __repr2__(self):
        return 'BinaryOp(left = {}, op = {}, right = {})'.format(repr(self.left), repr(self.op), repr(self.right))
    
    __repr__ = __repr2__

class UnaryOp:

    def __init__(self, pos, op = None, operand = None, pri = 6):
        self.op = op
        self.operand = operand
        self.pos = pos
        self.pri = pri

    def command(self, dictionary):
        self.COMMANDS = dictionary

    def copy(self):
        return UnaryOp(self.pos, self.op, self.operand, self.pri)

    @property
    def arity(self):
        return self.operand is None

    def __eq__(self, other):
        if not isinstance(other, UnaryOp):
            return False
        return (self.pos, self.op, self.operand, self.pri) == (other.pos, other.op, other.operand, other.pri)
        
    def __call__(self, argument):
        try: self.COMMANDS
        except: assert False
        
        cmd = self.COMMANDS[1][self.op]
        argument = self.operand(argument)
        ret = cmd(argument)

        if type(ret) == float and ret.is_integer():
            return int(ret)
        if type(ret) == complex and ret.imag == 0j:
            return ret.real
        return ret

    def __repr1__(self):
        oper = self.operand
        if self.operand is None:
            oper = '⋅'
        
        if self.pos == 'Prefix':
            #if self.op == '-':
            #    return '{}{}'.format(self.op, oper)
            return '{}({})'.format(self.op, oper)
        return '{}{}'.format(oper, self.op)

    def __repr2__(self):
        return '<{}>UnaryOp(op = {}, operand = {})'.format(self.pos, repr(self.op), repr(self.operand))

    __repr__ = __repr2__
    
class Variable:

    def __init__(self, name, value = None):
        self.name = name
        self.value = value
        self.arity = 0
        self.op = None
        self.pri = 6

    def copy(self):
        return Variable(self.name, self.value)

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        
        if self.name:
            if self.value:
                return (self.name == other.name) and (self.value == other.value)
            return (self.name == other.name) and (other.value is None)

        if self.value is not None:
            return (self.value == other.value) and (other.value is not None)

        return (other.value is None)

    def __call__(self, value = None):
        if self.value is None:
            return value
        return self.value

    def __repr1__(self):
        if self.value is None:
            return '{}'.format(self.name)
        if self.name is None:
            return '{}'.format(self.value)
        
        return '{}'.format(self.name)

    def __repr2__(self):
        if self.value is None:
            return 'Variable({})'.format(self.name)
        if self.name is None:
            return 'Variable({})'.format(self.value)
        
        return 'Variable({} = {})'.format(self.name, self.value)

    __repr__ = __repr2__

class Constant:

    def __init__(self, value):
        self.value = value
        self.arity = 0
        self.op = None
        self.pri = 6

    def copy(self):
        return Constant(self.value)

    def __eq__(self, other):
        if not isinstance(other, (Variable, Constant)):
            return False

        return self.value == other.value

    def __repr1__(self):
        return str(self.value)

    def __repr2__(self):
        return 'Constant({})'.format(self.value)

    __repr__ = __repr2__

class Function:

    def __init__(self, name, body = None):
        self.name = name
        self.body = body
        self.arity = -1
        self.op = None
        self.pri = 6

    def __call__(self, argument):
        if self.body is None:
            return argument
        return self.body(argument)

    def __repr__(self):
        return 'Function<{}>'.format(self.name)

def strip(string):
    if string[0] != '(' or string[-1] != ')':
        return string
    
    parts = []
    depth = 0
    for char in string:
        if char == '(':
            if depth:
                parts[-1] += char
                depth += 1
            else:
                depth += 1
                parts.append('(')
                parts.append('')
        elif char == ')':
            depth -= 1
            if depth:
                parts[-1] += char
            else:
                parts.append(')')
                parts.append('')
        else:
            parts[-1] += char

    parts = list(filter(None, parts))

    if len(parts) == 3:
        return strip(''.join(parts[1:-1]))

    return string

def tokeniser(string):
    tokens = []
    curr = ''
    index = 0

    string = strip(string)

    while index < len(string):
        if string[index] == '-' and False:
            
            if string[index + 1] in '123456789':
                if curr:
                    tokens.append(curr)

                curr = '-'
                index += 1
                
                while index < len(string) and string[index] in '1234567890':
                    curr += string[index]
                    index += 1
            
            elif string[index + 1] == '0':
                
                try:
                    
                    if string[index + 2] == '.':
                        if curr:
                            tokens.append(curr)
                        
                        curr = '-0.'
                        index += 2
                        
                        while index < len(string) and string[index] in '1234567890':
                            curr += string[index]
                            index += 1
                    else:
                        raise SyntaxError
                            
                except IndexError:
                    tokens.append(curr)

            else:
                if curr:
                    tokens.append(curr)
                curr = '-'
                index += 1

            index -= 1

        elif string[index] in '123456789':
            if curr:
                tokens.append(curr)

            curr = ''
            
            while index < len(string) and string[index] in '1234567890':
                curr += string[index]
                index += 1

            index -= 1

        elif string[index] in letters:
            if curr:
               tokens.append(curr)
            curr = ''

            while index < len(string) and string[index] in letters + '⁰¹²³⁴⁵⁶⁷⁸⁹':
                curr += string[index]
                index += 1
            index -= 1

        elif string[index] in '¹²³⁴⁵⁶⁷⁸⁹':
            if curr:
               tokens.append(curr)
            curr = ''

            while index < len(string) and string[index] in '⁰¹²³⁴⁵⁶⁷⁸⁹':
                curr += string[index]
                index += 1
            index -= 1

        else:
            if curr:
                tokens.append(curr)
                curr = ''
                
            tokens.append(string[index])

        index += 1

    if curr:
        tokens.append(curr)

    del curr
    return list(filter(lambda a: a != ' ', tokens))

def ispower(string):
    return string[1:] and all(i in '⁰¹²³⁴⁵⁶⁷⁸⁹' for i in string[1:])

def isexp(string):
    return string and all(i in '⁰¹²³⁴⁵⁶⁷⁸⁹' for i in string)

def getpower(string):
    return ''.join(filter(lambda a: a in '⁰¹²³⁴⁵⁶⁷⁸⁹', string))

def normalise(string):
        if not string:
                return 1
        total = 0
        for exp, digit in enumerate(map(unicodedata.numeric, string)):
                total += digit * 10 ** exp
        return int(total)

def evalable(string):
    try:
        eval(string)
        return True
    except:
        return False
    
def getparens(string):
    endexpr = -1
    depth = 1
    while depth:
        endexpr += 1
        if string[endexpr] == '(':
            depth += 1
        if string[endexpr] == ')':
            depth -= 1
        
    return parse(string[:endexpr], False), endexpr

def juxtaposition(left, right):
    if not isinstance(left, Constant):
        return False

    if isinstance(right, Variable):
        return True

    if isinstance(right, BinaryOp):
        return False

    return True

def order_key(value):
    value = value[1]
    if isinstance(value, BinaryOp):
        if value.left is not None and value.right is not None:
            return 6

    if isinstance(value, UnaryOp):
        if value.operand is not None:
            return 6

    return value.pri

def reduce(tkns):
    order = list(map(lambda a: a[0], sorted(enumerate(tkns), key = order_key)))
    cmd = tkns[order[0]]

    if isinstance(cmd, BinaryOp) and (cmd.left is None or cmd.right is None):
        cmd.left = reduce(tkns[:order[0]])
        cmd.right = reduce(tkns[order[0] + 1:])

    if isinstance(cmd, UnaryOp) and cmd.operand is None:
        if cmd.pos == 'Prefix':
            cmd.operand = reduce(tkns[order[0] + 1:])
        if cmd.pos == 'Postfix':
            if isexp(cmd.op):
                cmd.operand = reduce(tkns[order[0] + 1:])
            else:
                cmd.operand = reduce(tkns[:order[0]])

    return cmd

def parse(string, gettokens = True, red = True):
    if not string:
        return Constant(0)
    original = string
    if gettokens:
        string = tokeniser(string)
    
    ops = []
    skips = 0
    for index, tkn in enumerate(string, 1):
        if skips:
            skips -= 1
            continue

        if tkn == '-' and index == 1:
            tkn = Constant(-1)

        elif ispower(tkn):
            power = getpower(tkn)
            var = tkn.strip('⁰¹²³⁴⁵⁶⁷⁸⁹')
            tkn = UnaryOp('Postfix', op = power, operand = Variable(var), pri = 3)

        elif isexp(tkn):
            tkn = UnaryOp('Postfix', op = tkn, pri = 3)
            
        elif tkn in BINARY:
            tkn = BinaryOp(op = tkn, pri = PRIORITY[tkn])

        elif tkn == '(':
            tkn, skips = getparens(string[index:])

        elif evalable(tkn):
            tkn = Constant(eval(tkn))

        elif tkn in PREFIX:
            tkn = UnaryOp('Prefix', op = tkn, pri = PRIORITY[tkn])

        elif tkn in FUNCS:
            tkn = UnaryOp('Prefix', op = tkn)

        else:
            if tkn.isalpha():
                tkn = Variable(tkn)

        ops.append(tkn)

    if len(ops) == 1:
        return ops[0]

    if isinstance(ops[-1], UnaryOp) and ops[-1].pos == 'Postfix' and ops[-2] == ')':
        ops = [ops.pop()] + ops

    types = (UnaryOp, BinaryOp, Variable, Constant, Function)
    ops = list(filter(lambda a: isinstance(a, types), ops))

    if not red:
        return ops

    inserts = []
    for index, pair in enumerate(zip(ops, ops[1:])):
        if juxtaposition(*pair):
            inserts.append(index + 1)

    tkns = []
    for i, op in enumerate(ops):
        if i in inserts:
            tkns.append(BinaryOp(op = '⋅', pri = 3))
        tkns.append(op)

    return reduce(tkns)

X = Variable('x')

if __name__ == '__main__':
    print(parse(input()))
