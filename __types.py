import functools
import math
import operator

φ = (1 + 5 ** 0.5) / 2
π = math.pi
e = math.e

product = functools.partial(functools.reduce, operator.mul)
square = lambda a: a ** 2

def terms(value):
    if isinstance(value, int) or value.is_integer():
        return (int(value), 0)

    possible = (π / value, value / π, value * π)
    for selec, poss in enumerate(possible):
        poss = round(poss, 10)
        
        if round(abs(poss), 10) == round(π, 10):
            ratio = math.copysign(1, poss).as_integer_ratio()[::-1]
            
        elif poss == round(poss, 3):
            ratio = poss.as_integer_ratio()
            
        else:
            continue
        
        break
    
    else:
        ratio = (1 / value / π).as_integer_ratio()
        selec = 3

    action = [
        lambda a: a[::-1],
        lambda a: a,
        lambda a: (a[0], π*a[1]),
        lambda a: (a[1], π*a[0]),
    ][selec]

    return action(ratio)

def flatten(array):
    flat = []
    for elem in array:
        if isinstance(elem, list):
            flat += flatten(elem)
        else:
            flat.append(elem)
    return flat

def isjagged(array):
    lengths = []
    for value in array:
        if not hasattr(value, '__iter__'):
            return True
        lengths.append(len(value))
        
    return not all(elem == lengths[0] for elem in lengths)

class Vector:
    def __init__(self, start, *components):
        if isinstance(components[0], str):
            end, *components = components
            self.named = False
            self.name = '({}→{})'.format(start, end)
        else:
            self.named = True
            end = None
            self.name = start
            
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
        return '{}=({})'.format(self.start, ' '.join(map(str, self.parts)))

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

    def __eq__(self, other):
        return all(map(operator.eq, self.parts, other.parts))

    def angle(self, other):
        return math.degrees(math.acos((self * other) / (abs(self) * abs(other))))

    def parallel(self, other):
        return Vector.angle(self, other) == 0

    def perpendicular(self, other):
        return Vector.angle(self, other) == 90

    def x(self, other):
        dims = (len(self.parts), len(other.parts))
        if dims != (3, 3):
            raise TypeError('Cross-product only defined between 2 3D vectors')
        a = self.parts
        b = other.parts

        return Vector(
            '({}×{})'.format(self.name, other.name),
            (a[1] * b[2] - a[2] * b[1]),
            (a[0] * b[2] - a[2] * b[0]),
            (a[0] * b[1] - a[1] * b[0])
        )

class Coords:
    def __init__(self, name, *axis):
        self.name = name
        self.axis = axis

    def __repr__(self):
        return '{}({})'.format(self.name, ', '.join(map(str, self.axis)))

class InfSet:
    def __init__(self, bold, condition, countable = False, k = None):
        self.bold = bold
        self.cond = condition
        self.countable = countable
        
        if countable:
            self.k = k

            def gen():
                while True:
                    if condition(self.k):
                        yield self.k
                    self.k += 1

        else:

            def gen():
                while True:
                    yield None
                    
        self.gen = gen

    def __repr__(self):
        return 'x ∈ {}'.format(self.bold)

    def __str__(self):
        return self.bold

    def __contains__(self, other):
        return self.cond(other)

    def __add__(self, other):
        return InfSet('{}∪{}'.format(self.bold, other.bold), lambda a: self.cond(a) or other.cond(a))

    def __mul__(self, other):
        return InfSet('{}∩{}'.format(self.bold, other.bold), lambda a: self.cond(a) and other.cond(a))

    def __iter__(self):
        while True:
            yield next(self.gen())

    def __next__(self):
        return next(self.gen())

    __or__   = __add__
    __ror__  = __add__
    __radd__ = __add__
    __and__  = __mul__
    __rand__ = __mul__
    __rmul__ = __mul__

class Radian(float):
    def __init__(self, value):
        self.value = terms(value)
        
        if product(self.value) < 0:
            self.mult = -1
            self.value = list(map(abs, self.value))
        else:
            self.mult = 1

    def __repr__(self):
        if self.value[1] == 0:
            return str(self.value[0])
        
        if self.value[0] == self.value[1]:
            return '{}π'.format(self.disp)
        
        if self.value[0] == 1:
            return '{}π/{}'.format(self.disp, self.value[1])
        
        if self.value[1] == 1:
            return '{}{}π'.format(self.disp, self.value[0])
        
        return '{}{}π/{}'.format(self.disp, *self.value)

    @property
    def disp(self):
        return '-' * (self.mult == -1)

    def __abs__(self):
        n, d = self.value
        return Radian((n * π) / d)

    def __add__(self, other):
        return Radian(super().__add__(other))

    def __floordiv__(self, other):
        return Radian(super().__floordiv__(other))

    def __mul__(self, other):
        return Radian(super().__mul__(other))

    def __neg__(self):
        n, d = self.value
        m = -self.mult
        return Radian((m * n * π) / d)

    def __pos__(self):
        n, d = self.value
        return Radian((n * π) / d)

    def __rfloordiv__(self, other):
        return Radian(super().__rfloordiv__(other))

    def __rsub__(self, other):
        return Radian(super().__rsub__(other))

    def __rtruediv__(self, other):
        return Radian(super().__rtruediv__(other))

    def __sub__(self, other):
        return Radian(super().__sub__(other))

    def __truediv__(self, other):
        return Radian(super().__truediv__(other))

    def as_integer_ratio(self):
        return self.value

    def is_integer(self):
        return self.value[1] == 0

    def as_float(self):
        return self.value[0] / self.value[1]

    def easy(self):
        if any(i > 18 for i in self.value):
            return self.as_float()
        return self

    __radd__ = __add__
    __rmul__ = __mul__
    __str__ = __repr__

class Matrix:
    def __init__(self, arg, second = None, zero = False):
        self.next_row = 1
        self.next_column = 1
        
        if second is None:
            if isinstance(arg, Matrix):
                arg = arg.value
                
            if not hasattr(arg, '__iter__'):
                self.value = [[arg]]
                self.rows = self.columns = 1
                self.dims = [1, 1]
                
            else:
                arr = arg.copy()
                arg = []
                for a in arr:
                    if not hasattr(a, '__iter__'):
                        a = [a]
                    arg.append(a)
                    
                if isjagged(arg):
                    self.rows = len(arg)
                    self.columns = max(map(len, arg))
                    self.value = []
                    
                    for row in arg:
                        self.value.append([])
                        for index in range(self.columns):
                            if index < len(row):
                                self.value[-1].append(row[index])
                            else:
                                self.value[-1].append(0)
                            
                else:
                    self.rows = len(arg)
                    self.columns = len(arg[0])
                    mat = Matrix(self.rows, self.columns)
                    for value in flatten(arg):
                        mat.add_next(value)
                    self.value = mat.value.copy()
                                
                self.dims = [self.rows, self.columns]
            
        else:
            rows = arg
            columns = second
            self.value = []
            self.rows = rows
            self.columns = columns
            self.dims = [rows, columns]
            
            for i in range(rows):
                temp = []
                for j in range(columns):
                    temp.append(int(i == j) * (not zero))
                self.value.append(temp.copy())
                temp.clear()

    def __repr__(self):
        out = '('
        for row in self.value:
            out += ' '.join(list(map(str, row))).join('()')
        return out + ')'

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            raise Exception('TypeError: Unable to equate a Matrix to a non-Matrix type')
        
        for i in range(self.rows):
            for j in range(self.columns):
                if self.get_value(i, j) != other.get_value(i, j):
                    return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __abs__(self):
        return self.det()

    def set_value(self, row, column, value):
        copy = self.value.copy()
        copy[row - 1][column - 1] = value
        return copy

    def set_row(self, row, values):
        values = list(values)
        copy = self.value.copy()
        copy[row - 1] = values
        return copy

    def set_column(self, column, values):
        values = list(values)
        column -= 1
        copy = []
        for row in self.value:
            row[column] = values.pop(0)
            copy.append(row.copy())
        return copy

    def get_value(self, row, column):
        return self.value[row - 1][column - 1]

    def get_row(self, row):
        return self.value[row - 1]

    def get_column(self, column):
        return [row[column - 1] for row in self.value]

    def add_next(self, value):
        self.set_value(self.next_row, self.next_column, value)
        self.next_column += 1
        if self.next_column > self.columns:
            self.next_column = 1
            self.next_row += 1

        if self.next_row > self.rows:
            self.next_row = 1

    def mul(self, other):
        if isinstance(other, (int, float, complex)):
            ret = Matrix(self.value.copy())
            for i in range(self.rows):
                for j in range(self.columns):
                    ret.set_value(i, j, other * self.get_value(i, j))
            return ret

        if not isinstance(other, Matrix):
            raise Exception('TypeError: Can only multiply a Matrix by either an integer or a matrix')
        
        if self.columns != other.rows:
            raise Exception('TypeError: Non-conformable dimensions {}x{} and {}x{}'.format(self.rows, self.columns, *other.dims))
        
        new = Matrix(self.rows, other.columns, zero = True)
        left = self.value.copy()
        right = other.transpose().value.copy()

        while left:
            row = left.pop(0)
            for rrow in right:
                ret = 0
                for l, r in zip(row, rrow):
                    ret += l * r
                new.add_next(ret)

        return Matrix(new.value.copy())

    def add(self, other):
        if self.columns != other.columns or self.rows != other.rows:
            raise Exception('TypeError: Non-conformable dimensions {}x{} and {}x{}'.format(self.rows, self.columns, *other.dims))
        
        ret = Matrix(*self.dims, zero = True)
        for i in range(self.rows):
            for j in range(self.columns):
                sum_ = self.get_value(i, j) + other.get_value(i, j)
                ret.set_value(i, j, sum_)
        return ret

    def sub(self, other):
        return self.add(other.neg())

    def pow_(self, other):
        ret = Matrix(self.value.copy())
        for i in range(other - 1):
            ret *= self
        return ret

    def neg(self):
        return self.map_op(lambda a: -a)

    def map_op(self, op):
        ret = Matrix(self.value.copy())
        for i in range(self.rows):
            for j in range(self.columns):
                ret.set_value(i, j, op(self.get_value(i, j)))
        return ret

    def det(self):
        if self.rows == 1 == self.columns:
            return self.value[0][0]

        if self.rows == self.columns:
            mat = self.value.copy()
            det = 0
            for i, row in enumerate(mat):
                copy = mat.copy()
                copy.pop(i)
                copy = Matrix([r[1:] for r in copy])
                det += row[0] * copy.det() * (-1) ** i
            return det

        other = Matrix(self.transpose())
        return Matrix(self.mul(other)).det() ** 0.5

    def minor(self, i, j):
        if self.rows == self.columns == 1:
            return self.value[0][0]
        copy = []
        for each in self.value.copy():
            copy.append(each.copy())
        copy.pop(i - 1)
        minor_mat = []
        for k, row in enumerate(copy):
            row.pop(j - 1)
            minor_mat.append(row)
        minor_mat = Matrix(minor_mat)
        return minor_mat.det()

    def cofactor(self, i, j):
        return self.minor(i, j) * (-1) ** (i + j)

    def adjugate(self):
        adj = Matrix(self.rows, self.columns)
        for i in range(1, self.rows + 1):
            for j in range(1, self.columns + 1):
                adj.set_value(i, j, self.cofactor(i, j))
        return adj.transpose()

    def inverse(self):
        det = self.det()
        adj = self.adjugate()
        for i in range(1, self.rows + 1):
            for j in range(1, self.columns + 1):
                adj.set_value(i, j, adj.get_value(i, j) / det)
        return adj

    def transpose(self):
        new = [[] for _ in range(self.columns)]
        for row in self.value:
            for index, elem in enumerate(row):
                new[index].append(elem)
        return Matrix(new)
    
    def reshape(self, rows, columns, array = None):
        new = Matrix(rows, columns, zero = True)
        if array is None:
            values = flatten(self.value)
        else:
            values = list(array)
            
        while values:
            new.add_next(values.pop(0))
        return new

    __mul__ = mul
    __add__ = add
    __sub__ = sub
    __pow__ = pow_

    __rmul__ = mul
    __radd__ = add
    __rsub__ = sub

class InfSeq:
    def __init__(self, infinite_function, ordered = True, uniques = None, repeated = False):
        self.inf = infinite_function
        self.order = ordered
        self.gen = self.inf()
        self.uniques = uniques
        self.repeat = repeated

    def __contains__(self, obj):
        if self.uniques:
            return obj in self.uniques
        if self.repeat:
            return obj in self.take(self.repeat)
        
        for elem in self.inf():

            if elem == obj:
                return True
            
            if elem > obj and self.order == 1:
                return False
            elif elem < obj and self.order == -1:
                return False

    def __iter__(self):
        return self.inf()

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.start is None:
                return self.take(index.stop)

            if index.step is None:
                ret = self.take(index.stop)
                for _ in range(index.start): ret.pop(0)
                return ret

            return self.take(index.stop)[index.start : index.stop : index.step]
            
        return self.take(index + 1)[-1]

    def __next__(self):
        return next(self.gen)

    def __repr__(self):
        return '[{}, {}, {}...]'.format(*self.take(3))

    __str__ = __repr__

    @property
    def elements(self):
        for elem in self.inf():
            print(elem, end = ' ')
        return ''

    def index(self, elem):
        index = 0
        if elem not in self:
            return -1

        for gen_elem in self.inf():
            if gen_elem == elem:
                return index
            index += 1

    def take(self, num):
        taken = []
        while len(taken) < num:
            taken.append(next(self.gen))
        self.gen = self.inf()
        return taken

class Quaternion:

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self):
        ret = '{}'.format(self.a)
        if self.b >= 0: ret += '+'
        ret += '{}i'.format(self.b)
        if self.c >= 0: ret += '+'
        ret += '{}j'.format(self.c)
        if self.d >= 0: ret += '+'
        ret += '{}k'.format(self.d)

        return ret

    def __iter__(self):
        return iter((self.a, self.b, self.c, self.d))

    def __neg__(self):
        return Quaternion(
            -self.a,
            -self.b,
            -self.c,
            -self.d
        )

    def __abs__(self):
        return (self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2) ** 0.5

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.a + other.a,
                self.b + other.b,
                self.c + other.c,
                self.d + other.d
            )

        if isinstance(other, (int, float)):
            return Quaternion(
                self.a + other,
                self.b,
                self.c,
                self.d
            )

        if isinstance(other, complex):
            return Quaternion(
                self.a + other.real,
                self.b + other.imag,
                self.c,
                delf.d
            )

        raise TypeError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.a - other.a,
                self.b - other.b,
                self.c - other.c,
                self.d - other.d
            )

        if isinstance(other, (int, float)):
            return Quaternion(
                self.a - other,
                self.b,
                self.c,
                self.d
            )

        if isinstance(other, complex):
            return Quaternion(
                self.a - other.real,
                self.b - other.imag,
                self.c,
                delf.d
            )

        raise TypeError

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            x, y, z, w = other
            p = Matrix([[self.a], [self.b], [self.c], [self.d]])
            q = Matrix(
                [
                    [ x, -y, -z, -w],
                    [ y,  x,  w, -z],
                    [ z, -w,  x,  y],
                    [ w,  z, -y,  x]
                ]
            )
            return Quaternion(*(q * p).transpose().value[0])

        if isinstance(other, complex):
            return self * Quaternion(other.real, other.imag, 0, 0)

        if isinstance(other, (int, float)):
            return Quaternion(
                self.a * other,
                self.b * other,
                self.c * other,
                self.d * other
            )

        raise TypeError

    def __rmul__(self, other):
        if isinstance(other, complex):
            return Quaternion(other.real, other.imag, 0, 0) * self

        if isinstance(other, (int, float)):
            return Quaternion(
                self.a * other,
                self.b * other,
                self.c * other,
                self.d * other
            )

        raise TypeError

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inverse

        if isinstance(other, complex):
            return self / Quaternion(other.real, other.imag, 0, 0)

        if isinstance(other, (int, float)):
            return Quaternion(
                self.a / other,
                self.b / other,
                self.c / other,
                self.d / other
            )

        raise TypeError

    def __rtruediv__(self, other):
        if isinstance(other, complex):
            return Quaternion(other.real, other.imag, 0, 0) / self

        if isinstance(other, (int, float)):
            return Quaternion(
                other / self.a,
                other / self.b,
                other / self.c,
                other / self.d
            )

        raise TypeError

    @property
    def inverse(self):
        return self.conjugate / abs(self)

    @property
    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    

















