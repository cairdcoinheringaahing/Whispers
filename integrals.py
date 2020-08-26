import math
import sys

import derivatives
import mathparser
from mathparser import BinaryOp, UnaryOp, Variable, Constant
import simplify

sys.setrecursionlimit(1 << 30)

standard_integrals = {

    'sin' : mathparser.parse('-cos(x)'),
    'cos' : mathparser.parse('sin(x)'),
    'tan' : mathparser.parse('ln(sec(x))'),
    'csc' : mathparser.parse('ln(csc(x)-cot(x))'),
    'sec' : mathparser.parse('ln(sec(x)+tan(x))'),
    'cot' : mathparser.parse('ln(sin(x))'),

    'sinh' : mathparser.parse('cosh(x)'),
    'cosh' : mathparser.parse('sinh(x)'),
    'tanh' : mathparser.parse('ln(cosh(x))'),
    'csch' : mathparser.parse('ln(tanh(x÷2))'),
    'sech' : mathparser.parse('arctan(sinh(x))'),
    'coth' : mathparser.parse('ln(sinh(x))'),

    'arcsin' : mathparser.parse('x⋅arcsin(x)+√(1-x²)'),
    'arccos' : mathparser.parse('x⋅arccos(x)-√(1-x²)'),
    'arctan' : mathparser.parse('x⋅arctan(x)-ln(1-x²)÷2'),
    'arccsc' : mathparser.parse('x⋅arccsc(x)+arcosh(x)'),
    'arcsec' : mathparser.parse('x⋅arcsec(x)-arcosh(x)'),
    'arccot' : mathparser.parse('x⋅arccot(x)+ln(1-x²)÷2'),

    'arsinh' : mathparser.parse('x⋅arsinh(x)-√(x²+1)'),
    'arcosh' : mathparser.parse('x⋅arcosh(x)-√(x²-1)'),
    'artanh' : mathparser.parse('x⋅artanh(x)+ln(1-x²)÷2'),
    'arcsch' : mathparser.parse('x⋅arcsch(x)+arsinh(x)'),
    'arsech' : mathparser.parse('x⋅arsech(x)+arsinh(x)'),
    'arcoth' : mathparser.parse('x⋅arcoth(x)+ln(x²-1)÷2'),

    'exp' : mathparser.parse('exp(x)'),
    'ln'  : mathparser.parse('x⋅ln(x)-x'),

    '√' : mathparser.parse('(2x⋅√x)÷3'),
    '∛' : mathparser.parse('(3x⋅∛x)÷4'),

}

def is_axb(expr, var):
    if not isinstance(expr, BinaryOp):
        return is_x(expr, var)

    if expr.op in '+-':
        # b±ax
        # b±x÷a
        if (
            isinstance(expr.left, Constant)     and \
            isinstance(expr.right, BinaryOp)    and \
            expr.right.op in '⋅÷'               and \
            is_ax(expr.right, var)
        ):
            return True

        # ax±b
        # x÷a±b
        if (
            isinstance(expr.right, Constant)    and \
            isinstance(expr.left, BinaryOp)     and \
            expr.left.op in '⋅÷'                and \
            is_ax(expr.left, var)
        ):
            return True

        # b±x
        if (
            isinstance(expr.left, Constant)    and \
            is_x(expr.right, var)
        ):
            return True

        # x±b
        if (
            isinstance(expr.right, Constant)    and \
            is_x(expr.left, var)
        ):
            return True

    return is_ax(expr, var)

def is_ax(expr, var):
    if not isinstance(expr, BinaryOp):
        return is_x(expr, var)

    if expr.op == '⋅':
        # ax
        if (
            isinstance(expr.left, Constant)     and \
            is_x(expr.right, var)
        ):
            return True

    if expr.op == '÷':
        # x÷a
        if (
            isinstance(expr.right, Constant)     and \
            is_x(expr.left, var)
        ):
            return True

    return is_x(expr, var)

def is_x(expr, var):
    return isinstance(expr, Variable) and expr.name == var

def is_trig(expr):
    if not isinstance(expr, UnaryOp):
        return False

    return expr.op in ('sin', 'cos', 'tan', 'sec', 'csc', 'cot')

def is_power(expr):
    return (
        isinstance(expr, UnaryOp)   and \
        expr.pos == 'Postfix'       and \
        derivatives.REGEX['exponent'].search(expr.op)
    ) or (
        isinstance(expr, BinaryOp)  and \
        expr.op == '*'              and \
        isinstance(expr.right, Constant)
    )

def nest(branch, fruit, var):
    new = branch.copy()
    
    if isinstance(branch, UnaryOp):
        new.operand = nest(branch.operand, fruit, var)

    if isinstance(branch, BinaryOp):
        new.left = nest(branch.left, fruit, var)
        new.right = nest(branch.right, fruit, var)

    if is_x(branch, var):
        new = fruit.copy()

    return new

def reduce_trig(f, n, variable, limit):
    # Trig integration by reduction formula
    if n == 0:
        return Variable(variable)
    if n == 1:
        return antiderivative(f, variable, limit)

    if f.op == 'sin':
        # ∫sin(x)ⁿ dx = (-sinⁿ⁻¹(x)⋅cos(x) + (n-1)∫sin(x)ⁿ⁻² dx) ÷ n

        Ifunc = BinaryOp(
            left = BinaryOp(
                left = UnaryOp(
                    'Prefix',
                    op = '-',
                    operand = BinaryOp(
                        left = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'sin',
                                operand = Variable(variable)
                            ),
                            op = '*',
                            right = Constant(n-1)
                        ),
                        op = '⋅',
                        right = UnaryOp(
                            'Prefix',
                            op = 'cos',
                            operand = Variable(variable)
                        )
                    )
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(n-1),
                    op = '⋅',
                    right = antiderivative(
                        BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'sin',
                                operand = Variable(variable)
                            ),
                            op = '*',
                            right = Constant(n-2)
                        ),
                        variable,
                        limit
                    )
                )
            ),
            op = '÷',
            right = Constant(n)
        )

    if f.op == 'cos':
    # ∫cos(x)ⁿ dx = (sin(x)⋅cosⁿ⁻¹(x) + (n-1)∫cos(x)ⁿ⁻² dx) ÷ n

        Ifunc = BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = 'cos',
                            operand = Variable(variable)
                        ),
                        op = '*',
                        right = Constant(n-1)
                    ),
                    op = '⋅',
                    right = UnaryOp(
                        'Prefix',
                        op = 'sin',
                        operand = Variable(variable)
                    )
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(n-1),
                    op = '⋅',
                    right = antiderivative(
                        BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'cos',
                                operand = Variable(variable)
                            ),
                            op = '*',
                            right = Constant(n-2)
                        ),
                        variable,
                        limit
                    )
                )
            ),
            op = '÷',
            right = Constant(n)
        )

    if f.op == 'tan':
    # ∫tan(x)ⁿ dx = tanⁿ⁻¹(x) ÷ (n-1) - ∫tan(x)ⁿ⁻² dx

        Ifunc = BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'tan',
                        operand = Variable(variable)
                    ),
                    op = '*',
                    right = Constant(n-1)
                ),
                op = '÷',
                right = Constant(n-1)
            ),
            op = '-',
            right = antiderivative(
                BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'tan',
                        operand = Variable(variable)
                    ),
                    op = '*',
                    right = Constant(n-2)
                ),
                variable,
                limit
            )
        )

    if f.op == 'sec':
    # ∫sec(x)ⁿ dx = (sin(x)⋅secⁿ⁻¹(x) + (n-2)∫sec(x)ⁿ⁻² dx) ÷ (n-1)

        Ifunc = BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = 'sec',
                            operand = Variable(variable)
                        ),
                        op = '*',
                        right = Constant(n-1)
                    ),
                    op = '⋅',
                    right = UnaryOp(
                        'Prefix',
                        op = 'sin',
                        operand = Variable(variable)
                    )
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(n-2),
                    op = '⋅',
                    right = antiderivative(
                        BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'sec',
                                operand = Variable(variable)
                            ),
                            op = '*',
                            right = Constant(n-2)
                        ),
                        variable,
                        limit
                    )
                )
            ),
            op = '÷',
            right = Constant(n-1)
        )

    if f.op == 'csc':
    # ∫csc(x)ⁿ dx = (cos(x)⋅cscⁿ⁻¹(x) + (n-2)∫csc(x)ⁿ⁻² dx) ÷ (n-1)

        Ifunc = BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = 'csc',
                            operand = Variable(variable)
                        ),
                        op = '*',
                        right = Constant(n-1)
                    ),
                    op = '⋅',
                    right = UnaryOp(
                        'Prefix',
                        op = 'cos',
                        operand = Variable(variable)
                    )
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(n-2),
                    op = '⋅',
                    right = antiderivative(
                        BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'csc',
                                operand = Variable(variable)
                            ),
                            op = '*',
                            right = Constant(n-2)
                        ),
                        variable,
                        limit
                    )
                )
            ),
            op = '÷',
            right = Constant(n-1)
        )

    if f.op == 'cot':
    # ∫cot(x)ⁿ dx = cotⁿ⁻¹(x) ÷ (n-1) - ∫cot(x)ⁿ⁻² dx

        Ifunc = BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'cot',
                        operand = Variable(variable)
                    ),
                    op = '*',
                    right = Constant(n-1)
                ),
                op = '÷',
                right = Constant(n-1)
            ),
            op = '-',
            right = antiderivative(
                BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'cot',
                        operand = Variable(variable)
                    ),
                    op = '*',
                    right = Constant(n-2)
                ),
                variable,
                limit
            )
        )

    return Ifunc

def reduce_trig_prod(f, g, m, n, a, b, variable, limit):
    # ∫f(ax+b)ᵐ⋅g(ax+b)ⁿ dx

    if m == 0:
        return reduce_trig(g, n, variable, limit)
    if n == 0:
        return reduce_trig(f, m, variable, limit)
    if m == 1:
        return antiderivative(
            BinaryOp(
                left = f,
                op = '⋅',
                right = BinaryOp(
                    left = g,
                    op = '*',
                    right = n
                )
            ),
            variable,
            limit
        )
    if n == 1:
        return antiderivative(
            BinaryOp(
                left = g,
                op = '⋅',
                right = BinaryOp(
                    left = f,
                    op = '*',
                    right = m
                )
            ),
            variable,
            limit
        )
    
    # ₋₂ ⁺⁻¹

    axb = mathparser.parse('{}{}+{}'.format(a, variable, b))

    if (f.op == 'sin' and g.op == 'cos') or (f.op == 'csc' and g.op == 'cot'):
        # csc(ax+b)ᵐ⋅cot(ax+b)ⁿ = sin(ax+b)ᵐ⁺ⁿ⋅cos(ax+b)ⁿ
        #
        # ∫sin(ax+b)ᵐ⋅cos(ax+b)ⁿ dx
        #   = (-sinᵐ⁻¹(ax+b)⋅cosⁿ⁺¹(ax+b) ÷ a + (m-1)⋅∫sin(ax+b)ᵐ⁻²⋅cos(ax+b)ⁿ dx) ÷ (m+n)

        if f.op == 'csc' and g.op == 'cot':
            m += n

        I = antiderivative(
            BinaryOp(
                left = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'sin',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(m-2)
                ),
                op = '⋅',
                right = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'cos',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(n)
                )
            ),
            variable,
            limit
        )
                        
        return BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = '-',
                            operand = BinaryOp(
                                left = UnaryOp(
                                    'Prefix',
                                    op = 'sin',
                                    operand = axb
                                ),
                                op = '*',
                                right = Constant(m-1)
                            )
                        ),
                        op = '⋅',
                        right = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'cos',
                                operand = axb
                            ),
                            op = '*',
                            right = Constant(n+1)
                        )
                    ),
                    op = '÷',
                    right = Constant(a)
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(m-1),
                    op = '⋅',
                    right = I
                )
            ),
            op = '÷',
            right = Constant(m+n)
        )

    if (f.op == 'sin' and g.op in ('sec', 'tan')) or (f.op == 'tan' and g.op == 'sec'):
        # sin(ax+b)ᵐ⋅tan(ax+b)ⁿ = sin(ax+b)ᵐ⁺ⁿ⋅sec(ax+b)ⁿ
        # tan(ax+b)ᵐ⋅sec(ax+b)ⁿ = sin(ax+b)ᵐ⋅sec(ax+b)ᵐ⁺ⁿ
        #
        # ∫sin(ax+b)ᵐ⋅sec(ax+b)ⁿ dx
        #   = ∫sin(ax+b)ᵐ÷cos(ax+b)ⁿ dx
        #   = (-sinᵐ⁻¹(ax+b)⋅secⁿ⁻¹(ax+b) ÷ a + (m-1)⋅∫sin(ax+b)ᵐ⁻²⋅sec(ax+b)ⁿ dx) ÷ (m-n)

        if g.op == 'tan':
            m += n
        if f.op == 'tan':
            n += m

        I = antiderivative(
            BinaryOp(
                left = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'sin',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(m-2)
                ),
                op = '⋅',
                right = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'sec',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(n)
                )
            ),
            variable,
            limit
        )
                        
        return BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = '-',
                            operand = BinaryOp(
                                left = UnaryOp(
                                    'Prefix',
                                    op = 'sin',
                                    operand = axb
                                ),
                                op = '*',
                                right = Constant(m-1)
                            )
                        ),
                        op = '⋅',
                        right = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'sec',
                                operand = axb
                            ),
                            op = '*',
                            right = Constant(n-1)
                        )
                    ),
                    op = '÷',
                    right = Constant(a)
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(m-1),
                    op = '⋅',
                    right = I
                )
            ),
            op = '÷',
            right = Constant(m-n)
        )

    if f.op in ('csc', 'cot') and g.op == 'cos':
        # cot(ax+b)ᵐ⋅cos(ax+b)ⁿ = csc(ax+b)ᵐ⋅cos(ax+b)ᵐ⁺ⁿ
        #
        # ∫csc(ax+b)ᵐ⋅cos(ax+b)ⁿ dx
        #   = ∫cos(ax+b)ⁿ÷sin(ax+b)ᵐ dx
        #   = (cosⁿ⁻¹(ax+b)⋅cscᵐ⁻¹(ax+b) ÷ a + (n-1)⋅∫csc(ax+b)ᵐ⋅cos(ax+b)ⁿ⁻² dx) ÷ (n-m)

        if f.op == 'cot':
            n += m

        I = antiderivative(
            BinaryOp(
                left = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'csc',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(m)
                ),
                op = '⋅',
                right = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'cos',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(n-2)
                )
            ),
            variable,
            limit
        )

        return BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'cos',
                                operand = axb
                            ),
                            op = '*',
                            right = Constant(n-1)
                        ),
                        op = '⋅',
                        right = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'csc',
                                operand = axb
                            ),
                            op = '*',
                            right = Constant(m-1)
                        )
                    ),
                    op = '÷',
                    right = Constant(a)
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(n-1),
                    op = '⋅',
                    right = I
                )
            ),
            op = '÷',
            right = Constant(n-m)
        )

    if f.op == 'csc' and g.op == 'sec':
        # ∫csc(ax+b)ᵐ⋅sec(ax+b)ⁿ dx
        #   = ∫1÷(sin(ax+b)ᵐ⋅cos(ax+b)ⁿ) dx
        #   = (-csc(ax+b)ᵐ⁻¹⋅sec(ax+b)ⁿ⁻¹ ÷ a + (m+n-2)⋅∫csc(ax+b)ᵐ⁻²⋅sec(ax+b)ⁿ dx) ÷ (m-1)

        I = antiderivative(
            BinaryOp(
                left = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'csc',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(m-2)
                ),
                op = '⋅',
                right = BinaryOp(
                    left = UnaryOp(
                        'Prefix',
                        op = 'sec',
                        operand = axb
                    ),
                    op = '*',
                    right = Constant(n)
                )
            ),
            variable,
            limit
        )
                        
        return BinaryOp(
            left = BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = '-',
                            operand = BinaryOp(
                                left = UnaryOp(
                                    'Prefix',
                                    op = 'csc',
                                    operand = axb
                                ),
                                op = '*',
                                right = Constant(m-1)
                            )
                        ),
                        op = '⋅',
                        right = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'sec',
                                operand = axb
                            ),
                            op = '*',
                            right = Constant(n-1)
                        )
                    ),
                    op = '÷',
                    right = Constant(a)
                ),
                op = '+',
                right = BinaryOp(
                    left = Constant(m+n-2),
                    op = '⋅',
                    right = I
                )
            ),
            op = '÷',
            right = Constant(m-1)
        )

    return reduce_trig_prod(g, f, n, m, a, b, variable, limit)                        

def is_const_prod(left, right):
    if not(isinstance(left, BinaryOp) or isinstance(right, BinaryOp)):
        return left == right

    if isinstance(left, BinaryOp):
        left, right = right, left

    if not isinstance(right, BinaryOp):
        return False
    if right.op != '⋅':
        return False
    if not isinstance(right.left, Constant):
        return False

    if left == right.right:
        return True

    return is_const_prod(left, right.right)

def inspect(func, f, dg, var):
    # ∫f'(g(x))g'(x) dx = f(g(x))

    if not isinstance(f, UnaryOp):
        return False
    
    g = f.operand
    return is_const_prod(dg, derivatives.main(g))

def antiderivative(func, variable = 'x', limit = 1 << 12):
    
    def parts(u, dv):
        
        du = derivatives.main(u)
        v = antiderivative(dv, variable, limit)

        ddu = derivatives.main(du)
        ddv = derivatives.main(dv)

        if func in (BinaryOp(left = ddu, op = '⋅', right = dv), BinaryOp(left = dv, op = '⋅', right = ddu)):
            return BinaryOp(
                left = BinaryOp(
                    left = BinaryOp(
                        left = u,
                        op = '⋅',
                        right = v
                    ),
                    op = '-',
                    right = BinaryOp(
                        left = du,
                        op = '⋅',
                        right = dv
                    )
                ),
                op = '÷',
                right = Constant(2)
            )
                    
        vdu = simplify.simplify(
            antiderivative(
                BinaryOp(
                    left = du,
                    op = '⋅',
                    right = v
                ),
                variable,
                limit
            )
        )

        return BinaryOp(
            left = BinaryOp(
                left = u,
                op = '⋅',
                right = v
            ),
            op = '-',
            right = vdu
        )

    if not limit:
        raise Exception('Unable to find antiderivative in time')
    limit -= 1

    if isinstance(func, str):
        if func in standard_integrals:
            return standard_integrals[func]
        
        func = mathparser.parse(func)
        
    if func in derivatives.standard_derivatives.values():
        for key, value in derivatives.standard_derivatives.items():
            if value == func:
                return mathparser.parse(key + '({})'.format(variable))
            
    if isinstance(func, Constant):
        Ifunc = mathparser.parse('{}{}'.format(func.value, variable))

    if isinstance(func, Variable):
        if is_x(func, variable):
            Ifunc = mathparser.parse('{}²÷2'.format(variable))
        else:
            Ifunc = mathparser.parse('{}{}'.format(func.name, variable))

    if isinstance(func, UnaryOp):

        if func.pos == 'Prefix':

            if func.op in standard_integrals and is_x(func.operand, variable):
                return nest(standard_integrals[func.op], Variable(variable), 'x')
            
            if is_axb(func.operand, variable):
                if func.op in '√∛':
                    # Power rule with fractional powers
                    # ∫(ax+b)ⁿ dx = ∫uⁿ du÷a, u = ax+b, du = a dx
                    #             = u*(n+1)÷(a(n+1))
                    #             = (ax+b)*(n+1) ÷ (a(n+1))

                    f = func.operand
                    n = 1 / ('√∛'.index(func.op) + 2)
                    a = derivatives.main(f).value

                    Ifunc = BinaryOp(
                        left = BinaryOp(
                            left = f,
                            op = '*',
                            right = Constant(n+1)
                        ),
                        op = '÷',
                        right = Constant(a*(n+1))
                    )

                else:
                    # Basic u-substitution
                    # ∫f(ax+b) dx = ∫f(u) du÷a, u = ax+b, du = a dx
                    
                    fu = UnaryOp('Prefix', op = func.op, operand = Variable('u'))
                    Ifu = nest(antiderivative(fu, 'u', limit), Variable(variable), 'u')
                    a = derivatives.main(func.operand)

                    Ifunc = BinaryOp(
                        left = nest(Ifu, func.operand, variable),
                        op = '÷',
                        right = a
                    )

            if func.op == '-':
                return UnaryOp('Prefix', op = '-', operand = antiderivative(func.operand))

        if func.pos == 'Postfix':

            if derivatives.REGEX['exponent'].search(func.op):
                n = mathparser.normalise(func.op)
                f = func.operand

                if is_axb(f, variable):
                    # Power rule
                    # ∫(ax+b)ⁿ dx = ∫uⁿ du÷a, u = ax+b, du = a dx
                    #             = u*(n+1)÷(a(n+1))
                    #             = (ax+b)*(n+1) ÷ (a(n+1))

                    a = derivatives.main(f).value
                    
                    Ifunc = BinaryOp(
                        left = BinaryOp(
                            left = f,
                            op = '*',
                            right = Constant(n+1)
                        ),
                        op = '÷',
                        right = Constant(a*(n+1))
                    )

                else:
                    # Generalised power rule
                    # ∫f(x)ⁿ dx
                    
                    if is_trig(f):
                        # Integration by reduction formula

                        if is_x(f.operand, variable):
                            Ifunc = reduce_trig(f, n, variable, limit)
                            
                        elif is_axb(f.operand, variable):
                            fu = UnaryOp('Prefix', op = f.op, operand = Variable('u'))
                            Ifu = nest(antiderivative(fu, 'u', limit), Variable(variable), 'u')
                            a = derivatives.main(func.operand)

                            Ifunc = BinaryOp(
                                left = nest(Ifu, func.operand, variable),
                                op = '÷',
                                right = a
                            )
                            

    if isinstance(func, BinaryOp):
        f = func.left
        op = func.op
        g = func.right
            
        df = derivatives.main(f)
        dg = derivatives.main(g)

        if op in '+-':
            # Addition rule
            # ∫f(x)±g(x) dx = ∫f(x) dx ± ∫g(x) dx

            If = antiderivative(f, variable, limit = limit)
            Ig = antiderivative(g, variable, limit = limit)

            Ifunc = BinaryOp(
                left = If,
                op = op,
                right = Ig
            )

        if op == '⋅':
            
            if isinstance(f, Constant):
                # Constant product rule
                # ∫af(x) dx = a∫f(x) dx
                
                Ig = antiderivative(g, variable, limit = limit)

                Ifunc = BinaryOp(
                    left = f,
                    op = '⋅',
                    right = Ig
                )

            elif inspect(func, f, g, variable) or inspect(func, g, f, variable):
                # Integration by inspection
                # ∫f'(g(x))g'(x) dx = f(g(x))

                if inspect(func, g, f, variable):
                    df, dg = g, f
                else:
                    df, dg = f, g

                f = antiderivative(
                    UnaryOp(
                        df.pos,
                        op = df.op,
                        operand = Variable('u')
                    ),
                    'u', limit)
                
                g = df.operand
                d_g = derivatives.main(g)

                consts = []

                while dg != d_g and is_const_prod(dg, d_g):
                    if isinstance(d_g.left, Constant):
                        consts.append(d_g.left.value)
                        d_g = d_g.right

                prod = 1
                for v in consts:
                    prod *= v
                prod = Constant(1/prod)

                Ifunc = BinaryOp(
                    left = prod,
                    op = '⋅',
                    right = nest(f, g, 'u')
                )

            elif is_power(f) and is_power(g):
                if is_trig(f.operand) and is_trig(g.operand):
                    m = mathparser.normalise(f.op)
                    n = mathparser.normalise(g.op)
                    f = f.operand
                    g = g.operand
                    
                if is_trig(f.left) and is_trig(g.left):
                    m = f.right.value
                    n = g.right.value
                    f = f.left
                    g = g.left
                    
                # Product of exponentiated trig
                # ∫f(x)ᵐ⋅g(x)ⁿ dx

                if is_x(f.operand, variable) and is_x(g.operand, variable):
                    Ifunc = reduce_trig_prod(f, g, m, n, 1, 0, variable, limit)

                elif is_ax(f.operand, variable) and is_ax(g.operand, variable) and f.operand == g.operand:
                    a = f.operand.left.value
                    Ifunc = reduce_trig_prod(f, g, m, n, a, 0, variable, limit)

                elif is_axb(f.operand, variable) and is_axb(g.operand, variable) and f.operand == g.operand:
                    if isinstance(f.operand.right, Constant):
                        a = f.operand.left.left.value
                        b = f.operand.right.value
                    else:
                        a = f.operand.right.left.value
                        b = f.operand.left.value
                    Ifunc = reduce_trig_prod(f, g, m, n, a, b, variable, limit)

            else:
                # Integration by parts
                # ∫u dv = uv - ∫v du
                # ∫f(x)g(x) dx = f(x)∫g(x) dx - ∫(∫g(x) dx)f'(x) dx

                try:
                    Ifunc = parts(f, g)
                except:
                    Ifunc = parts(g, f)

        if op == '*':

            if is_axb(f, variable) and isinstance(g, Constant):
                if g.value != -1:
                    # Power rule
                    # ∫(ax+b)ⁿ dx = ∫uⁿ du÷a, u = ax+b, du = a dx
                    #             = u*(n+1)÷(a(n+1))
                    #             = (ax+b)*(n+1) ÷ (a(n+1))
                    
                    a = df.value
                    n = g.value
                    
                    Ifunc = BinaryOp(
                        left = BinaryOp(
                            left = f,
                            op = '*',
                            right = Constant(n+1)
                        ),
                        op = '÷',
                        right = Constant(a*(n+1))
                    )

                if g.value == -1:
                    # Log rule
                    # ∫(ax+b)⁻¹ dx = ∫u⁻¹ du÷a, u = ax+b, du = a dx
                    #              = ln(u)÷a
                    #              = ln(ax+b)÷a
                    
                    a = df
                    
                    Ifunc = BinaryOp(
                        left = UnaryOp(
                            'Prefix',
                            op = 'ln',
                            operand = f
                        ),
                        op = '÷',
                        right = a
                    )

            if isinstance(f, Constant) and is_axb(g, variable):
                # Exponential rule
                # ∫c*(ax+b) dx = c*b∫c*(ax) dx
                #              = (c*(ax+b)) ÷ (aln(c))
                
                a = dg
                c = f
                
                Ifunc = BinaryOp(
                    left = func,
                    op = '÷',
                    right = BinaryOp(
                        left = a,
                        op = '⋅',
                        right = UnaryOp(
                            'Prefix',
                            op = 'ln',
                            operand = c
                        )
                    )
                )

            if is_trig(f) and isinstance(g, Constant):
                # Integration by reduction formula

                n = g.value

                if is_x(f.operand, variable):
                    Ifunc = reduce_trig(f, n, variable, limit)
                    
                elif is_axb(f.operand, variable):
                    fu = UnaryOp('Prefix', op = f.op, operand = Variable('u'))
                    Ifu = nest(antiderivative(fu, 'u', limit), Variable(variable), 'u')
                    a = derivatives.main(func.operand)

                    Ifunc = BinaryOp(
                        left = nest(Ifu, func.operand, variable),
                        op = '÷',
                        right = a
                    )

        if op == '÷':

            if isinstance(simplify.simplify(f), Constant):
                f = simplify.simplify(f)

            if isinstance(simplify.simplify(g), Constant):
                g = simplify.simplify(g)

            if isinstance(f, Constant):
                if (
                    isinstance(g, UnaryOp)  and \
                    g.pos == 'Postfix'      and \
                    derivatives.REGEX['exponent'].search(g.op)
                ) or (
                    isinstance(g, UnaryOp)  and \
                    g.pos == 'Prefix'       and \
                    g.op in '√∛'
                ):
                    # ∫c÷f(x)ⁿ dx = ∫cf(x)⁻ⁿ
                    
                    if g.pos == 'Postfix':
                        n = mathparser.normalise(g.op)
                    else:
                        n = 1 / ('√∛'.index(g.op) + 2)
                        
                    if is_axb(g.operand, variable):
                        # ∫c÷(ax+b)ⁿ dx
                        
                        if n != -1:
                            # Power rule with negative power
                            # ∫c÷(ax+b)ⁿ dx = c∫(ax+b)⁻ⁿ dx
                            #               = c(ax+b)*(1-n) ÷ (a(1-n))
                            
                            a = derivatives.main(g.operand).value
                            
                            Ifunc = BinaryOp(
                                left = f,
                                op = '⋅',
                                right = BinaryOp(
                                    left = BinaryOp(
                                        left = g.operand,
                                        op = '*',
                                        right = Constant(1-n)
                                    ),
                                    op = '÷',
                                    right = Constant(a*(1-n))
                                )
                            )

                        if n == -1:
                            # Log rule
                            # ∫c÷(ax+b) dx = c∫1÷(ax+b) dx
                            #              = cln(ax+b)÷a
                            
                            a = derivatives.main(g.operand)

                            Ifunc = BinaryOp(
                                left = f,
                                op = '⋅',
                                right = BinaryOp(
                                    left = UnaryOp(
                                        'Prefix',
                                        op = 'ln',
                                        operand = g.operand
                                    ),
                                    op = '÷',
                                    right = a
                                )
                            )

                if is_axb(g, variable):
                    # ∫c÷(ax+b) dx = c∫1÷(ax+b) dx
                    #              = cln(ax+b)÷a
                    
                    a = dg

                    Ifunc = BinaryOp(
                        left = f,
                        op = '⋅',
                        right = BinaryOp(
                            left = UnaryOp(
                                'Prefix',
                                op = 'ln',
                                operand = g
                            ),
                            op = '÷',
                            right = a
                        )
                    )

            if isinstance(g, Constant):
                # ∫f(x)÷a dx = ∫f(x) dx÷a

                Ifunc = BinaryOp(
                    left = simplify.simplify(antiderivative(
                        f,
                        variable,
                        limit
                    )),
                    op = '÷',
                    right = g
                )

            if is_poly(f, variable) and is_poly(g, variable):
                print('>', f, g)

    try:
        return Ifunc
    except:
        raise Exception('Unable to find antiderivative')

def simpson(f, a, b, n = 1 << 16):
    
    delta = (b - a) / n
    total = 0
    
    for i in range(n+1):
        x_i = a + i * delta
        
        if i in (0, n):  m = 1
        elif i % 2 == 1: m = 4
        elif i % 2 == 0: m = 2

        f_x = nest(f, Constant(x_i), 'x')
        total += simplify.evaluate(f_x) * m

    return total * (delta / 3)

def integrate(expr, a, b):
    # ∫ₐᵇ f(x) dx = [F(x)]ₐᵇ

    try:
        anti = antiderivative(expr)
    except:
        return simpson(mathparser.parse(expr), a, b)

    left  = nest(anti, Constant(b), 'x')
    right = nest(anti, Constant(a), 'x')

    return simplify.evaluate(left) - simplify.evaluate(right)
    
def main(expr, *limits):
    if limits:
        upper, lower = limits
        return integrate(expr, upper, lower)
    
    else:
        return simplify.main(antiderivative(expr))















