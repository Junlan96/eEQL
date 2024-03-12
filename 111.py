


from numbers import Number
import sympy as sy
from sympy import symbols, sin, log, exp, Abs, cos


#Simplify the expression
def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

x=symbols('x')
y=symbols('y')
z=symbols('z')

X1=symbols('X1')
X2=symbols('X2')

x1=symbols('x1')
x2=symbols('x2')
x3=symbols('x3')
x4=symbols('x4')
x5=symbols('x5')
x6=symbols('x6')
x7=symbols('x7')
x8=symbols('x8')
x9=symbols('x9')




f=-4.49718912596563e-11*x**4 + 87.3745779321216*(-x**2 - 0.18064*x)**2 + 0.980349166680753

# f=4-0.019*sy.log(sy.Abs(-0.018*x**2 + 0.007*sy.log(0.007*sy.Abs(x)) + 0.006)) + 1.343*sy.log(sy.Abs(1.559*x**2 + 0.005*sy.log(0.007*sy.Abs(x)) + 1.085)) + 0.014

# f=22.854*x**2 + 2.106*x*(1 - 0.693*x)**2 + 4.557*x + 37.014*(-0.53*x**2 - 0.074*x*(1 - 0.693*x)**2 + 1)**2 + 0.639*(-0.61*x**2 - 0.611*x*(1 - 0.693*x)**2 + x - 0.003)**2 + 2.75*(0.533*x**2 - 0.003*x*(1 - 0.693*x)**2 - 0.002*x - 1)**2*(1.13*x**2 - 0.006*x*(1 - 0.693*x)**2 - 0.004*x - 3.121) - 28.362
# f=-0.885*sy.sin(-0.634*sy.sin(0.503*x) + 0.323*sy.sin(4.95*x) + 1.547) + 0.192*sy.cos(0.023*sy.sin(0.503*x) + 1.16*sy.cos(3.626*x) - 0.154*sy.cos(9.068*x) + 1.11) - 0.014


print(round_expr(sy.expand(f),0))   #表达式展开为最小项表达式
print(round_expr(sy.factor(f),3))   #表达式因式分解
print(round_expr(sy.together(f),3))  #表达式合并

# print(sy.apart(f))     #表达式拆分
print(round_expr(sy.sympify(f),2))   #表达式简化
print(round_expr(sy.trigsimp(f),2))




