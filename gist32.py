# -*- coding:utf8 -*-

def answer(m):
    n = len(m)
    if n<=1:
        return [0]
    for i in range(n):
        m[i][i] = 0
    for i in range(n):
        if len([item for item in m[i] if item != 0])== 1 and sum(m[i]) != 1:
            return [0]

    newl = []
    for i in range(n):
        if list(set(m[i])) == [0]:
            newl.append(i)
    gap = len(newl)-1 # gap都是absorbing matrix, 对角线都为1. gap = 3
    if gap < 0:
        return [0]
    for i in range(n):
        if i not in newl:
            newl.append(i) # newl = [2, 3, 4, 5, 0, 1]

    # 构造一个new matrix, 横纵坐标为[2, 3, 4, 5, 0, 1]，分为I块,O块，R和Q块一共四块
    matrix = [[0 for j in range(n)] for i in range(n)]
    # 第一步求F = (I-Q)^-1

    # Q
    length = n-(gap+1)
    Q = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            row = newl[gap+1+i]
            col = newl[gap+1+j]
            Q[i][j] = [m[row][col],sum(m[row])]# Q = [[[0, 2], [1, 2]], [[4, 9], [0, 9]]]

    # I
    I = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            tmp = Q[i][0][1]
            if i == j:
                I[i][j] = [tmp, tmp]
            else:
                I[i][j] = [0,tmp] # I = [[[2, 2], [0, 2]], [[0, 9], [9, 9]]]

    # I-Q
    IminusQ = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            IminusQ[i][j] = [I[i][j][0] - Q[i][j][0],I[i][j][1]] # IminusQ = [[[2, 2], [-1, 2]], [[-4, 9], [9, 9]]]

    # 求(I-Q)的逆矩阵F
    # 先转成整数，记录最小公倍数lcmvalue。再求逆矩阵，记录行列式的值A
    lcmvalue = IminusQ[0][0][1]
    for i in range(1, length):
        lcmvalue = lcm(lcmvalue, IminusQ[i][0][1]) # lcmvalue = 18
    for i in range(length):
        for j in range(length):
            IminusQ[i][j] = IminusQ[i][j][0] * (lcmvalue / IminusQ[i][j][1]) # IminusQ = [[18, -9], [-8, 18]]
    A = MatrixGetDet(IminusQ) # A = 252
    denominator = A / lcmvalue # 14
    F = [[0 for j in range(length)] for i in range(length)]
    # 求伴随矩阵，注意先删j行，再删i列
    for i in range(length):
        for j in range(length):
            copy = IminusQ[:j]+IminusQ[j+1:]
            for x in range(len(copy)):
                copy[x] = copy[x][:i]+copy[x][i+1:]
            F[i][j] = [MatrixGetDet(copy) * pow(-1,i+j), denominator] # 不用除以A，保证全部是整数。F=[[[18, 14], [9, 14]], [[8, 14], [18, 14]]]

    # 第二步求FR
    R = [[0 for j in range(gap+1)] for i in range(length)]
    for i in range(length):
        for j in range(gap+1):
            row = newl[gap + 1 + i]
            col = newl[j]
            R[i][j] = [m[row][col], sum(m[row])] # 2.R = [[[0, 2], [0, 2], [0, 2], [1, 2]], [[0, 9], [3, 9], [2, 9], [0, 9]]]
    F=F[0]
    for i in range(len(F)):
        tmp = hcf(F[i][0],F[i][1])
        F[i] = [F[i][0]/tmp,F[i][1]/tmp] # 1.F = [[9, 7], [9, 14]]
    FR = [0 for i in range(gap+1)]
    for j in range(gap+1):
        FR[j] = multiply(F[0], R[0][j])
        for i in range(1, length):
            tmp = multiply(F[i],R[i][j])
            if tmp[0] == 0:
                continue
            FR[j] = add(FR[j],tmp)  # FR=[[0, 0], [3, 14], [1, 7], [9, 14]]
    print FR
    if [FR[i][1] for i in range(len(FR)) if FR[i][0] != 0] != []:
        common = reduce((lambda x, y: lcm(x,y)), [FR[i][1] for i in range(len(FR)) if FR[i][0] != 0]) # common = 14
        new = []
        for i in range(len(FR)):
            if FR[i][0] == 0:
                new.append(0)
            else:
                new.append(FR[i][0]*(common/FR[i][1]))
        return new
    else:
        return [0]

def add(x,y):
    if x[0] == 0 and y[0] == y:
        return [0,0]
    if x[0] == 0:
        return y
    if y[0] == 0:
        return x
    den = lcm(x[1],y[1])
    num = x[0]*(den/x[1])+y[0]*(den/y[1])
    tmp = hcf(num,den)
    return [num/tmp,den/tmp]
def multiply(x,y):
    if x[0] == 0 or y[0] == 0:
        return [0,0]
    den = x[1]*y[1]
    num = x[0]*y[0]
    tmp = hcf(num,den)
    return [num/tmp,den/tmp]

# 最大公约数
def hcf(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    hcf = 1
    for i in range(1, smaller + 1):
        if ((x % i == 0) and (y % i == 0)):
            hcf = i
    return hcf

# 最小公倍数
def lcm(x, y):
    if x > y:
        greater = x
    else:
        greater = y
    lcm = x*y
    while (True):
        if ((greater % x == 0) and (greater % y == 0)):
            lcm = greater
            break
        greater += 1
    return lcm

# 求行列式的值
def MatrixGetDet(M):
    length = len(M)
    if length == 1:
        return M[0][0]
    if length == 2:
        return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    import itertools
    # sum positive
    positive = 0
    indexer1 = itertools.cycle(range(length))
    indexer2 = itertools.cycle(range(length))
    for index in range(length):
        indexer1.next()
        _tmp = 1
        for index in range(length):
            index1 = indexer1.next()
            index2 = indexer2.next()
            _tmp *= M[index1][index2]
        positive += _tmp
    # sum negative
    negative = 0
    indexer1 = itertools.cycle(range(length))
    indexer2 = itertools.cycle(range(length)[::-1])
    for index in range(length):
        indexer1.next()
        _tmp = 1
        for index in range(length):
            index1 = indexer1.next()
            index2 = indexer2.next()
            _tmp *= M[index1][index2]
        negative += _tmp
    # return
    return positive - negative


m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
print answer(m)
m= [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
print answer(m)
m= [[0, 1, 0, 0, 0, 1, 2, 0, 0, 0], [4, 0, 0, 3, 2, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 5, 3, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
print answer(m)
m= [[0, 1, 5], [0, 0, 0], [0, 0, 0]]
print answer(m)






from fractions import Fraction
from copy import deepcopy

def gcd(a, b):
    if a == 0 or b == 0:
        return 1
    if a < b:
        a, b = b, a
    while b:
        a, b = b, a % b
    return a


def lcm(numbers):
    res = 1
    for n in numbers:
        res = res * n / gcd(res, n)
    return res


class Matrix(list):
    def __init__(self, data):
        self.extend(data)

    def __sub__(self, other):
        rows = len(self)
        columns = len(self[0])
        m = []
        for r in xrange(rows):
            row = []
            for c in xrange(columns):
                row.append(self[r][c] - other[r][c])
            m.append(row)
        return Matrix(m)

    def multiply(self, other):
        x = len(self)
        n = len(self[0])
        y = len(other[0])
        m = []
        for i in xrange(x):
            r = [0] * y
            for j in xrange(y):
                for k in xrange(n):
                    r[j] += self[i][k] * other[k][j]
            m.append(r)
        return Matrix(m)

    def inverse(self):
        size = len(self)
        I = self.identity(size)

        a = deepcopy(self)
        for i in xrange(size):
            a[i] += I[i]

        # Upper triangular matrix
        for i in xrange(size):
            # First swap row i for a lower row if a[i][i] is zero
            if not a[i][i]:
                for j in xrange(i + 1, size):
                    if a[j][i]:
                        t = a[i]
                        a[i] = a[j]
                        a[j] = a[t]
                        break

            # If a[i][i] is not 1, make it so
            if a[i][i] != 1:
                c = 1 / a[i][i]
                for k in xrange(len(a[i])):
                    a[i][k] = a[i][k] * c

            # Use row operations to make zeros below
            for j in xrange(i + 1, size):
                if a[j][i]:
                    c = (0 - a[j][i]) / a[i][i]
                    for k in xrange(len(a[j])):
                        a[j][k] = a[j][k] + (c * a[i][k])


        for i in xrange(size - 1, -1, -1):
            for j in xrange(i - 1, -1, -1):
                if a[j][i]:
                    c = (0 - a[j][i]) / a[i][i]
                    for k in xrange(len(a[j])):
                        a[j][k] = a[j][k] + (c * a[i][k])

        inverse = [r[size:] for r in a]
        return Matrix(inverse)

    @classmethod
    def identity(cls, n):
        m = []
        for i in xrange(n):
            r = [Fraction(0, 1) for j in xrange(n)]
            r[i] = Fraction(1, 1)
            m.append(r)
        return cls(m)

class Row(list):
    def __init__(self, state_number, data):
        super(Row, self).__init__()
        self.extend(data)
        self.state_number = state_number
        self.absorbing = not any(data)
        if self.absorbing:
            self[self.state_number] = 1

    def __repr__(self):
        return "s{}:\t {} -- absorbing: {}".format(self.state_number, super(Row, self).__repr__(), self.absorbing)

    def fractionalize(self):
        s = sum(self)
        for i, v in enumerate(self):
            self[i] = Fraction(v, s)


class P:
    def __init__(self, data):
        transition_states = []
        absorbing_states = []
        for i, row in enumerate(data):
            r = Row(i, row)
            if r.absorbing:
                absorbing_states.append(r)
            else:
                transition_states.append(r)
        self.t = len(transition_states)
        self.r = len(absorbing_states)
        self.rows = []
        self.rows.extend(transition_states)
        self.rows.extend(absorbing_states)

        self._reorder()

        self._fractionalize()

    def _fractionalize(self):
        for r in self.rows:
            r.fractionalize()

    def swap_columns(self, c1, c2):
        for r in self.rows:
            r[c1] = r[c1] ^ r[c2]
            r[c2] = r[c2] ^ r[c1]
            r[c1] = r[c1] ^ r[c2]

    def _reorder(self):
        swapped = set()
        schedule = [r.state_number for r in self.rows]

        for r in self.rows:
            newrow = [r[s] for s in schedule]
            for i in xrange(len(r)):
                r.pop()
            r.extend(newrow)

    def __repr__(self):
        return "\n".join([r.__repr__() for r in self.rows])

    def Q(self):
        q = []
        for i in xrange(self.t):
            q.append(self.rows[i][:self.t])
        return Matrix(q)

    def R(self):
        r = []
        for i in xrange(self.t):
            r.append(self.rows[i][self.t:])
        return Matrix(r)

    def N(self):
        Q = self.Q()
        i_minus_q = Matrix.identity(len(Q)) - Q
        N = i_minus_q.inverse()
        return N

    def NR(self):
        N = self.N()
        R = self.R()
        NR = N.multiply(R)
        return NR

def answer(data):
    if len(data) == 1:
        return [1, 1]

    p = P(data)
    nr = p.NR()
    denoms = [f.denominator for f in nr[0]]
    denom = lcm(denoms)
    nums = []
    for f in nr[0]:
        c = denom / f.denominator
        nums.append(f.numerator * c)
    return nums + [denom]