from __future__ import division
from itertools import compress
from itertools import starmap
from operator import mul
import fractions


def convertMatrix(transMatrix):
    probMatrix = []
    for i in range(len(transMatrix)):
        row = transMatrix[i]
        newRow = []
        rowSum = sum(transMatrix[i])
        if all([v == 0 for v in transMatrix[i]]):
            for j in transMatrix[i]:
                newRow.append(0)
            newRow[i] = 1
            probMatrix.append(newRow)
        else:
            for j in transMatrix[i]:
                if j == 0:
                    newRow.append(0)
                else:
                    newRow.append(j / rowSum)
            probMatrix.append(newRow)
    return probMatrix


def answer(m):
    # convert matrix numbers into probabilities
    probMatrix = convertMatrix(m)

    # find terminal states
    terminalStateFilter = []
    for row in range(len(m)):
        if all(x == 0 for x in m[row]):
            terminalStateFilter.append(True)
        else:
            terminalStateFilter.append(False)

    # multiply matrix by probability vector
    oldFirstRow = probMatrix[0]
    probVector = None
    for i in range(3000):
        probVector = [sum(starmap(mul, zip(oldFirstRow, col))) for col in zip(*probMatrix)]
        oldFirstRow = probVector

    # generate numerators
    numerators = []
    for i in probVector:
        numerator = fractions.Fraction(i).limit_denominator().numerator
        numerators.append(numerator)

    # generate denominators
    denominators = []
    for i in probVector:
        denominator = fractions.Fraction(i).limit_denominator().denominator
        denominators.append(denominator)

    # calculate factors to multiply numerators by
    factors = [max(denominators) / x for x in denominators]
    # multiply numerators by factors
    numeratorsTimesFactors = [a * b for a, b in zip(numerators, factors)]
    # filter numerators by terminal state booleans
    terminalStateNumerators = list(compress(numeratorsTimesFactors, terminalStateFilter))

    # append numerators and denominator to answer
    answerlist = []
    for i in terminalStateNumerators:
        answerlist.append(i)
    answerlist.append(max(denominators))

    return list(map(int, answerlist))


print(answer([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))



# 一稿，超时
def answer(m):
    n = len(m)
    newl = []
    for i in range(n):
        if list(set(m[i])) == [0]:
            newl.append(i)
    gap = len(newl)-1 # gap都是absorbing matrix, 对角线都为1. gap = 3
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
            R[i][j] = [m[row][col], sum(m[row])] # R = [[[0, 2], [0, 2], [0, 2], [1, 2]], [[0, 9], [3, 9], [2, 9], [0, 9]]]
    # R的分母转化为相同值
    lcmvalue = R[0][0][1]
    for i in range(1, length):
        lcmvalue = lcm(lcmvalue, R[i][0][1])  # lcmvalue = 18
    for i in range(length):
        for j in range(gap+1):
            R[i][j] = [R[i][j][0] * (lcmvalue / R[i][j][1]), lcmvalue] # R = [[[0, 18], [0, 18], [0, 18], [9, 18]], [[0, 18], [6, 18], [4, 18], [0, 18]]]
    FR = [0 for i in range(gap+1)]
    for i in range(gap+1):
        FR[i] = F[0][0][0] * R[0][i][0]
        for j in range(1, length):
            FR[i] += F[0][j][0] * R[j][i][0] # FR = [0, 54, 36, 162]
    FR += [sum(FR)] # [0, 54, 36, 162, 252]
    res = FR[-1]
    for i in range(len(FR)-1,-1,-1):
        if FR[i] != 0:
            res = hcf(res,FR[i])
    for i in range(len(FR)):
        FR[i] /= res
    return FR

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


# 二稿，40行lcmvalue = IminusQ[0][0][1]IndexError

def answer(m):
    n = len(m)
    oldl = [i for i in range(n)]
    newl = []
    for i in range(n):
        if list(set(m[i])) == [0]:
            newl.append(i)
    gap = len(newl)-1 # gap都是absorbing matrix, 对角线都为1. gap = 3
    for i in range(n):
        if i not in newl:
            newl.append(i)
    # 构造一个new matrix, 横纵坐标为[2, 3, 4, 5, 0, 1]
    matrix = [[0 for j in range(n)] for i in range(n)]
    # I块,O块，R和Q块
    # 求F = (I-Q)^-1
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
    # 求FR
    R = [[0 for j in range(gap+1)] for i in range(length)]
    for i in range(length):
        for j in range(gap+1):
            row = newl[gap + 1 + i]
            col = newl[j]
            R[i][j] = [m[row][col], sum(m[row])] # R = [[[0, 2], [0, 2], [0, 2], [1, 2]], [[0, 9], [3, 9], [2, 9], [0, 9]]]
    # R的分母转化为相同值
    lcmvalue = R[0][0][1]
    for i in range(1, length):
        lcmvalue = lcm(lcmvalue, R[i][0][1])  # lcmvalue = 18
    for i in range(length):
        for j in range(gap+1):
            R[i][j] = [R[i][j][0] * (lcmvalue / R[i][j][1]), lcmvalue] # R = [[[0, 18], [0, 18], [0, 18], [9, 18]], [[0, 18], [6, 18], [4, 18], [0, 18]]]
    FR = [0 for i in range(gap+1)]
    for i in range(gap+1):
        FR[i] = F[0][0][0] * R[0][i][0]
        for j in range(1, length):
            FR[i] += F[0][j][0] * R[j][i][0]  # FR = [0, 54, 36, 162]
        FR[i] /= max(F[0][0][1], F[0][0][1])
    FR += [sum(FR)] # [0, 54, 36, 162, 252]
    return FR

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

#三稿，1,2,4pass
# -*- coding:utf8 -*-

def add(x, y):
    if x[0] == 0 and y[0] == 0:
        return [0, 0]
    if x[0] == 0:
        return y
    if y[0] == 0:
        return x
    den = lcm(x[1], y[1])
    num = x[0] * (den / x[1]) + y[0] * (den / y[1])
    tmp = hcf(num, den)
    return [num / tmp, den / tmp]

def minus(x, y):
    if x[0] == 0 and y[0] == 0:
        return [0, 0]
    if x[0] == 0:
        return [-1*y[0],y[1]]
    if y[0] == 0:
        return x
    den = lcm(x[1], y[1])
    num = x[0] * (den / x[1]) - y[0] * (den / y[1])
    tmp = hcf(num, den)
    return [num / tmp, den / tmp]

def multiply(x, y):
    if x[0] == 0 or y[0] == 0:
        return [0, 0]
    den = x[1] * y[1]
    num = x[0] * y[0]
    tmp = hcf(num, den)
    return [num / tmp, den / tmp]

# 最大公约数
def hcf(a, b):
    if a < b:
        a, b = b, a
    while b:
        a, b = b, a % b
    return a

# 最小公倍数
def lcm(a, b):
    if hcf(a, b) == 0:
        return 1
    return a * b / hcf(a, b)

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

def answer(m):
    n = len(m)
    if sum([sum([m[j][i] for i in range(n)]) for j in range(n)]) == 0:
        return [1]+[0]*(n-1)+[1] #一开始就把所有元素为0的矩阵情况删除
    newl = []
    for i in range(n):
        if list(set(m[i])) == [0]:
            newl.append(i)
    gap = len(newl)-1 # gap都是absorbing matrix, 对角线都为1. gap = 3
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
                I[i][j] = [1, 1]
            else:
                I[i][j] = [0,tmp] # I = [[[1, 1], [0, 2]], [[0, 9], [1, 1]]]

    # I-Q
    IminusQ = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            IminusQ[i][j] = minus(I[i][j], Q[i][j]) # IminusQ = [[[1, 1], [-1, 2]], [[-4, 9], [1, 1]]]
    # 求(I-Q)的逆矩阵F
    # 先转成整数，记录最小公倍数lcmvalue。再求逆矩阵，记录行列式的值A
    lcmvalue = IminusQ[0][0][1]
    for i in range(length):
        for j in range(length):
            if IminusQ[i][j][0] != 0:
                lcmvalue = lcm(lcmvalue, IminusQ[i][j][1]) # lcmvalue = 18
    for i in range(length):
        for j in range(length):
            if IminusQ[i][j][0] != 0:
                IminusQ[i][j] = IminusQ[i][j][0] * (lcmvalue / IminusQ[i][j][1]) # IminusQ = [[18, -9], [-8, 18]]
            else:
                IminusQ[i][j] = 0 # 把IminusQ的分母都去掉
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

    if len(F) == 1:
        F = [[1]]

    # 第二步求FR
    R = [[0 for j in range(gap+1)] for i in range(length)]
    for i in range(length):
        for j in range(gap+1):
            row = newl[gap + 1 + i]
            col = newl[j]
            R[i][j] = [m[row][col], sum(m[row])] # 2.R = [[[0, 2], [0, 2], [0, 2], [1, 2]], [[0, 9], [3, 9], [2, 9], [0, 9]]]
    if F == [[1]]:
        new = [R[0][i][0] for i in range(len(R[0]))]
        return new+[sum(new)]

    F = F[0]
    for i in range(len(F)):
        tmp = hcf(F[i][0], F[i][1])
        F[i] = [F[i][0] / tmp, F[i][1] / tmp]  # 1.F = [[9, 7], [9, 14]]

    FR = [0 for i in range(gap + 1)]
    for j in range(gap + 1):
        FR[j] = multiply(F[0], R[0][j])
        for i in range(1, length):
            tmp = multiply(F[i], R[i][j])
            if tmp[0] == 0:
                continue
            FR[j] = add(FR[j], tmp)  # FR=[[0, 0], [3, 14], [1, 7], [9, 14]]
    if [FR[i][1] for i in range(len(FR)) if FR[i][0] != 0] != []:
        common = reduce((lambda x, y: lcm(x,y)), [FR[i][1] for i in range(len(FR)) if FR[i][0] != 0]) # common = 14
        new = []
        for i in range(len(FR)):
            if FR[i][0] == 0:
                new.append(0)
            else:
                new.append(FR[i][0]*(common/FR[i][1]))
        if len(new) == 1:
            return [1,1]
        return new + [sum(new)]
    else:
        return [0]



m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
print answer(m)
m1 =  [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
print answer(m1)
m1 =  [[0, 1, 2], [0, 0, 0], [0, 0, 0]]
print answer(m1)
m1 =  [[0, 1], [0, 0]]
print answer(m1)
m1 =  [[0, 0,0], [0, 0,0],[0,0,0]]
print answer(m1)
m1 =  [[0]]
print answer(m1)
m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 2, 0, 0, 3, 0], [1, 1, 3, 0, 0, 0], [0, 0, 0, 1, 0, 2], [0, 0, 0, 0, 0, 0]]
print answer(m)

# 第四稿细改，还是只有1,2,4pass
# -*- coding:utf8 -*-
#假设负数都出现在分子
def add(x, y):
    if x[0] == 0 and y[0] == 0:
        return [0, 1]
    if x[0] == 0:
        return y
    if y[0] == 0:
        return x
    den = lcm(x[1], y[1])
    num = x[0] * (den / x[1]) + y[0] * (den / y[1])
    tmp = hcf(abs(num), abs(den)) #注意负数
    return [num / tmp, den / tmp]

def minus(x, y):
    if x[0] == 0 and y[0] == 0:
        return [0, 1]
    if x[0] == 0:
        return [-1*y[0], y[1]]
    if y[0] == 0:
        return x
    den = lcm(x[1], y[1])
    num = x[0] * (den / x[1]) - y[0] * (den / y[1])
    tmp = hcf(abs(num), abs(den))  # 注意负数
    return [num / tmp, den / tmp]

def multiply(x, y):
    if x[0] == 0 or y[0] == 0:
        return [0, 1]
    den = x[1] * y[1]
    num = x[0] * y[0]
    tmp = hcf(abs(num), abs(den))  # 注意负数
    return [num / tmp, den / tmp]

# 最大公约数
def hcf(a, b):
    if a == 0 or b == 0:
        return 1
    if a < b:
        a, b = b, a
    while b:
        a, b = b, a % b
    return a

# 最小公倍数
def lcm(a, b):
    return a * b / hcf(a, b)

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

def answer(m):
    n = len(m)
    special = [0] * n
    for i in range(n):
        if sum(m[i]) != 0:
            special[i] = 1
    if sum(special) == 0:
        return [1] + [0] * (n - 1) + [1]  # 一开始就把所有元素为0的矩阵情况删除
    if sum(special) == n-1: # 只有一行全为0，也就是只有一个end state
        return [1,1]
    if sum(special) == 1: # 只有第一行有数字
        index = 0
        for i in range(1,n):
            if m[0][i] !=0:
                index = i
                break
        new = m[0][index:]
        return new + [sum(new)]

    newl = []
    for i in range(n):
        if list(set(m[i])) == [0]:
            newl.append(i)
    gap = len(newl)-1 # gap都是absorbing matrix, 对角线都为1. gap = 3
    length = n-(gap+1)
    for i in range(n):
        if i not in newl:
            newl.append(i) # newl = [2, 3, 4, 5, 0, 1]

    # 构造一个new matrix, 横纵坐标为[2, 3, 4, 5, 0, 1]，分为I块,O块，R和Q块一共四块
    # 第一步求F = (I-Q)^-1

    # Q
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
            if i == j:
                I[i][j] = [1, 1]
            else:
                I[i][j] = [0, 1] # I = [[[1, 1], [0, 1]], [[0, 1], [1, 1]]]

    # I-Q
    IminusQ = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            IminusQ[i][j] = minus(I[i][j], Q[i][j]) # IminusQ = [[[1, 1], [-1, 2]], [[-4, 9], [1, 1]]]

    # 求(I-Q)的逆矩阵F
    # 先转成整数，记录最小公倍数lcmvalue。再求逆矩阵，记录行列式的值A
    lcmvalue = IminusQ[0][0][1]
    for i in range(length):
        for j in range(length):
            if IminusQ[i][j][0] != 0:
                lcmvalue = lcm(lcmvalue, IminusQ[i][j][1]) # lcmvalue = 18
    for i in range(length):
        for j in range(length):
            if IminusQ[i][j][0] != 0:
                IminusQ[i][j] = IminusQ[i][j][0] * (lcmvalue / IminusQ[i][j][1]) # IminusQ = [[18, -9], [-8, 18]]
            else:
                IminusQ[i][j] = 0 # 把IminusQ的分母都去掉
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

    F = F[0]
    for i in range(len(F)):
        tmp = hcf(abs(F[i][0]), abs(F[i][1]))
        F[i] = [F[i][0] / tmp, F[i][1] / tmp]  # 1.F = [[9, 7], [9, 14]]

    FR = [0 for i in range(gap + 1)]
    for j in range(gap + 1):
        FR[j] = multiply(F[0], R[0][j])
        for i in range(1, length):
            FR[j] = add(FR[j], multiply(F[i], R[i][j]))  # FR=[[0, 1], [3, 14], [1, 7], [9, 14]]

    common = FR[0][1]
    for i in range(1, len(FR)):
        common = lcm(common, FR[i][1])
    new = []
    for i in range(len(FR)):
         new.append(FR[i][0]*(common/FR[i][1]))
    return new + [sum(new)]


m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
print answer(m)
m1 =  [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
print answer(m1)
m1 =  [[0, 1, 2], [0, 0, 0], [0, 0, 0]]
print answer(m1)
m1 =  [[0, 1], [0, 0]]
print answer(m1)
m1 =  [[0, 0,0], [0, 0,0],[0,0,0]]
print answer(m1)
m1 =  [[0]]
print answer(m1)
m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 2, 0, 0, 3, 0], [1, 1, 3, 0, 0, 0], [0, 0, 0, 1, 0, 2], [0, 0, 0, 0, 0, 0]]
print answer(m)



from itertools import product
from fractions import Fraction
from functools import reduce
#this is for matrix inversion
def invert(matrix):
    n = len(matrix)
    inverse = [[Fraction(0) for col in range(n)] for row in range(n)]
    for i in range(n):
        inverse[i][i] = Fraction(1)
    for i in range(n):
        for j in range(n):
            if i != j:
                if matrix[i][i] == 0:
                    return False
                ratio = matrix[j][i] / matrix[i][i]
                for k in range(n):
                    inverse[j][k] = inverse[j][k] - ratio * inverse[i][k]
                    matrix[j][k] = matrix[j][k] - ratio * matrix[i][k]
    for i in range(n):
        a = matrix[i][i]
        if a == 0:
            return False
        for j in range(n):
            inverse[i][j] = inverse[i][j] / a
    return inverse
#finding sum of a row in matrix
def sumRow(m, r):
    return sum(m[r])
#subtracting two matrices
def substract(matr_a, matr_b):
    output = []
    for i in range(len(matr_a)):
        tmp = []
        for valA, valB in zip(matr_a[i], matr_b[i]):
            tmp.append(valA - valB)
        output.append(tmp[:])
    return output[:]
#matrix multiplication
def matrixmult(matr_a, matr_b):
    #cols = len(matr_b[0])
    rows = len(matr_b)
    if rows is not 0:
        cols = len(matr_b[0])
    else:
        cols = 0
    resRows = range(len(matr_a))
    rMatrix = [[0] * cols for _ in resRows]
    for idx in resRows:
        for j, k in product(range(cols), range(rows)):
            rMatrix[idx][j] += matr_a[idx][k] * matr_b[k][j]
    if cols is not 0:
        return rMatrix
    else:
        return 0
    # return rMatrix
#gcd to find lcm
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
#lcm to find last value of the output
def lcm(a,n):
    ans = a[0]
    for i in range(1,n):
        ans = (a[i]*ans)//gcd(a[i],ans)
    return ans
#main function
def answer(m):
    num = len(m)
    f=[]
    #finding zero rows
    for i in range(0,num):
        j=[]
        j.append(sumRow(m,i))
        j.append(i)
        f.append(j)
    k=0;
    #Fraction Conversion
    for i in range(0,num):
        for j in range(0,num):
            if f[i][0]!=0:
                m[i][j]=Fraction(m[i][j],f[i][0])
    j=[]
    for i in range(0,len(f)):
        if f[i][0].numerator==0:
            j.append(i)
            del m[f[i][1]-k]
            k=k+1
    q=m
    k=[]
    t=0
    e=m
   # print(m)
    q=[]
    for i in range(0,len(m)):
        row=[]
        for r in range(0,len(m[0])):
            for t in range(0,len(j)):
                if j[t] is r:
                    row.append(m[i][r])
        q.append(row)
   # print(q)
    t=0;
    w=0;
    e=[]
    for i in range(0,len(m)):
        row=[]
        flag=1
        for r in range(0,len(m[0])):
            for t in range(0,len(j)):
                if j[t] is r:
                    flag=0
                    break
            if flag is 1:
                row.append(m[i][r])
        e.append(row)
    #print(e)
    l=[]
    for i in range(0,len(e)):
        k=[]
        for b in range(0,len(e)):
            if i==b:
                k.append(1)
            else:
                k.append(0)
        l.append(k)
    #print(l)
    #print(e)
    l=substract(l,e)
    #print(l)
    l=invert(l)
    r = matrixmult(l,q)
    #print(r)
    if r == 0:
        return 0
    else:
        m =r[0]
    e=[]
    for i in range(0,len(m)):
        e.append(m[i].denominator)
    k=lcm(e,len(e))
    e=[]
    for i in range(0,len(m)):
        e.append((m[i].numerator*k)//m[i].denominator)
    e.append(k)
    #print(e)
    return e


# 第五稿 1,2,3,5,6,9,10pass
# -*- coding:utf8 -*-
from itertools import product
from fractions import Fraction
from functools import reduce


# this is for matrix inversion
def invert(matrix):  # 求逆矩阵
    n = len(matrix)
    inverse = [[Fraction(0) for col in range(n)] for row in range(n)]
    for i in range(n):
        inverse[i][i] = Fraction(1)  # 先把对角线变成1/1
    for i in range(n):  # 如matrix对角线为0，False； ratio = matrix[j][i]/对角线，
        for j in range(n):
            if i != j:
                if matrix[i][i] == 0:
                    return False
                ratio = matrix[j][i] / matrix[i][i]
                for k in range(n):
                    inverse[j][k] = inverse[j][k] - ratio * inverse[i][k]
                    matrix[j][k] = matrix[j][k] - ratio * matrix[i][k]
    for i in range(n):
        a = matrix[i][i]
        if a == 0:
            return False
        for j in range(n):
            inverse[i][j] = inverse[i][j] / a
    return inverse


# finding sum of a row in matrix
def sumRow(m, r):  # 求矩阵第几行的元素和
    return sum(m[r])


# subtracting two matrices
def substract(matr_a, matr_b):  # 求两个矩阵相减后的矩阵
    output = []
    for i in range(len(matr_a)):
        tmp = []
        for valA, valB in zip(matr_a[i], matr_b[i]):
            tmp.append(valA - valB)
        output.append(tmp[:])
    return output[:]


# matrix multiplication
def matrixmult(matr_a, matr_b):  # 求两个矩阵相乘后的矩阵
    # cols = len(matr_b[0])
    rows = len(matr_b)
    if rows is not 0:
        cols = len(matr_b[0])
    else:
        cols = 0
    resRows = range(len(matr_a))
    rMatrix = [[0] * cols for _ in resRows]
    for idx in resRows:
        for j, k in product(range(cols), range(rows)):
            rMatrix[idx][j] += matr_a[idx][k] * matr_b[k][j]
    if cols is not 0:
        return rMatrix
    else:
        return 0
        # return rMatrix


# gcd to find lcm
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# lcm to find last value of the output
def lcm(a, n):
    ans = a[0]
    for i in range(1, n):
        ans = (a[i] * ans) // gcd(a[i], ans)
    return ans


# main function
def answer(m):
    num = len(m)
    # if num >= 3: # 如果加上这一段，9failed
    #     for i in range(1, num):
    #         if sum([1 for item in m[i] if item != 0]) == 1:
    #             m[i] = [0] * num

    f = []
    # finding zero rows
    for i in range(0, num):
        j = []
        j.append(sumRow(m, i))
        j.append(i)
        f.append(j)
    k = 0;
    # Fraction Conversion
    for i in range(0, num):
        for j in range(0, num):
            if f[i][0] != 0:
                m[i][j] = Fraction(m[i][j], f[i][0])
    j = []
    for i in range(0, len(f)):
        if f[i][0].numerator == 0:
            j.append(i)
            del m[f[i][1] - k]
            k = k + 1
    q = m
    k = []
    t = 0
    e = m
    # print(m) # 此时m为所有元素化为最简分数的矩阵，且不含全部为0的那些行

    q = []
    for i in range(0, len(m)):
        row = []
        for r in range(0, len(m[0])):
            for t in range(0, len(j)):
                if j[t] is r:
                    row.append(m[i][r])
        q.append(row)
        # print(q), q = [[], [], []]

    t = 0;
    w = 0;
    e = []
    for i in range(0, len(m)):
        row = []
        flag = 1
        for r in range(0, len(m[0])):
            for t in range(0, len(j)):
                if j[t] is r:
                    flag = 0
                    break
            if flag is 1:
                row.append(m[i][r])
        e.append(row)
    # print(e)，e与m差不多，只是如果m[i][j]指向的j行如果全部为0，则不考虑这个m[i][j]

    l = []
    for i in range(0, len(e)):
        k = []
        for b in range(0, len(e)):
            if i == b:
                k.append(1)
            else:
                k.append(0)
        l.append(k)
    # print(l) #单位矩阵
    # print(e) #仍为e

    l = substract(l, e)
    # print(l) # 可能为[]或者行列不相等的矩阵！


    if l == []:
        return 0

    if len(l) != len(l[0]):
        return 0

    l = invert(l)
    r = matrixmult(l, q)
    # print(r)
    if r == 0:
        return 0
    else:
        m = r[0]
    e = []
    for i in range(0, len(m)):
        e.append(m[i].denominator)
    k = lcm(e, len(e))
    e = []
    for i in range(0, len(m)):
        e.append((m[i].numerator * k) // m[i].denominator)
    e.append(k)
    # print(e)
    return e


m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
print answer(m)
m1 =  [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
print answer(m1)
m1 =  [[0, 1, 2], [0, 0, 0], [0, 0, 0]]
print answer(m1)
m1 =  [[0, 1], [0, 0]]
print answer(m1)
m1 =  [[0, 1,2], [1, 0,0],[0,0,0]]
print answer(m1)
m1 =  [[0]]
print answer(m1)
m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 2, 0, 0, 3, 0], [1, 1, 3, 0, 0, 0], [0, 0, 0, 1, 0, 2], [0, 0, 0, 0, 0, 0]]
print answer(m)
m= [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
print answer(m)


# 第六稿，调整过的
from __future__ import division
from itertools import compress
from itertools import starmap
from operator import mul
import fractions

def convertMatrix(transMatrix):
    probMatrix = []
    for i in range(len(transMatrix)):
        row = transMatrix[i]
        newRow = []
        rowSum = sum(transMatrix[i])
        if all([v == 0 for v in transMatrix[i]]):
            for j in transMatrix[i]:
                newRow.append(0)
            newRow[i] = 1
            probMatrix.append(newRow)
        else:
            for j in transMatrix[i]:
                if j == 0:
                    newRow.append(0)
                else:
                    newRow.append(j / rowSum)
            probMatrix.append(newRow)
    return probMatrix


def answer(m):
    if len(m) == 0:
        return []
    if sum([1 for total in [sum(item) for item in m] if total == 0]) == 0:
        return [0]
    # convert matrix numbers into probabilities
    probMatrix = convertMatrix(m)

    # find terminal states
    terminalStateFilter = []
    for row in range(len(m)):
        if all(x == 0 for x in m[row]):
            terminalStateFilter.append(True)
        else:
            terminalStateFilter.append(False)

    # multiply matrix by probability vector
    oldFirstRow = probMatrix[0]
    probVector = None
    for i in range(3000):
        probVector = [sum(starmap(mul, zip(oldFirstRow, col))) for col in zip(*probMatrix)]
        oldFirstRow = probVector

    # generate numerators
    numerators = []
    for i in probVector:
        numerator = fractions.Fraction(i).limit_denominator(max_denominator=2147483648).numerator
        numerators.append(numerator)

    # generate denominators
    denominators = []
    for i in probVector:
        denominator = fractions.Fraction(i).limit_denominator(max_denominator=2147483648).denominator
        denominators.append(denominator)

    # calculate factors to multiply numerators by
    factors = [max(denominators) / x for x in denominators]
    # multiply numerators by factors
    numeratorsTimesFactors = [a * b for a, b in zip(numerators, factors)]
    # filter numerators by terminal state booleans
    terminalStateNumerators = list(compress(numeratorsTimesFactors, terminalStateFilter))

    # append numerators and denominator to answer
    answerlist = []
    for i in terminalStateNumerators:
        answerlist.append(i)
    answerlist.append(max(denominators))

    return list(map(int, answerlist))


#第六稿，加了fraction的自己的方法，1,2pass
# -*- coding:utf8 -*-
from fractions import Fraction
from itertools import product


def matrixmult(matr_a, matr_b):
    rows = len(matr_b)
    if rows != 0:
        cols = len(matr_b[0])
    else:
        cols = 0
    rMatrix = [0] * cols
    for j, k in product(range(cols), range(rows)):
        rMatrix[j] += matr_a[k] * matr_b[k][j]
    if cols != 0:
        return rMatrix
    else:
        return [0]

# 最大公约数
def hcf(a, b):
    if a == 0 or b == 0:
        return 1
    if a < b:
        a, b = b, a
    while b:
        a, b = b, a % b
    return a

# 最小公倍数
def lcm(a, b):
    return a * b / hcf(a, b)

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

def answer(m):
    n = len(m)
    if n == 1 and m != [[0]]:
        return [0]
    if n == 1 and m == [[0]]:
        return [1]

    newl = []
    for i in range(n):
        if list(set(m[i])) == [0]:
            newl.append(i)
    gap = len(newl)-1 # gap都是absorbing matrix, 对角线都为1. gap = 3
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
            Q[i][j] = Fraction(m[row][col],sum(m[row]))
    # I
    I = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            if i == j:
                I[i][j] = 1
            else:
                I[i][j] = 0

    # I-Q
    IminusQ = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            IminusQ[i][j] = I[i][j] - Q[i][j]
    # 求(I-Q)的逆矩阵F
    A = MatrixGetDet(IminusQ)
    F = [[0 for j in range(length)] for i in range(length)]
    # 求伴随矩阵，注意先删j行，再删i列
    if len(IminusQ) == 1:
        F = [[1]]
    else:
        for i in range(length):
            for j in range(length):
                copy = IminusQ[:j]+IminusQ[j+1:]
                for x in range(len(copy)):
                    copy[x] = copy[x][:i]+copy[x][i+1:]
                F[i][j] = Fraction(MatrixGetDet(copy) * pow(-1,i+j), A)
    # 第二步求FR
    R = [[0 for j in range(gap+1)] for i in range(length)]
    for i in range(length):
        for j in range(gap+1):
            row = newl[gap + 1 + i]
            col = newl[j]
            R[i][j] = Fraction(m[row][col], sum(m[row]))
    F = F[0]
    FR = matrixmult(F,R)
    if FR == []:
        return [0]
    new = []
    # 提取分数分母的最小公倍数
    common = FR[0].denominator
    for i in range(1, len(FR)):
        common = lcm(common, FR[i].denominator)
    for i in range(len(FR)):
        new.append(FR[i].numerator * common / FR[i].denominator)
    return new + [sum(new)]


m= [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
print answer(m)
m1 =  [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
print answer(m1)
m1 =  [[0, 1, 2], [0, 0, 0], [0, 0, 0]]
print answer(m1)
m1 =  [[0, 1], [0, 0]]
print answer(m1)
m1 =  [[0]]
print answer(m1)



# weezer,78fail
from fractions import Fraction
from copy import deepcopy

Q_size = 0


def hcf(a, b):
    if a == 0 or b == 0:
        return 1
    if a < b:
        a, b = b, a
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    return a * b / hcf(a, b)


def convert_matrix(mat):
    global Q_size
    mat_len = len(mat)
    frac_mat = [[0 for i in range(mat_len)] for j in range(mat_len)]
    for i in range(mat_len):
        sum_line = sum(mat[i])
        if sum_line == 0:
            Q_size = i
            break
        else:
            for j in range(mat_len):
                if mat[i][j]:
                    frac_mat[i][j] = Fraction(mat[i][j], sum_line)
    return frac_mat


def get_Q(mat):
    global Q_size
    Q_mat = []
    for i in range(Q_size):
        Q_mat.append(mat[i][:Q_size])
    return Q_mat


def get_I(mat):
    global Q_size
    I_mat = [[0 for i in range(Q_size)] for j in range(Q_size)]
    for i in range(Q_size):
        I_mat[i][i] = 1
    return I_mat


def get_R(mat):
    global Q_size
    R_mat = []
    for i in range(Q_size):
        R_mat.append(mat[i][Q_size:])
    return R_mat


def get_I_minus_Q(mat):
    global Q_size
    mat_b = [[0 for i in range(Q_size)] for j in range(Q_size)]
    for i in range(Q_size):
        mat_b[i][i] = 1

    for i in range(Q_size):
        for j in range(Q_size):
            mat[i][j] = mat_b[i][j] - mat[i][j]

    return mat


def inverse_IQ(mat):
    global Q_size
    for i in range(Q_size):
        if mat[i][i] != 1:
            numerator = 1 / mat[i][i]
            for j in range(Q_size * 2):
                mat[i][j] = mat[i][j] * numerator
        for k in range(i + 1, Q_size):
            if mat[k][i] != 0:
                multiper_num = -mat[k][i]
                for j in range(Q_size * 2):
                    mat[k][j] = mat[i][j] * multiper_num + mat[k][j]
    for i in range(Q_size - 1, -1, -1):
        for k in range(i - 1, -1, -1):
            if mat[k][i] != 0:
                multiper_num = -mat[k][i]
                for j in range(Q_size * 2):
                    mat[k][j] = mat[i][j] * multiper_num + mat[k][j]
    F_mat = []
    for i in range(Q_size):
        F_mat.append(mat[i][Q_size:])
    return F_mat


def get_F_R(F_mat, R_mat):
    if R_mat == []:
        return [0]
    F_R_mat = [0 for i in range(len(R_mat[0]))]
    for i in range(len(R_mat[0])):
        for j in range(len(F_mat[0])):
            F_R_mat[i] += F_mat[0][j] * R_mat[j][i]
    return F_R_mat


def answer(m):
    n = len(m)
    if n == 1 and m != [[0]]:
        return [0, 0]
    if n == 1 and m == [[0]]:
        return [1, 1]
    if n == 0:
        return [0, 0]
    for i in range(len(m)):
        sum_line = sum(m[i])
        if sum_line == 0:
            break
        if i == len(m) - 1:
            return [0, 0]
    flag = False
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i][j] != 0:
                if sum(m[j]) == 0:
                    flag = True
            if flag:
                break
    if flag is False:
        return [0, 0]
    check_list = [0]
    visited = []
    while check_list:
        current = check_list.pop()
        visited.append(current)
        for i in range(len(m)):
            if m[current][i] != 0 and i not in visited:
                check_list.append(i)
    flag = False
    for i in visited:
        if sum(m[i]) == 0:
            flag = True
    if flag is False:
        return [0, 0]

    frac_mat = convert_matrix(m)

    Q_mat = get_Q(frac_mat)

    I_mat = get_I(frac_mat)

    I_minus_Q = get_I_minus_Q(Q_mat)

    for i in range(Q_size):
        I_minus_Q[i].extend(I_mat[i])

    F_mat = inverse_IQ(I_minus_Q)

    R_mat = get_R(frac_mat)

    F_R_result = get_F_R(F_mat, R_mat)
    new = []
    common = F_R_result[0].denominator
    for i in range(1, len(F_R_result)):
        common = lcm(common, F_R_result[i].denominator)
    for i in range(len(F_R_result)):
        new.append(F_R_result[i].numerator * common / F_R_result[i].denominator)
    return new + [sum(new)]


# james
from fractions import Fraction
from copy import deepcopy

def gcd(a, b):
    """Binary method of finding gcd. Lifted from wikipedia

    Used to reduce vector headings to their simplist form.
    Negative numbers are allowed as input. They are converted
    to positive integers before the process begins.

    :param a: Some integer a
    :type a: int
    :param b: Some integer b
    :type b: int
    :returns: The greatest common divisor of both a and b
    :rtype: int
    """
    p = 0
    if a < 0:
        a *= -1
    if b < 0:
        b *= -1

    if a == 0:
        return b
    if b == 0:
        return a

    while a % 2 == 0 and b % 2 == 0:
        a >>= 1
        b >>= 1
        p += 1
    while a != b:
        if a % 2 == 0:
            a >>= 1
        elif b % 2 == 0:
            b >>= 1
        elif a > b:
            a = (a - b) >> 1
        else:
            b = (b - a) >> 1
    return a << p


def lcm(numbers):
    """Find least common multiple from a list of integers.

    :param numbers: List of positive integers
    :type numbers: list
    :returns: Least common multiple
    :rtype: dict
    """
    res = 1
    for n in numbers:
        res = res * n / gcd(res, n)
    return res


class Matrix(list):
    """Quick matrix matrix class."""
    def __init__(self, data):
        """Takes a list of lists to represent a matrix.

        :param data: List of lists
        :type data: list
        """
        self.extend(data)

    def __sub__(self, other):
        """Matrix subtraction.

        :param other: Matrix to subtract from self.
        :type other: Matrix
        :returns: New matrix that is self - other
        :rtype: Matrix
        """
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
        """Multiply self by other matrix.

        :param other: Matrix to multiply self by
        :type other: Matrix
        :returns: New matrix that is self * other
        :rtype: Matrix
        """
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
        """Finds the inverse of self.

        Uses the method where the identity matrix is appended
        to the right of self. Row operations are performed to reduce
        the left side to the identity matrix. The resulting right
        side is the inverse matrix.

        :returns: Inverse of self.
        :rtype: Matrix
        """
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

        # We should have an upper triangular matrix now with non zero diagonals
        # Reduce the upper part as well
        # Start at the bottom right
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
        """Create a n x n identity matrix.

        :param n: Positive integer dimension
        :type n: int
        :returns: n x n identity matrix.
        :rtype: Matrix
        """
        m = []
        for i in xrange(n):
            r = [Fraction(0, 1) for j in xrange(n)]
            r[i] = Fraction(1, 1)
            m.append(r)
        return cls(m)

class Row(list):
    """Quick convenience class for fractionalizing and row classification."""
    def __init__(self, state_number, data):
        """Inits the row.

        :param state_number: Original position within transition matrix
        :type state_number: int
        :param data: List of integers
        :type data: list
        """
        super(Row, self).__init__()
        self.extend(data)
        self.state_number = state_number
        self.absorbing = not any(data)
        if self.absorbing:
            self[self.state_number] = 1

    def __repr__(self):
        """For visualization purposes.

        :returns: String representation of row.
        :rtype: list
        """
        return "s{}:\t {} -- absorbing: {}".format(self.state_number, super(Row, self).__repr__(), self.absorbing)

    def fractionalize(self):
        """Convert values to fractions."""
        s = sum(self)
        for i, v in enumerate(self):
            self[i] = Fraction(v, s)


class P:
    """Represents the transition matrix."""
    def __init__(self, data):
        """Inits the transition matrix.

        Identifies transition and absorbing states.

        :param data: List of rows
        :type data: list
        """
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
        """Turns rows of ints to rows of fractions."""
        for r in self.rows:
            r.fractionalize()

    def swap_columns(self, c1, c2):
        """Swap columns in self.

        :param c1: 0 indexed column 1
        :type c1: int
        :param c2: 0 indexed column 2
        :type c2: int
        """
        for r in self.rows:
            r[c1] = r[c1] ^ r[c2]
            r[c2] = r[c2] ^ r[c1]
            r[c1] = r[c1] ^ r[c2]

    def _reorder(self):
        """Checks if any states are not in their original order."""
        swapped = set()
        schedule = [r.state_number for r in self.rows]

        for r in self.rows:
            newrow = [r[s] for s in schedule]
            for i in xrange(len(r)):
                r.pop()
            r.extend(newrow)

        #for i, r in enumerate(self.rows):
        #    if i != r.state_number and (i, r.state_number) not in swapped:
        #        self.swap_columns(i, r.state_number)
        #        swapped.add((i, r.state_number))
        #        swapped.add((r.state_number, i))

    def __repr__(self):
        """For display purposes.

        :returns: String representation
        :rtype: str
        """
        return "\n".join([r.__repr__() for r in self.rows])

    def Q(self):
        """Returns t x t Q portion of matrix.

        :returns: Matrix Q
        :rtype: Matrix
        """
        q = []
        for i in xrange(self.t):
            q.append(self.rows[i][:self.t])
        return Matrix(q)

    def R(self):
        """Returns t x r R portion of matrix.

        :returns: Matrix R
        :rtype: Matrix
        """
        r = []
        for i in xrange(self.t):
            r.append(self.rows[i][self.t:])
        return Matrix(r)

    def N(self):
        """Returns fundamental matrix

        :returns: Matrix N or fundamental matrix
        :rtype: Matrix
        """
        Q = self.Q()
        i_minus_q = Matrix.identity(len(Q)) - Q
        N = i_minus_q.inverse()
        return N

    def NR(self):
        """Returns absorbing probability matrix NR

        :returns: Matrix NR
        :rtype: Matrix
        """
        N = self.N()
        R = self.R()
        NR = N.multiply(R)
        return NR

def answer(data):
    """Computes probabilities of entering absorbing states from state 0.

    :param data: List of list of integers
    :type data: list
    """

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
