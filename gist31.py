# -*- coding:utf8 -*-

def bunny(n):
    k = 2
    total = 0
    while k <= int(pow(n * 2, 0.5)):
        total += method(n, k)
        k += 1
    return total

def method(n, k):
    remain = n - (1 + k) * k / 2  # 确保每个盒子不为空，且已递增为1-4
    if remain < 0:
        return 0
    if remain == 0:  # remain=0，只有1-k这一种情况
        return 1
    l = [i for i in range(1, k + 1)]  # [1, 2, 3, 4]含义为撒1个球的方法：1,1,1,1; 0,1,1,1;0,0,1,1;0,0,0,1.
    return dp_increament(l, remain)

def dp_increament(nums, target):
    matrix = [1] + [0] * target
    for i in nums:
        for j in range(1, target+1):
            if j >= i:
                matrix[j] += matrix[j-i]
    return matrix[target]
print bunny(200)


# 延伸知识
def sa(array,target): # target用array里面的元素组成，元素可重复使用，能有多少种组合方法
    n = len(array)
    dp = [[1]*n]+[[0]*n for x in range(target)]
    for i in range(1, target+1):
        dp[i][0] = 1
    for i in range(1, target + 1): # 动态转移方程dp[i][j] = dp[i][j-1]+dp[i-array[j]][j]
        for j in range(1,n):
            dp[i][j] += dp[i][j-1]
            if i>=array[j]:
                dp[i][j] += dp[i-array[j]][j]
    return dp
# [[1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 3, 4, 5]]
print sa([1,2,3,4],4)