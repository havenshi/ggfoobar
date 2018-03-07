import math

def gcd(a, b):
    if b == 0:
        return abs(a)
    else:
        return abs(gcd(b, a % b))

def answer(dimensions, captain_position, badguy_position, _distance):
    slope = {}
    array = []
    captured = {}
    vectors = {}

    distance = _distance
    wExtend = 1 + ((distance + captain_position[0]) / dimensions[0]) # largest width x can extend to
    hExtend = 1 + ((distance + captain_position[1]) / dimensions[1]) # largest height y can extend to

    x = captain_position[0]
    y = captain_position[1]
    bx = badguy_position[0]
    by = badguy_position[1]

    deltaY = by - y
    deltaX = bx - x
    deltaGCD = gcd(deltaX, deltaY)
    deltaD = math.sqrt(pow(deltaY, 2) + pow(deltaX, 2)) # current distance between me and guard
    deltaY /= deltaGCD
    deltaX /= deltaGCD

    count = 0
    if distance - deltaD >= 0:
        count = 1
        captured[(bx, by)] = 1 # put guard into {captured}
        slope[(deltaY, deltaX)] = deltaGCD # put each slope (simplified deltay/deltax:gcd) into {slope}
        vectors[(deltaY, deltaX)] = 1       # put each simplified delta into {vectors}


    H(x, y, hExtend, dimensions, True, x, y, bx, by, distance, slope, array)
    V(x, y, hExtend, wExtend, dimensions, True, x, y, bx, by, distance, slope, array)

    H(badguy_position[0], badguy_position[1], hExtend, dimensions, False, x, y, bx, by, distance, slope, array)
    V(badguy_position[0], badguy_position[1], hExtend, wExtend, dimensions, False, x, y, bx, by, distance, slope, array)


    for i in range(len(array)):
        m = array[i]
        if (m[0], m[1]) not in captured:
            deltaY = m[1] - y
            deltaX = m[0] - x
            deltaGCD = gcd(deltaX, deltaY)
            deltaX /= deltaGCD
            deltaY /= deltaGCD

            if (deltaY, deltaX) not in vectors:
                if (deltaY, deltaX) not in slope:
                    count += 1
                    vectors[(deltaY, deltaX)] = 1
                else:
                    slopeMultiply = slope[(deltaY, deltaX)]
                    if deltaGCD < slopeMultiply:
                        count += 1
                        vectors[(deltaY, deltaX)] = 1
            captured[(m[0], m[1])] = 1

    return count


def addToArray(currentX, currentY, x, y, bx, by, distance, slope, array):
    deltaY = currentY - y
    deltaX = currentX - x
    target = [currentX, currentY]
    deltaD = math.sqrt(math.pow(deltaY, 2) + math.pow(deltaX, 2))

    if distance - deltaD >= 0:
        array.append(target)


def addSlope(currentX, currentY, x, y, bx, by, distance, slope, array):
    deltaY = currentY - y
    deltaX = currentX - x
    deltaGCD = gcd(deltaX, deltaY)

    deltaD = math.sqrt(math.pow(deltaY, 2) + math.pow(deltaX, 2))
    deltaX /= deltaGCD
    deltaY /= deltaGCD

    if distance - deltaD >= 0:
        if (deltaY, deltaX) not in slope:
            slope[(deltaY, deltaX)] = deltaGCD
        elif deltaGCD < slope[(deltaY, deltaX)]:
            slope[(deltaY, deltaX)] = deltaGCD


def V(_currentX, currentY, hExtend, wExtend, dimensions, flag, x, y, bx, by, distance, slope, array):
    if flag:
        rightMargin = dimensions[0] - x
        leftMargin = x
    else:
        rightMargin = dimensions[0] - bx
        leftMargin = bx

    currentX = _currentX

    for i in range(1, wExtend+1):
        currentX += rightMargin * 2
        rightMargin = dimensions[0] - rightMargin

        if flag:
            addSlope(currentX, currentY, x, y, bx, by, distance, slope, array)
        else:
            addToArray(currentX, currentY, x, y, bx, by, distance, slope, array)
        H(currentX, currentY, hExtend, dimensions, flag, x, y, bx, by, distance, slope, array)

    currentX = _currentX

    for i in range(1, wExtend+1):
        currentX -= leftMargin * 2
        leftMargin = dimensions[0] - leftMargin

        if flag:
            addSlope(currentX, currentY, x, y, bx, by, distance, slope, array)
        else:
            addToArray(currentX, currentY, x, y, bx, by, distance, slope, array)
        H(currentX, currentY, hExtend, dimensions, flag, x, y, bx, by, distance, slope, array)


def H(currentX, _currentY, hExtend, dimensions, flag, x, y, bx, by, distance, slope, array):
    if flag:
        topMargin = dimensions[1] - y
        downMargin = y
    else:
        topMargin = dimensions[1] - by
        downMargin = by

    currentY = _currentY

    for i in range(1, hExtend+1):
        currentY += topMargin * 2
        topMargin = dimensions[1] - topMargin
        if flag:
            addSlope(currentX, currentY, x, y, bx, by, distance, slope, array)
        else:
            addToArray(currentX, currentY, x, y, bx, by, distance, slope, array)

    currentY = _currentY

    for i in range(1, hExtend+1):
        currentY -= downMargin * 2
        downMargin = dimensions[1] - downMargin

        if flag:
            addSlope(currentX, currentY, x, y, bx, by, distance, slope, array)
        else:
            addToArray(currentX, currentY, x, y, bx, by, distance, slope, array)


print answer([3, 2],[1, 1],[2, 1],4)