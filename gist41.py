import math

def gcd(a, b):
    if b == 0:
        return abs(a)
    else:
        return abs(gcd(b, a % b))

def answer(dimensions, captain_position, badguy_position, _distance):
    slopes = {}
    map = []

    distance = _distance

    captured = {}
    vectors = {}

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
        slopes[(deltaY, deltaX)] = deltaGCD # put each slope (simplified deltay/deltax:gcd) into {slopes}
        vectors[(deltaY, deltaX)] = 1       # put each simplified delta into {vectors}


    fillH(x, y, hExtend, dimensions, True, x, y, bx, by, distance, slopes, map)
    fillV(x, y, hExtend, wExtend, dimensions, True, x, y, bx, by, distance, slopes, map)

    fillH(badguy_position[0], badguy_position[1], hExtend, dimensions, False, x, y, bx, by, distance, slopes, map)
    fillV(badguy_position[0], badguy_position[1], hExtend, wExtend, dimensions, False, x, y, bx, by, distance, slopes, map)


    for i in range(len(map)):
        m = map[i]
        if (m[0], m[1]) not in captured:
            deltaY = m[1] - y
            deltaX = m[0] - x
            deltaGCD = gcd(deltaX, deltaY)
            deltaX /= deltaGCD
            deltaY /= deltaGCD

            if (deltaY, deltaX) not in vectors:
                if (deltaY, deltaX) not in slopes:
                    count += 1
                    vectors[(deltaY, deltaX)] = 1
                else:
                    slopeMultiply = slopes[(deltaY, deltaX)]
                    if deltaGCD < slopeMultiply:
                        count += 1
                        vectors[(deltaY, deltaX)] = 1

            captured[(m[0], m[1])] = 1

    return count


def addToMap(currentX, currentY, x, y, bx, by, distance, slopes, map):
    deltaY = currentY - y
    deltaX = currentX - x

    target = [currentX, currentY]

    deltaD = math.sqrt(math.pow(deltaY, 2) + math.pow(deltaX, 2))

    if distance - deltaD >= 0:
        map.append(target)


def addSlope(currentX, currentY, x, y, bx, by, distance, slopes, map):
    deltaY = currentY - y
    deltaX = currentX - x
    deltaGCD = gcd(deltaX, deltaY)

    deltaD = math.sqrt(math.pow(deltaY, 2) + math.pow(deltaX, 2))

    deltaX /= deltaGCD
    deltaY /= deltaGCD

    if distance - deltaD >= 0:
        if (deltaY, deltaX) not in slopes:
            slopes[(deltaY, deltaX)] = deltaGCD

        elif deltaGCD < slopes[(deltaY, deltaX)]:
            slopes[(deltaY, deltaX)] = deltaGCD


def fillV(_currentX, currentY, hExtend, wExtend, dimensions, isHero, x, y, bx, by, distance, slopes, map):
    if isHero:
        rightMargin = dimensions[0] - x
        leftMargin = x
    else:
        rightMargin = dimensions[0] - bx
        leftMargin = bx

    currentX = _currentX

    for i in range(1, wExtend+1):
        currentX += rightMargin * 2
        rightMargin = dimensions[0] - rightMargin

        if isHero:
            addSlope(currentX, currentY, x, y, bx, by, distance, slopes, map)
        else:
            addToMap(currentX, currentY, x, y, bx, by, distance, slopes, map)

        fillH(currentX, currentY, hExtend, dimensions, isHero, x, y, bx, by, distance, slopes, map)


    currentX = _currentX

    for i in range(1, wExtend+1):
        currentX -= leftMargin * 2
        leftMargin = dimensions[0] - leftMargin

        if isHero:
            addSlope(currentX, currentY, x, y, bx, by, distance, slopes, map)
        else:
            addToMap(currentX, currentY, x, y, bx, by, distance, slopes, map)

        fillH(currentX, currentY, hExtend, dimensions, isHero, x, y, bx, by, distance, slopes, map)


def fillH(currentX, _currentY, hExtend, dimensions, isHero, x, y, bx, by, distance, slopes, map):
    if isHero:
        topMargin = dimensions[1] - y
        downMargin = y
    else:
        topMargin = dimensions[1] - by
        downMargin = by

    currentY = _currentY

    for i in range(1, hExtend+1):
        currentY += topMargin * 2
        topMargin = dimensions[1] - topMargin

        if isHero:
            addSlope(currentX, currentY, x, y, bx, by, distance, slopes, map)
        else:
            addToMap(currentX, currentY, x, y, bx, by, distance, slopes, map)


    currentY = _currentY

    for i in range(1, hExtend+1):
        currentY -= downMargin * 2
        downMargin = dimensions[1] - downMargin

        if isHero:
            addSlope(currentX, currentY, x, y, bx, by, distance, slopes, map)
        else:
            addToMap(currentX, currentY, x, y, bx, by, distance, slopes, map)


print answer([3, 2],[1, 1],[2, 1],4)