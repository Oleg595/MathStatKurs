import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def readSignal(str):
    data = []
    with open(str, 'r') as inf:
        for line in inf.readlines():
            remove_dirst_str = line.replace("[", "")
            remove_next_str = remove_dirst_str.replace("]", "")
            data.append(remove_next_str.split(", "))

    data_float_format = []
    for item in data:
        for x in item:
            data_float_format.append(float(x))
    return data_float_format

def getArray(data, num, size):
    result = np.zeros(size)
    for i in range(size):
        result[i] = data[(num - 1) * size + i]
    return result

def delBlowout(data, size):
    time_result = data
    for j in range(1, 2):
        for i in range(1, size - 1):
            if time_result[i] > (time_result[i + 1] + time_result[i - 1]) / 2:
                time_result[i] = (time_result[i + 1] + time_result[i - 1]) / 2
                j = i - 1
                while time_result[j] > (time_result[j + 1] + time_result[j - 1]) / 2:
                    time_result[j] = (time_result[j + 1] + time_result[j - 1]) / 2
                    j -= 1
    return time_result

def readFile(file_name: str):

    array_size = 1024
    stream_size = 10

    file = open(file_name, 'r')
    s1 = file.readline()
    s1 = s1.replace("STOP Position =  ", "")
    num = int(s1)

    result = {}
    for i in range (array_size):
        s1 = file.readline()
        s1 = s1.replace("\n", " ")
        array = {}
        for j in range(stream_size):
            string = s1[: s1.find(" ")]
            if(j != 0):
                array[j - 1] = float(string)
            s1 = s1.replace(str(string) + '    ', "", 1)
        result[(array_size + i - num) % array_size] = array

    return result

def readLines(string: str):
    num_lines = 100
    txt = '.txt'

    data = {}

    for i in range(num_lines):
        s = string + str(i) + txt
        data[i] = readFile(s)

    return data

def average(arr, size):
    result = 0.

    for i in range(size):
        result += arr[i]

    result /= size

    return result

def averageElement(data, size):
    for i in range (size):
        data[i] = average(data[i], len(data[i]))

    return data

def averageData(data, size):
    for i in range (size):
        data[i] = averageElement(data[i], len(data[i]))

    result = np.zeros(len(data[0]))

    for i in range (len(data[0])):
        for j in range (size):
            result[i] += data[j][i]
        result[i] /= size

    return result

def printData(data, size, color: str):
    x = []
    y = []

    for i in range(size):
        y.append(i)
        x.append(data[i])

    plt.plot(y, x, color)

def readSinus(string: str):
    num_lines = 1000
    txt = '.txt'

    data = {}

    for i in range(num_lines):
        s = string + str(i) + txt
        data[i] = readFile(s)

    return data

def newScale(data, size):
    pos_max = 0.
    neg_max = 0.

    result = {}

    for i in range(size):
        if data[i] > pos_max:
            pos_max = data[i]
        if data[i] < neg_max:
            neg_max = data[i]

    for i in range(size):
        if data[i] > 0.:
            result[i] = data[i] / pos_max
        if data[i] < 0.:
            result[i] = -data[i] / neg_max

    return result

def averageSinusData(data, size):
    for i in range (size):
        data[i] = averageElement(data[i], len(data[i]))

    for i in range (size):
        result = {}
        for j in range (len(data[i])):
            result[j] = data[i][j]
        data[i] = result

    return data

def task(a, size_a, b, size):
    result_A = []
    result_B = []

    for i in range(size):
        result1 = []
        result2 = []
        for j in range(size_a):
            result1.append(a[i][j])
            result2.append(a[i][j])
        result_A.append(result1)
        result_A.append(result2)
        result_B.append(b[i] + 0.015 * math.fabs(b[i]))
        result_B.append(b[i] - 0.015 * math.fabs(b[i]))

    return result_A, result_B

def step(data, size):
    count = 0

    for i in range(size - 1):
        if data[i] < 0 and data[i + 1] > 0:
            count += 1

    return size / count

def posNums(a, b, size, start1, start2):
    aMax, aMin = (b[0] - start1 * a[0][1]) / (a[0][0]), (start2 * a[size - 1][1] - b[size - 1]) / (size / 2 + 3 - a[size - 1][0])

    for i in range(2, size):
        if i % 2 == 0:
            if (b[i] - start1 * a[i][1]) / (a[i][0]) > aMax:
                aMax = (b[i] - start1 * a[i][1]) / (a[i][0])

        else:
            if (-b[size - i] + start2 * a[size - i][1]) / (size / 2 + 3 - a[size - i][0]) > aMin:
                aMin = (-b[size - i] + start2 * a[size - i][1]) / (size / 2 + 2 - a[size - i][0])

    return aMax, aMin, start1, start2 - aMin * (size / 2 + 3)

def negNums(a, b, size, start1, start2):
    aMin = (start2 * a[size - 1][1] - b[size - 1]) / (size / 2 + 3 - a[size - 1][0])
    aMax = (b[0] - start1 * a[0][1]) / (a[0][0])

    for i in range(2, size):
        if i % 2 == 0:
            if (b[i] - start1 * a[i][1]) / (a[i][0]) < aMax:
                aMax = (b[i] - start1 * a[i][1]) / (a[i][0])

        else:
            if (start2 * a[size - i][1] - b[size - i]) / (size / 2 + 3 - a[size - i][0]) < aMin:
                aMin = (start2 * a[size - i][1] - b[size - i]) / (size / 2 + 3 - a[size - i][0])

    return aMax, aMin, start1, start2 - aMin * (size / 2 + 3)

def printLines(start, size, a, b, color: str):
    x = []
    y = []

    for i in range(-5, size + 5):
        x.append(start + i)
        y.append(a * i + b)

    plt.plot(x, y, color)

def scale(data, pos_data, neg_data, pos: bool):
    if pos_data[1] == None or neg_data[1] == None:
        return data

    if pos:
        dist = 0

        for i in range(pos_data[0] + 1, neg_data[0] + 100):
            if data[i] > 0 and data[i - 1] < 0:
                dist += 1

            if dist != 0:
                dist += 1

            if data[i] > 0 and data[i + 1] < 0 and dist != 0:
                break

        if dist == 0:
            return data

        yMax = pos_data[1][0] * dist / 2
        yMin = pos_data[1][1] * dist / 2
        y = (yMax + yMin) / 2
        c = math.pi / (2 * y)
        for i in range(pos_data[0], neg_data[0] + 100):
            data[i] *= c
            if data[i] > 0 and data[i + 1] < 0:
                break

        return data

    else:
        dist = 0

        for i in range(pos_data[0] + 1, neg_data[0] + 100):
            if data[i] < 0 and data[i - 1] > 0:
                dist += 1

            if dist != 0:
                dist += 1

            if data[i] < 0 and data[i + 1] > 0 and dist != 0:
                break

        if dist == 0:
            return data

        yMax = pos_data[1][0] * dist / 2
        yMin = pos_data[1][1] * dist / 2
        y = (yMax + yMin) / 2
        c = -math.pi / (2 * y)
        for i in range(pos_data[0], neg_data[0] + 100):
            data[i] *= c
            if data[i] < 0 and data[i + 1] > 0:
                break

        return data

def partition(data, size):
    pos = True

    if data[0] > data[1]:
        pos = False

    count = 0
    array = {}

    pos_data = {}
    neg_data = {}

    for i in range(0, size - 1):
        if pos and data[i] > data[i + 1]:
            array[count] = data[i]
            pos_data[0] = i - count
            pos_data[1] = USLAU(i - count, array, count + 1, pos)
            if len(neg_data) != 0:
                scale(data, neg_data, pos_data, not pos)
                neg_data.clear()
            array.clear()
            count = 0
            pos = False
        if not pos and data[i] < data[i + 1]:
            array[count] = data[i]
            neg_data[0] = i - count
            neg_data[1] = USLAU(i - count, array, count + 1, pos)
            if len(pos_data) != 0:
                scale(data, pos_data, neg_data, not pos)
                pos_data.clear()
            array.clear()
            count = 0
            pos = True

        if pos and data[i] < data[i + 1]:
            array[count] = data[i]
            count += 1
        if not pos and data[i] > data[i + 1]:
            array[count] = data[i]
            count += 1

    printData(data, size, 'b')
    plt.show()

    printTime(data, size)

def printTime(data, size):
    x = np.zeros(size)
    y = np.zeros(size)

    for i in range(size):
        y[i] = data[i] / (2 * math.pi)
        x[i] = 1

    plt.scatter(y, x)
    plt.show()

    max = 0
    min = size
    delta = np.zeros(size - 1)

    for i in range(1, size):
        if max < y[i] - y[i - 1]:
            max = y[i] - y[i - 1]
        if min > y[i] - y[i - 1]:
            min = y[i] - y[i - 1]
        delta[i - 1] = y[i] - y[i - 1]

    sns.distplot(delta)
    plt.xlabel('delta(t[i])')
    plt.ylabel('count')
    plt.show()

def USLAU(start, data, size, pos: bool):
    A = {}
    b = {}

    if size < 7:
        return

    for i in range(3, size - 3):
        A[i - 3] = [i, 1]
        b[i - 3] = data[i]

    A, b = task(A, len(A[0]), b, len(A))

    if pos:
        aMax, aMin, bMax, bMin = posNums(A, b, len(A), data[2], data[size - 2])
    else:
        aMax, aMin, bMax, bMin = negNums(A, b, len(A), data[2], data[size - 2])

    #printLines(start, size, aMax, bMax, 'g')
    #printLines(start, size, aMin, bMin, 'r')

    return [aMax, aMin, bMax, bMin]

main_lexem = 'Bazenov'

lexem1 = '\Sin_100MHz\sin_100MHz_'
lexem2 = '\ZeroLine\ZeroLine_'
lexem3 = '\-0_25V\-0_25V_'
lexem4 = '\-0_5V\-0_5V_'
lexem5 = '\+0_25V\+0_25V_'
lexem6 = '\+0_5V\+0_5V_'

color1 = 'b'
color2 = 'g'
color3 = 'r'
color4 = 'c'
color5 = 'm'
color6 = 'k'

data1 = readFile(main_lexem + lexem1 + "0.txt")
data2 = readLines(main_lexem + lexem2)
data3 = readLines(main_lexem + lexem3)
data4 = readLines(main_lexem + lexem4)
data5 = readLines(main_lexem + lexem5)
data6 = readLines(main_lexem + lexem6)

data1 = averageElement(data1, len(data1))
data2 = averageData(data2, len(data2))
data3 = averageData(data3, len(data3))
data4 = averageData(data4, len(data4))
data5 = averageData(data5, len(data5))
data6 = averageData(data6, len(data6))

printData(data2, len(data2), color2)
printData(data3, len(data3), color3)
printData(data4, len(data4), color4)
printData(data5, len(data5), color5)
printData(data6, len(data6), color6)

plt.show()

printData(data1, len(data1), color1)
printData(data2, len(data2), color2)
printData(data3, len(data3), color3)
printData(data4, len(data4), color4)
printData(data5, len(data5), color5)
printData(data6, len(data6), color6)

plt.show()

data1 = delBlowout(data1, len(data1))

printData(data1, len(data1), color1)
plt.show()

#data1 = readFile(main_lexem + lexem1 + "0.txt")
#data1 = averageElement(data1, len(data1))

data1 = newScale(data1, len(data1))

printData(data1, len(data1), color1)
plt.show()

partition(data1, len(data1))
