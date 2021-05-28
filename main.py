import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tolsolvty.src.tolsolvty import tolsolvty

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

    plt.xlabel("номер измерения")
    plt.ylabel("значение масштабированного сигнала")
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

def getId(val, consts, size):
    if val < consts[0]:
        return 0

    if val > consts[size - 1]:
        return size - 2

    for i in range(size):
        if val > consts[i] and val < consts[i + 1]:
            return i

def getInterpolation(x, y, x_val):
    return y[0] + (x_val - x[0]) / (x[1] - x[0]) * (y[1] - y[0])

def interpolation(data, dc, constants, size, num_const):
    result = np.zeros(size)

    for i in range(size):
        id = getId(data[i], constants[:, i], num_const)
        result[i] = getInterpolation([constants[id, i], constants[id + 1, i]], [dc[id], dc[id + 1]], data[i])

    return result

def printTime(data, size):
    x = np.zeros(size)
    y = np.zeros(size)

    for i in range(size):
        y[i] = data[i] / (2 * math.pi)
        x[i] = 1

    plt.ylabel("номер измерения")
    plt.xlabel('t[i] = (y[i] - y[i - 1]) / (2 * pi * teta)')
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
    plt.xlabel('t[i] = (y[i] - y[i - 1]) / (2 * pi * teta)')
    plt.ylabel('количество точек')
    plt.show()

def get_asin_amp(bin_val, ids):
     dy = 0.005
     di = 1 / 3
     A2_bot = np.zeros((len(ids[1]), 3))
     A2_top = np.zeros((len(ids[1]), 3))
     B2_bot = np.zeros((len(ids[1]), 1))
     B2_top = np.zeros((len(ids[1]), 1))

     A1_bot = np.zeros((len(ids[0]), 3))
     A1_top = np.zeros((len(ids[0]), 3))
     B1_bot = np.zeros((len(ids[0]), 1))
     B1_top = np.zeros((len(ids[0]), 1))

     count = 0

     for i in range(len(ids[0])):
         if i != 0 and ids[0][i] - ids[0][i - 1] > 2:
            count += 1

         A1_bot[i, 0] = ids[0][i] - di + 1
         A1_bot[i, 1] = 1
         A1_bot[i, 2] = count
         B1_bot[i, 0] = bin_val[ids[0][i]] - dy * abs(bin_val[ids[0][i]])

         A1_top[i, 0] = ids[0][i] + di + 1
         A1_top[i, 1] = 1
         A1_top[i, 2] = count
         B1_top[i, 0] = bin_val[ids[0][i]] + dy * abs(bin_val[ids[0][i]])

     count = 0

     for i in range(len(ids[1])):
         if i != 0 and ids[1][i] - ids[1][i - 1] > 2:
            count += 1

         A2_bot[i, 0] = ids[1][i] - di
         A2_bot[i, 1] = 1
         A2_bot[i, 2] = count
         B2_bot[i, 0] = bin_val[ids[1][i]] - dy * abs(bin_val[ids[1][i]])

         A2_top[i, 0] = ids[1][i] + di
         A2_top[i, 1] = 1
         A2_top[i, 2] = count
         B2_top[i, 0] = bin_val[ids[1][i]] + dy * abs(bin_val[ids[1][i]])

     [tolmax, argmax, envs, ccode] = tolsolvty(A1_bot, A1_top, B1_bot, B1_top)
     a1 = argmax[0]
     b1 = argmax[1]
     [tolmax, argmax, envs, ccode] = tolsolvty(A2_bot, A2_top, B2_bot, B2_top)
     a2 = argmax[0]
     b2 = argmax[1]
     y = abs((b1*a2-b2*a1)/(a2-a1))

     return [y, a1, b1, a2, b2]

def partition(data, size):

    elements = [[], []]

    is_pos = True

    if data[0] < data[1]:
        is_pos = True
        elements[0].append(0)
    else:
        is_pos = False
        elements[1].append(0)

    for i in range(1, size):
        if data[i] >= data[i - 1]:
            elements[0].append(i)
        else:
            elements[1].append(i)

    return elements

def printData_Lines(data, a1, b1, a2, b2):
    x = np.zeros(40)
    y1 = np.zeros(40)
    y2 = np.zeros(40)

    time_data = []

    for i in range(40):
        time_data.append(data[i])
        x[i] = i
        y1[i] = a1 * (x[i] + 5) + b1
        y2[i] = a2 * (x[i] + 4) + b2

    plt.xlabel("номер измерения")
    plt.ylabel("амплитуда сигнала")
    plt.plot(x[: 40], time_data[: 40], 'y')
    plt.plot(x[: 40], y1, 'g')
    plt.plot(x[: 40], y2, 'r')

def scale(data, size, ampl):
    result = []

    coef = math.pi / (2 * ampl)

    for i in range(size):
        result.append(data[i] * coef)

    return result

main_lexem = 'data'

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

constants = np.array([data4, data3, data2, data5, data6])
dc = [-0.5, -0.25, 0.0, 0.25, 0.5]
data1 = interpolation(data1, dc, constants, len(data1), len(constants))

printData(data1, len(data1), color1)
plt.show()

data1 = newScale(data1, len(data1))

printData(data1, len(data1), color1)
plt.show()

elements = partition(data1, len(data1))
[ampl, a1, b1, a2, b2] = get_asin_amp(data1, elements)

printData_Lines(data1, a1, b1, a2, b2)
plt.show()

data1 = scale(data1, len(data1), ampl)

printData(data1, len(data1), 'g')
plt.show()

printTime(data1, len(data1))
plt.show()
