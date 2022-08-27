import random

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


cpu_types = []
cpu_powers = []
with open('cpu_test_31757.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split('\t')
        end = arr[-1]
        end = end[0:-1]
        if is_number(end):
            cpu_types.append(arr[0])
            cpu_powers.append(float(end))
cpu_powers, cpu_types = (list(t) for t in zip(*sorted(zip(cpu_powers, cpu_types))))
cpu_len = len(cpu_powers)
print(cpu_len)

def cpu_min(number):
    if number>=cpu_len:
        return False, False
    return cpu_powers[0:number], cpu_types[0:number]


def cpu_window(start, end):
    if end<=cpu_len:
        return cpu_powers[start:end], cpu_powers[start:end]
    return False, False


def cpu_random(number=10):
    total = random.randint(2,number)
    tmp_powers=[]
    tmp_types=[]
    for i in range(total):
        idx = random.randint(0,cpu_len-1)
        tmp_powers.append(cpu_powers[idx])
        tmp_types.append(cpu_types[idx])
    return tmp_powers,tmp_types,total


