from collections import defaultdict

d = defaultdict(tuple)

d[(1, 2)] = 3
d[(2, 3)] = 4
d[(1, 2)] = 5
d[(1, 2)] = 6


print(d)
print(d[(1, 2)])

idx = (1, 2)

if d[idx] == 6:
    print('yes')