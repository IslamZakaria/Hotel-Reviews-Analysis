import pandas as pd
def tagsencoder (TY):
    data = TY
    biglist = set()

    for i in data["Tags"].values:
        i = i.replace("'", '')
        i = i.replace('[', '')
        i = i.replace(']', '')
        i = i.replace(' ', '')
        b = i.split(',')
        for h in b:
            biglist.add(h)

    lis = []
    for o in biglist:
        lis.append(o)

    lioflo = [[]]

    for ii in data["Tags"].values:
        ii = ii.replace("'", '')
        ii = ii.replace('[', '')
        ii = ii.replace(']', '')
        ii = ii.replace(' ', '')
        bb = ii.split(',')
        curli = []
        for h in lis:
            bo = 0
            for ind in bb:
                if h == ind:
                    bo = 1
            curli.append(bo)
        lioflo.append(curli)
    lioflo.remove(lioflo[0])
    Tlioflo = zip(*lioflo)
    cou = 0
    for l in Tlioflo:
        data[lis[cou]] = l
        cou += 1

    return data



