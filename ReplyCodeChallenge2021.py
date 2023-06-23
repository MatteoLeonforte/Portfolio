import math
class Palazzo:

    def __init__(self, x = 0,y = 0,latenza = 0, speed = 0):
        self.pos = [x,y]
        self.latenza = latenza
        self.speed = speed

    def __eq__(self,y) :
        return self.speed == y.speed

    def __ne__(self,y) :
        return self.speed != y.speed

    def __et__(self,y) :
        return self.speed < y.speed

    def __le__(self,y) :
        return self.speed <= y.speed

    def __gt__(self,y) :
        return self.speed > y.speed

    def __ge__(self,y) :
        return self.speed >= y.speed

class Antenna:
    def __init__(self,id = 0,raggio = 0,speed = 0):
        self.id = id
        self.raggio = raggio
        self.speed = speed

    def __eq__(self,y) :
        return self.raggio == y.raggio

    def __ne__(self,y) :
        return self.raggio != y.raggio

    def __et__(self,y) :
        return self.raggio < y.raggio

    def __le__(self,y) :
        return self.raggio <= y.raggio

    def __gt__(self,y) :
        return self.raggio > y.raggio

    def __ge__(self,y) :
        return self.raggio >= y.raggio
    
def leggiFile(file):
    f = open(file, 'r', encoding='UTF-8')
    global lista_antenne
    global lista_palazzi
    global numPalazzi
    global numAntenne
    global larghezzaGriglia
    global altezzaGriglia
    global reward
    global nomeFile
    global listaTuple
    global matriceBooleana 
    listaTuple = []
    lista_antenne = []
    lista_palazzi = []
    numPalazzi = 0
    numAntenne = 0
    larghezzaGriglia = 0
    altezzaGriglia = 0
    reward = 0
    nomeFile = ''
    nomeFile = file.split('_')[2]
    print(nomeFile)
    larghezzaGriglia, altezzaGriglia = map(int, f.readline().strip().split())
    numPalazzi, numAntenne, reward = map(int, f.readline().strip().split())
    for n in range(numPalazzi):
        p = Palazzo()
        p.pos[0], p.pos[1], p.latenza, p.speed = map(int, f.readline().strip().split())

        lista_palazzi.append(p)
    index = 0
    for m in range(numAntenne):
        a = Antenna()
        a.raggio, a.speed = map(int, f.readline().strip().split())
        a.id = index
        index += 1
        lista_antenne.append(a)
    matriceBooleana = [[False for i in range(larghezzaGriglia)] for k in range(altezzaGriglia)]
    lista_antenne.sort(reverse=True)
    lista_palazzi.sort(reverse=True)

def generaCopertura(y,x,r):
    global matriceBooleana
    global larghezzaGriglia
    global altezzaGriglia
    i = x-r
    while i<x+r and i<altezzaGriglia:
        if i<0:
            i+=1
            continue
        j = y-r
        while j<y+r and j<larghezzaGriglia:
            if j<0:
                j+=1
                continue
            if abs(x-i)+(y-j)<r:
                matriceBooleana[i][j] = True
            j+=1
        i+=1
    
def checkMatrice(y,x,r):
    global matriceBooleana
    global larghezzaGriglia
    global altezzaGriglia
    i = x-r
    while i<x+r and i<altezzaGriglia:
        if i<0:
            i+=1
            continue
        j = y-r
        while j<y+r and j<larghezzaGriglia:
            if j<0:
                j+=1
                continue
            if matriceBooleana[i][j] == True:
                return False
            j+=1
        i+=1
    return True

def troppoEsterno(p,a):
    global altezzaGriglia
    global larghezzaGriglia
    x = p.pos[0]
    y = p.pos[1]
    r = a.raggio
    if (x+r>larghezzaGriglia) or (x-r<0) or (y+r>altezzaGriglia) or (y-r<0):
        return True
    return False
    

def generaTuple():
    global lista_palazzi
    global lista_antenne
    global listaTuple
    global numPalazzi
    global numAntenne
    global matriceBooleana
    i=0
    while(i<numPalazzi and numAntenne>0):
        if (checkMatrice(lista_palazzi[i].pos[1], lista_palazzi[i].pos[0], lista_antenne[0].raggio) == True) or (troppoEsterno(lista_palazzi[i],lista_antenne[0]) == True):
            i+=1
            continue
        tupla = (lista_palazzi[i],lista_antenne[0])
        listaTuple.append(tupla)
        generaCopertura(lista_palazzi[i].pos[1], lista_palazzi[i].pos[0], lista_antenne[0].raggio)
        lista_palazzi.pop(i)
        lista_antenne.pop(0)
        numAntenne -= 1
        numPalazzi -= 1
    i=0
    j=0
    while (i<numPalazzi and j<numAntenne):
        tupla = (lista_palazzi[i],lista_antenne[j])
        listaTuple.append(tupla)
        i+=1
        j+=1
    return

def generaOutput():
    global nomeFile
    global listaTuple
    f = open(nomeFile+".out",'w',encoding = 'UTF-8')
    f.write(str(len(listaTuple)) + "\n")
    for tupla in listaTuple:
        f.write(str(tupla[1].id) + " " + str(tupla[0].pos[0]) + " " + str(tupla[0].pos[1]) + "\n")
    f.close()
    return

def svolgi(file):
    leggiFile(file)
    generaTuple()
    generaOutput()

lista_antenne = []
lista_palazzi = []
numPalazzi = 0
numAntenne = 0
larghezzaGriglia = 0
altezzaGriglia = 0
reward = 0
nomeFile = ''
listaTuple = []
matrice = [[]]


svolgi('data_scenarios_a_example.in')
svolgi('data_scenarios_b_mumbai.in')
svolgi('data_scenarios_c_metropolis.in')
svolgi('data_scenarios_d_polynesia.in')
svolgi('data_scenarios_e_sanfrancisco.in')
svolgi('data_scenarios_f_tokyo.in')

