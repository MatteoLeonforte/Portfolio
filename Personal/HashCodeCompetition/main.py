

fin=open('a_example.txt','r')
l=[]
listaefficienza=[]
efficienzaordinata=[]
a=0
for riga in fin:
    riga=riga.strip().split()
    l.append(riga)
totLibri=int(l[0][0])
totLibrerie=int(l[0][1])
totGiorni=int(l[0][2])
listaPunteggi=l[1]
l2=l.copy()
l2.remove(l2[0])
l2.remove(l2[0]) #lista delle librerie
listautili=[]

a=0
while a<len(l2):
    punteggio = 0
    libri= 0
    for elem in l2[a+1]:
        if elem not in listautili:
            punteggio+=int(l[1][int(elem)])
            libri+=1
            listautili.append(elem)
    sommatempi=int(l2[a][1])+((libri//int(l2[a][2])+1)) #tempo di sign up + libri/tempo di libri scann al giorno
    efficienza= punteggio/sommatempi
    listaefficienza.append(efficienza)
    a+=2

for numero in listaefficienza:
    i=listaefficienza.index(max(listaefficienza))
    efficienzaordinata.append(i)
    listaefficienza[i]=0
fin.close()

####
file = open('a_example.txt','r')
informazioni=file.readline().strip().split()
#Trasformzaione della prima riga in intero
for i in range(len(informazioni)):
    informazioni[i] = int(informazioni[i])


num_librerie = informazioni[1]
dizionarioLibrerie = {}
for iD in range(num_librerie):
    dizionarioLibrerie[iD]= []
    descrizioneLibreria = file.readline().strip().split()
    for i in range(len(descrizioneLibreria)):
        descrizioneLibreria[i]=int(descrizioneLibreria[i])
    dizionarioLibrerie[iD].append(descrizioneLibreria[1])
    tempoStampa = (descrizioneLibreria[0]//descrizioneLibreria[2]) + 1
    dizionarioLibrerie[iD].append(tempoStampa)
    insiemeLibriLibreria = file.readline().strip().split()
    for i in range(len(insiemeLibriLibreria)):
        insiemeLibriLibreria[i] = int(insiemeLibriLibreria[i])
    insiemeLibriOrdinati = []
    dizionarioLibrerie[iD].append(insiemeLibriLibreria)
    libriGiorno = descrizioneLibreria[2]
    dizionarioLibrerie[iD].append(libriGiorno)


librerieRegistrate={}
libriScansionati=[]


tempoTrascorso = 0

i=0
while tempoTrascorso<=totGiorni:
    
    ID = efficienzaordinata[i]
    librerieRegistrate[ID]=[]

    tempoTrascorso+=dizionarioLibrerie[ID][0]

    tempoRimanente=totGiorni-tempoTrascorso
    tempoStampa=dizionarioLibrerie[ID][1]

    listaLibri=dizionarioLibrerie[ID][2]
    
    tempoRelativo=tempoTrascorso
    libri_alGiorno=dizionarioLibrerie[ID][3]
    j=0
    while j<len(listaLibri) and tempoRelativo<=totGiorni:
        libro=listaLibri[j]
        
        if libro not in libriScansionati:
            libriScansionati.append(libro)
            librerieRegistrate[ID].append(libro)
            if (j/libri_alGiorno) is int:
                tempoRelativo+=1
        j+=1
fileOutput=open('fileOutput.txt','w')
fileOutput.write(str(len(librerieRegistrate)))
fileOutput.write('\n')

for libreria in librerieRegistrate:
    
    K=librerieRegistrate[libreria]
    riga1=str(libreria)+' '+str(len(K))
    fileOutput.write(riga1)
    fileOutput.write('\n')
    riga2=''
    for x in K:
        riga2=riga2+str(x)+' '
    riga2=riga2+'\b'

    fileOutput.write(riga2)
fileOutput.close()
