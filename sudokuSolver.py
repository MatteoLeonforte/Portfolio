#VARIABILI
import time
dizionarioQuadranti={1:[(1,3),(1,3)],
                     2:[(1,3),(4,6)],
                     3:[(1,3),(7,9)],
                     4:[(4,6),(1,3)],
                     5:[(4,6),(4,6)],
                     6:[(4,6),(7,9)],
                     7:[(7,9),(1,3)],
                     8:[(7,9),(4,6)],
                     9:[(7,9),(7,9)]}



#FUNZIONI AUSILIARIE
    

def converti_in_matrice(fileInput):
    fileInput=open(fileInput,'r')
    file=fileInput.readlines()
    fileInput.close()
    matrice=[]
    for riga in file:
        riga=riga.strip().split(' ')
        for k in range(len(riga)):
            if riga[k] in '123456789':
                riga[k]=int(riga[k])
        matrice.append(riga)
    return matrice



    
def rimuovi_caselleVuote(sudoku):
    
    for i in range(len(sudoku)):
        for j in range(len(sudoku[i])):
            if sudoku[i][j]=='*':
                sudoku[i][j]=[1,2,3,4,5,6,7,8,9]
                caselle_vuote.append((i,j))
            else:
                caselle_piene.append((i,j))
    return sudoku




def semplifica_sudoku(i,j,numero,sudoku):
    
    sudoku=semplifica_riga(i,numero,sudoku)
    sudoku=semplifica_colonna(j,numero,sudoku)
    quadrante=verifica_Quadrante(i,j)
    sudoku=semplifica_quadrante(quadrante,numero,sudoku)
    return sudoku

def semplifica_riga(i,numero,sudoku):
    
    for casella in sudoku[i]:
        if type(casella)==list and numero in casella:
            casella.remove(numero)
    return sudoku

def semplifica_colonna(j,numero,sudoku):

    for riga in sudoku:
        casella=riga[j]
        if type(casella)==list and numero in casella:
            casella.remove(numero)
    return sudoku

def semplifica_quadrante(quadrante,numero,sudoku):

    caselleQuadrante = elenca_caselleQuadrante(quadrante)
    for casella in caselleQuadrante:
        if casella in caselle_vuote:
            casellaSudoku=sudoku[casella[0]][casella[1]]
            if numero in casellaSudoku:
                casellaSudoku.remove(numero)
    return sudoku



        
    
def elenca_caselleQuadrante(quadrante):
    
    caselleQuadrante=[]
    
    intervallo_righe = dizionarioQuadranti[quadrante][0]
    intervallo_colonne = dizionarioQuadranti[quadrante][1]
    
    for riga in range(intervallo_righe[0]-1,intervallo_righe[1]):
        for colonna in range(intervallo_colonne[0]-1,intervallo_colonne[1]):
            caselleQuadrante.append((riga,colonna))
    return caselleQuadrante




def verifica_Quadrante(i,j):

    for quadrante in dizionarioQuadranti:
        if dizionarioQuadranti[quadrante][0][0] <=i+1<= dizionarioQuadranti[quadrante][0][1] and \
           dizionarioQuadranti[quadrante][1][0] <=j+1<= dizionarioQuadranti[quadrante][1][1]:
            return quadrante


        
def risolvi_incrociandoRiga(i,numero,sudoku):
    contaNumero=0
    for j in range(len(sudoku[i])):
        if (i,j)in caselle_vuote:
            if numero in sudoku[i][j]:
                contaNumero+=1
                x=i
                y=j
                
    if contaNumero==1:
        sudoku[x][y]=numero
        caselle_vuote.remove((x,y))
        caselle_piene.append((x,y))
        sudoku = semplifica_sudoku(x,y,numero,sudoku)
    return sudoku

def risolvi_incrociandoColonna(j,numero,sudoku):
    contaNumero=0
    for i in range(len(sudoku)):
        if (i,j) in caselle_vuote:
            if numero in sudoku[i][j]:
                contaNumero+=1
                x=i
                y=j
    if contaNumero==1:
        sudoku[x][y]=numero
        caselle_vuote.remove((x,y))
        caselle_piene.append((x,y))
        sudoku = semplifica_sudoku(x,y,numero,sudoku)
    return sudoku
        
def risolvi_incrociandoQuadrante(quadrante,numero,sudoku):
    caselleQuadrante = elenca_caselleQuadrante(quadrante)
    contaNumero=0
    for casella in caselleQuadrante:
        if casella in caselle_vuote:
            i=casella[0]
            j=casella[1]
            if numero in sudoku[i][j]:
                contaNumero+=1
                x=i
                y=j
    if contaNumero==1:
        sudoku[x][y]=numero
        caselle_vuote.remove((x,y))
        caselle_piene.append((x,y))
        sudoku = semplifica_sudoku(x,y,numero,sudoku)
    return sudoku
                
    
                


def converti_valoriSingoli(sudoku):
    for i in range(len(sudoku)):
        for j in range(len(sudoku[i])):
            casella=sudoku[i][j]
            if type(casella)==list and len(casella)==1:
                sudoku[i][j]=int(casella[0])
                caselle_vuote.remove((i,j))
                caselle_piene.append((i,j))
    return sudoku
                    
        


def check_Sudoku(sudoku):
    for riga in sudoku:
            for casella in riga:
                if not(type(casella)==int):
                    return False
    return True
                
    





                
#PROGRAMMA PRINCIPALE
    
def risolvi_Sudoku(file):
    
    global caselle_vuote, caselle_piene
    caselle_vuote = []
    caselle_piene = []
    
    
    #Assegno lista [1-9] a caselle vuote
    sudoku = converti_in_matrice(file)
    sudoku = rimuovi_caselleVuote(sudoku)
            
    sudokuRisolto=False
    while sudokuRisolto==False:            
       
        #Per ogni casella piena semplifico riga/colonna/quadrante
        for casellaPiena in caselle_piene:
            i=casellaPiena[0]
            j=casellaPiena[1]
            numero=sudoku[i][j]

            sudoku = semplifica_sudoku(i,j,numero,sudoku)
            
        #Risolvo incorciando: vedo se numero compare una sola volta nelle liste in riga/colonna/quadrante
        for numero in range(1,10):
            
            for i in range(0,9):
                sudoku=risolvi_incrociandoRiga(i,numero,sudoku)
            for j in range(0,9):
                sudoku=risolvi_incrociandoColonna(j,numero,sudoku)
            for quadrante in range(1,10):
                sudoku=risolvi_incrociandoQuadrante(quadrante,numero,sudoku)
                
        #Controllo se il sudoku è risolto       
        sudokuRisolto = check_Sudoku(sudoku)

    return sudoku


#TESTER
def stampa(sudoku):
    print('')
    for riga in sudoku:
        print (riga)
    print('')
        
def tester(fileSbagliato,fileCorretto):
    risultatoTester='NEGATIVO'
    matrice_sudokuRisolto = risolvi_Sudoku(fileSbagliato)
    matrice_sudokuCorretto = converti_in_matrice(fileCorretto)
    if matrice_sudokuRisolto==matrice_sudokuCorretto:
        risultatoTester='POSITIVO'
    #print( risultatoTester)
    for riga in matrice_sudokuRisolto:
        print (riga)
    

        
################################################################################################

tester('sudoku_nonRisoltoDifficile.txt','sudoku_Risolto.txt')
print ('FINITO')


