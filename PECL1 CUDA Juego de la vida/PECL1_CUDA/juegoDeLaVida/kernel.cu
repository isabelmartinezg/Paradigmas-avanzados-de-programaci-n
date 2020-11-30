#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Windows.h>
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <string>
#include <stdexcept>
#include "../common/book.h"
using namespace std;

#define viva 'X'
#define muerta ' '

__global__ void kernelMemoriaCompartida(char* Ad, char* Bd, int numColumnas, int numFilas, int TILE_WIDTH)
{
    //Definimos la matriz de memoria compartida
    extern __shared__ char Ads[];

    //Calculamos la posicion, fila y columa de la célula
    int fila = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int columna = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int posicion = fila * numColumnas + columna;


    /*Las siguientes comprobaciones se encargan de añadir una columna y una fila extras a la matriz compartida con el objetivo de poder
    calcular correctamente todos los valores de las células de la matriz (ya que cada célula tiene que acceder a elementos de esa fila
    y columna extras para poder conocer su siguiente valor)*/
    if (blockIdx.x == 0 && threadIdx.x == TILE_WIDTH - 1)
    {
        Ads[posicion + 1] = Ad[posicion + 1];
    }

    if (blockIdx.y == 0 && threadIdx.y == TILE_WIDTH - 1)
    {
        Ads[posicion + numColumnas] = Ad[posicion + numColumnas];
    }

    if (blockIdx.x == 1 && threadIdx.x == 0)
    {
        Ads[posicion - 1] = Ad[posicion - 1];
    }

    if (blockIdx.y == 1 && threadIdx.y == 0)
    {
        Ads[posicion - numColumnas] = Ad[posicion - numColumnas];
    }

    //Sincronizamos todos los hilos para que no haya errores a la hora de comprobar la matriz compartida.
    Ads[posicion] = Ad[posicion];
    __syncthreads();


    /*Las siquientes variables se corresponden a la posición de las células vecinas en la matriz de origen
   para coger el valor correctamente se utiliza el valor de la célula correspondiente a cada hilo y, utilizando
   la variable numColumnas (lo que sería el Width) se adquieren las posiciones de las vecinas*/
    int diagArribaIzq = posicion - numColumnas - 1;
    int diagArribaDch = posicion - numColumnas + 1;
    int arriba = posicion - numColumnas;
    int abajo = posicion + numColumnas;
    int diagAbajoIzq = posicion + numColumnas - 1;
    int diagAbajoDch = posicion + numColumnas + 1;
    int derecha = posicion + 1;
    int izquierda = posicion - 1;

    /*En lo siquiente guardamos cual es el estado de la célula e inicializamos variables que nos servirán para llevar el
    control del número de vecinas de la célula y de cuántas de ellas están vivas*/
    char estadoCelula = Ads[posicion];
    int nVivas = 0;
    int numVecinas;

    //Esta comprobación se realiza para procesar las posiciones que realmente pertenecen a la matriz real y no pertenecen a hilos que no queremos procesar (se salen de la matriz)
    if (columna < numColumnas || fila < numFilas)
    {
        int* vecinas;

        /*A partir de este momento comienzan todas las comprobaciones para saber cuáles son las células vecinas a la célula en cuestión teniendo
        en cuenta la posición de la misma. Según donde esté ubicada la célula, el número de vecinas varía, por tanto, en estos condicionales se
        recorren todas las posibilidades para saber con exactitud cuál es la cantidad de vecinas que tiene la célula de interés y qué posición
        ocupa cada una de ellas dentro de la matriz.*/
        if (fila == 0) 
        {
            if (columna == 0) 
            {
                vecinas = new int[3]{ derecha,abajo,diagAbajoDch };
                numVecinas = 3;
            }
            else if (columna == numColumnas - 1)
            {
                vecinas = new int[3]{ izquierda,abajo,diagAbajoIzq };
                numVecinas = 3;
            }
            else
            { 
                vecinas = new int[5]{ derecha,abajo,diagAbajoDch,izquierda,diagAbajoIzq };
                numVecinas = 5;
            }

        }
        else if (fila == numFilas - 1) 
        {
            if (columna == 0) 
            {
                vecinas = new  int[3]{ derecha,arriba,diagArribaDch };
                numVecinas = 3;
            }
            else if (columna == numColumnas - 1) 
            {
                vecinas = new int[3]{ izquierda,arriba,diagArribaIzq };
                numVecinas = 3;
            }
            else { 
                vecinas = new int[5]{ derecha,arriba,diagArribaDch,diagArribaIzq,izquierda };
                numVecinas = 5;
            }
        }
        else if (columna == 0) 
        {
            vecinas = new int[5]{ derecha,arriba,diagArribaDch,diagAbajoDch,abajo };
            numVecinas = 5;
        }
        else if (columna == numColumnas - 1) 
        {
            vecinas = new int[5]{ izquierda,arriba,diagArribaIzq,diagAbajoIzq,abajo };
            numVecinas = 5;
        }
        else
        {
            vecinas = new int[8]{ izquierda,arriba,diagArribaIzq,diagAbajoIzq,abajo,diagAbajoDch,diagArribaDch,derecha };

            numVecinas = 8;
        }

        /* Una vez que conocemos el número y la posición de las vecinas, recorremos un dichas posiciones en la matriz de origen
       y vamos actualizando el valor de las células vecinas que están vivas*/
        for (int i = 0; i < numVecinas; i++)
        {

            if (Ads[vecinas[i]] == viva)
            {
                nVivas++;
            }
        }

        /* En los siguientes condicionales se actualizará el estado de la célula dependiendo del número de vecinas vivas que tenga*/
        if (estadoCelula == viva)
        {
            if (nVivas == 2 || nVivas == 3)
            {
                estadoCelula = viva;
            }
            else
            {
                estadoCelula = muerta;
            }
        }
        else
        {
            if (nVivas == 3)
            {
                estadoCelula = viva;
            }
        }
        Bd[posicion] = estadoCelula;
    }
}

__global__ void kernelMultiplesBloques(char* Ad, char* Bd, int numColumnas, int numFilas, int TILE_WIDTH)
{

    //Calculamos la posicion, fila y columa de la célula
    int fila = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int columna = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int posicion = fila * numColumnas + columna;


    /*Las siquientes variables se corresponden a la posición de las células vecinas en la matriz de origen
   para coger el valor correctamente se utiliza el valor de la célula correspondiente a cada hilo y, utilizando
   la variable numColumnas (lo que sería el Width) se adquieren las posiciones de las vecinas*/
    int diagArribaIzq = posicion - numColumnas - 1;
    int diagArribaDch = posicion - numColumnas + 1;
    int arriba = posicion - numColumnas;
    int abajo = posicion + numColumnas;
    int diagAbajoIzq = posicion + numColumnas - 1;
    int diagAbajoDch = posicion + numColumnas + 1;
    int derecha = posicion + 1;
    int izquierda = posicion - 1;

    /*En lo siquiente guardamos cual es el estado de la célula e inicializamos variables que nos servirán para llevar el
    control del número de vecinas de la célula y de cuántas de ellas están vivas*/
    char estadoCelula = Ad[posicion];
    int nVivas = 0;
    int numVecinas;

    //Esta comprobación se realiza para que si hay más hilos que posiciones de la matriz, no se acceda a posiciones de matriz que no existen
    if (posicion < numFilas * numColumnas)
    {
        int* vecinas;

        /*A partir de este momento comienzan todas las comprobaciones para saber cuáles son las células vecinas a la célula en cuestión teniendo
        en cuenta la posición de la misma. Según donde esté ubicada la célula, el número de vecinas varía, por tanto, en estos condicionales se
        recorren todas las posibilidades para saber con exactitud cuál es la cantidad de vecinas que tiene la célula de interés y qué posición
        ocupa cada una de ellas dentro de la matriz.*/
        if (fila == 0) 
        {
            if (columna == 0) 
            {
                vecinas = new int[3]{ derecha,abajo,diagAbajoDch };
                numVecinas = 3;
            }
            else if (columna == numColumnas - 1) 
            {
                vecinas = new int[3]{ izquierda,abajo,diagAbajoIzq };
                numVecinas = 3;
            }
            else
            { 
                vecinas = new int[5]{ derecha,abajo,diagAbajoDch,izquierda,diagAbajoIzq };
                numVecinas = 5;
            }

        }
        else if (fila == numFilas - 1) 
        {
            if (columna == 0) 
            {
                vecinas = new  int[3]{ derecha,arriba,diagArribaDch };
                numVecinas = 3;
            }
            else if (columna == numColumnas - 1) 
            {
                vecinas = new int[3]{ izquierda,arriba,diagArribaIzq };
                numVecinas = 3;
            }
            else { 
                vecinas = new int[5]{ derecha,arriba,diagArribaDch,diagArribaIzq,izquierda };
                numVecinas = 5;
            }
        }
        else if (columna == 0) 
        {
            vecinas = new int[5]{ derecha,arriba,diagArribaDch,diagAbajoDch,abajo };
            numVecinas = 5;
        }
        else if (columna == numColumnas - 1) 
        {
            vecinas = new int[5]{ izquierda,arriba,diagArribaIzq,diagAbajoIzq,abajo };
            numVecinas = 5;
        }
        else
        {
            vecinas = new int[8]{ izquierda,arriba,diagArribaIzq,diagAbajoIzq,abajo,diagAbajoDch,diagArribaDch,derecha };

            numVecinas = 8;
        }

        /* Una vez que conocemos el número y la posición de las vecinas, recorremos un dichas posiciones en la matriz de origen
       y vamos actualizando el valor de las células vecinas que están vivas*/
        for (int i = 0; i < numVecinas; i++)
        {

            if (Ad[vecinas[i]] == viva)
            {
                nVivas++;
            }
        }

        /* En los siguientes condicionales se actualizará el estado de la célula dependiendo del número de vecinas vivas que tenga*/
        if (estadoCelula == viva)
        {
            if (nVivas == 2 || nVivas == 3)
            {
                estadoCelula = viva;
            }
            else
            {
                estadoCelula = muerta;
            }
        }
        else
        {
            if (nVivas == 3)
            {
                estadoCelula = viva;
            }
        }
        Bd[posicion] = estadoCelula;
    }
}

__global__ void kernelUnBloque(char* Ad, char* Bd, int numColumnas, int numFilas)
{
    int posicion = threadIdx.y * numColumnas + threadIdx.x;
    int fila = threadIdx.y;
    int columna = threadIdx.x;

    /*Las siquientes variables se corresponden a la posición de las células vecinas en la matriz de origen
    para coger el valor correctamente se utiliza el valor de la célula correspondiente a cada hilo y, utilizando
    la variable numColumnas (lo que sería el Width) se adquieren las posiciones de las vecinas*/
    int diagArribaIzq = posicion - numColumnas - 1;
    int diagArribaDch = posicion - numColumnas + 1;
    int arriba = posicion - numColumnas;
    int abajo = posicion + numColumnas;
    int diagAbajoIzq = posicion + numColumnas - 1;
    int diagAbajoDch = posicion + numColumnas + 1;
    int derecha = posicion + 1;
    int izquierda = posicion - 1;

    /*En lo siquiente guardamos cual es el estado de la célula e inicializamos variables que nos servirán para llevar el
    control del número de vecinas de la célula y de cuántas de ellas están vivas*/
    char estadoCelula = Ad[posicion];
    int nVivas = 0;
    int numVecinas;

    //Esta comprobación se realiza para que si hay más hilos que posiciones de la matriz, no se acceda a posiciones de matriz que no existen
    if (posicion < numFilas * numColumnas)
    {
        int* vecinas;

        /*A partir de este momento comienzan todas las comprobaciones para saber cuáles son las células vecinas a la célula en cuestión teniendo
        en cuenta la posición de la misma. Según donde esté ubicada la célula, el número de vecinas varía, por tanto, en estos condicionales se
        recorren todas las posibilidades para saber con exactitud cuál es la cantidad de vecinas que tiene la célula de interés y qué posición
        ocupa cada una de ellas dentro de la matriz.*/
        if (fila == 0) 
        {
            if (columna == 0) 
            {
                vecinas = new int[3]{ derecha,abajo,diagAbajoDch };
                numVecinas = 3;
            }
            else if (columna == numColumnas - 1) 
            {
                vecinas = new int[3]{ izquierda,abajo,diagAbajoIzq };
                numVecinas = 3;
            }
            else
            { 
                vecinas = new int[5]{ derecha,abajo,diagAbajoDch,izquierda,diagAbajoIzq };
                numVecinas = 5;
            }

        }
        else if (fila == numFilas - 1) 
        {
            if (columna == 0) 
            {
                vecinas = new  int[3]{ derecha,arriba,diagArribaDch };
                numVecinas = 3;
            }
            else if (columna == numColumnas - 1) 
            {
                vecinas = new int[3]{ izquierda,arriba,diagArribaIzq };
                numVecinas = 3;
            }
            else { 
                vecinas = new int[5]{ derecha,arriba,diagArribaDch,diagArribaIzq,izquierda };
                numVecinas = 5;
            }
        }
        else if (columna == 0) 
        {
            vecinas = new int[5]{ derecha,arriba,diagArribaDch,diagAbajoDch,abajo };
            numVecinas = 5;
        }
        else if (columna == numColumnas - 1) 
        {
            vecinas = new int[5]{ izquierda,arriba,diagArribaIzq,diagAbajoIzq,abajo };
            numVecinas = 5;
        }
        else
        {
            vecinas = new int[8]{ izquierda,arriba,diagArribaIzq,diagAbajoIzq,abajo,diagAbajoDch,diagArribaDch,derecha };

            numVecinas = 8;
        }

        /* Una vez que conocemos el número y la posición de las vecinas, recorremos un dichas posiciones en la matriz de origen
        y vamos actualizando el valor de las células vecinas que están vivas*/
        for (int i = 0; i < numVecinas; i++)
        {

            if (Ad[vecinas[i]] == viva)
            {
                nVivas++;
            }
        }


        /* En los siguientes condicionales se actualizará el estado de la célula dependiendo del número de vecinas vivas que tenga*/
        if (estadoCelula == viva)
        {
            if (nVivas == 2 || nVivas == 3)
            {
                estadoCelula = viva;
            }
            else
            {
                estadoCelula = muerta;
            }
        }
        else
        {
            if (nVivas == 3)
            {
                estadoCelula = viva;
            }
        }
        Bd[posicion] = estadoCelula;
        

    }
}


/* La siguiente función tiene como objetivo llenar la matriz de origen con un número especificado de células vivas repartidas en orden aleatorio.
Para ubicarlas de forma aleatoria se utiliza un número aleatorio.*/
void llenarMatriz(char* A_h, int numFilas, int numColumnas, int Vivas)
{
    srand(time(0));
    int r;
    for (int i = 0; i < numFilas; i++)
    {
        for (int j = 0; j < numColumnas; j++)
        {
            r = rand() % 2;
            if (r == 1 && Vivas > 0 && i > numFilas / 3)
            {
                A_h[j + (i * numColumnas)] = viva;
                Vivas = Vivas - 1;
            }
            else
            {
                A_h[j + (i * numColumnas)] = muerta;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    size_t pos;
    //Las siguientes variables se cogen a partir del comando por consola introducido por el usuario y corresponden al Width y al Heigth
    const int numFilas = stoi(argv[2], &pos);
    const int numColumnas = stoi(argv[3], &pos);
    int numElementos = numFilas * numColumnas;

    size_t size = numElementos * sizeof(char);

    char* A_h;
    char* R_h;
    A_h = new char[numElementos];
    R_h = new char[numElementos];

    char* Ad;
    char* Bd;
    

    /*Cogemos las características del dispositivo*/
    cudaDeviceProp prop;
    int count;
    int maxThreadsPerBlock;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {


        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        maxThreadsPerBlock = prop.maxThreadsPerBlock;
    }

    //Llenamos la matriz con células asignando que el número de células vivas sea un tercio del número total de elementos
    int nVivas = (numFilas * numColumnas) / 3;
    llenarMatriz(A_h, numFilas, numColumnas, nVivas);

    //Asignamos memoria
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);

    cudaMemcpy(Ad, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, R_h, size, cudaMemcpyHostToDevice);

    int modoEjecucion = -1;

    while (modoEjecucion != 1 && modoEjecucion != 2 && modoEjecucion!= 3) {
        cout << "Seleccione el modo de ejecucion del kernel\n 1) Memoria Global 1 Bloque\n 2) Memoria Global Multiples Bloques \n 3) Memoria Compartida ";
        cin >> modoEjecucion;

        //Comprobamos que la configuración del kernel no viole las restricciones del dispositivo
        if (modoEjecucion == 1 && numElementos > maxThreadsPerBlock)
        {
            modoEjecucion = -1;
            printf("No se puede ejecutar en el modo que ha elegido por que las caracteristicas del dispositivo no lo permiten.\n Elija otro modo o reinicie el programa para introducir otra matriz");
        }
    }

    for (int i = 0; i < numFilas; i++)
    {
        for (int j = 0; j < numColumnas; j++)
        {
            printf("%c ", A_h[j + (i * numColumnas)]);
        }
        printf("\n\n");
    }
    if (modoEjecucion == 1)
    {
        dim3 dimGrid(1, 1);
        dim3 dimBlock(numColumnas, numFilas);

        for (int i = 0; i < numFilas; i++)
        {
            //Forma manual
            if (argv[1][1] == 'm')
            {
                int valorUsuario = -1;

                while (valorUsuario != 5)
                {
                    printf("Introduzca 5 para seguir ");
                    scanf("%d", &valorUsuario);
                }
                kernelUnBloque <<< dimGrid, dimBlock >>> (Ad, Bd, numColumnas, numFilas);
                cudaMemcpy(R_h, Bd, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(Ad, Bd, size, cudaMemcpyDeviceToDevice);
                for (int i = 0; i < numFilas; i++)
                {
                    for (int j = 0; j < numColumnas; j++)
                    {
                        printf("%c ", R_h[j + (i * numColumnas)]);
                    }
                    printf("\n\n");
                }
            }
            else { //Forma automatica
                Sleep(1000);
                kernelUnBloque <<< dimGrid, dimBlock >>> (Ad, Bd, numColumnas, numFilas);
                cudaMemcpy(R_h, Bd, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(Ad, Bd, size, cudaMemcpyDeviceToDevice);
               
                for (int i = 0; i < numFilas; i++)
                {
                    for (int j = 0; j < numColumnas; j++)
                    {
                        printf("%c ", R_h[j + (i * numColumnas)]);
                    }
                    printf("\n\n");
                }
            }
        }
    }else if (modoEjecucion == 2)
    {
        int TILE_WIDTH;
        if (numColumnas % 2 == 0) {
            TILE_WIDTH = numColumnas / 2;
        }
        else
        {
            TILE_WIDTH = (numColumnas / 2) + 1;
        }
        int numBloquesX;
        int numBloquesY;
        if (numColumnas % TILE_WIDTH == 0)
        {
            numBloquesX = numColumnas / TILE_WIDTH;
        }
        else
        {
            numBloquesX = (numColumnas / TILE_WIDTH) + 1;
        }
        if (numFilas % TILE_WIDTH == 0)
        {
            numBloquesY = numFilas / TILE_WIDTH;
        }
        else
        {
            numBloquesY = (numFilas / TILE_WIDTH) + 1;
        }
        dim3 dimGrid(numBloquesX, numBloquesY);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

        //Comprobamos que la configuración del kernel no viole las restricciones del dispositivo
        if ((TILE_WIDTH * TILE_WIDTH)> maxThreadsPerBlock)
        {
            printf("Error, pruebe con otra matriz, la matriz introducida no está admitida por las características del dispositivo.");
        }
        else
        {
            for (int i = 0; i < numFilas; i++)
            {
                //Forma manual
                if (argv[1][1] == 'm')
                {
                    int valorUsuario = -1;

                    while (valorUsuario != 5)
                    {
                        printf("Introduzca 5 para seguir ");
                        scanf("%d", &valorUsuario);
                    }
                    kernelMultiplesBloques << < dimGrid, dimBlock >> > (Ad, Bd, numColumnas, numFilas, TILE_WIDTH);
                    cudaMemcpy(R_h, Bd, size, cudaMemcpyDeviceToHost);
                    cudaMemcpy(Ad, Bd, size, cudaMemcpyDeviceToDevice);
                    for (int i = 0; i < numFilas; i++)
                    {
                        for (int j = 0; j < numColumnas; j++)
                        {
                            printf("%c ", R_h[j + (i * numColumnas)]);
                        }
                        printf("\n\n");
                    }
                }
                else { //Forma automatica
                    Sleep(1000);
                    kernelMultiplesBloques << < dimGrid, dimBlock >> > (Ad, Bd, numColumnas, numFilas, TILE_WIDTH);
                    cudaMemcpy(R_h, Bd, size, cudaMemcpyDeviceToHost);
                    cudaMemcpy(Ad, Bd, size, cudaMemcpyDeviceToDevice);
                    
                    for (int i = 0; i < numFilas; i++)
                    {
                        for (int j = 0; j < numColumnas; j++)
                        {
                            printf("%c ", R_h[j + (i * numColumnas)]);
                        }
                        printf("\n\n");
                    }
                }
            }
        }
        
    }
    else if (modoEjecucion == 3)
    {
   
        int TILE_WIDTH;
        if (numColumnas % 2 == 0) {
            TILE_WIDTH = numColumnas / 2;
        }
        else
        {
            TILE_WIDTH = (numColumnas / 2) + 1;
        }
        int numBloquesX;
        int numBloquesY;
        if (numColumnas % TILE_WIDTH == 0)
        {
            numBloquesX = numColumnas / TILE_WIDTH;
        }
        else
        {
            numBloquesX = (numColumnas / TILE_WIDTH) + 1;
        }
        if (numFilas % TILE_WIDTH == 0)
        {
            numBloquesY = numFilas / TILE_WIDTH;
        }
        else
        {
            numBloquesY = (numFilas / TILE_WIDTH) + 1;
        }
        dim3 dimGrid(numBloquesX, numBloquesY);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

       
        for (int i = 0; i < numFilas; i++)
        {
            int numBloquesX;
            int numBloquesY;

            if (numColumnas % TILE_WIDTH == 0)
            {
                numBloquesX = numColumnas / TILE_WIDTH;
            }
            else
            {
                numBloquesX = (numColumnas / TILE_WIDTH) + 1;
            }
            if (numFilas % TILE_WIDTH == 0)
            {
                numBloquesY = numFilas / TILE_WIDTH;
            }
            else
            {
                numBloquesY = (numFilas / TILE_WIDTH) + 1;
            }

            dim3 dimGrid(numBloquesX, numBloquesY);
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
            if (argv[1][1] == 'm')
            {
                int valorUsuario = -1;

                while (valorUsuario != 5)
                {
                    printf("Introduzca 5 para seguir ");
                    scanf("%d", &valorUsuario);
                }
                kernelMemoriaCompartida << < dimGrid, dimBlock, ((TILE_WIDTH * TILE_WIDTH) + 1) * sizeof(char*) >> > (Ad, Bd, numColumnas, numFilas, TILE_WIDTH);
                cudaMemcpy(R_h, Bd, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(Ad, Bd, size, cudaMemcpyDeviceToDevice);
                for (int i = 0; i < numFilas; i++)
                {
                    for (int j = 0; j < numColumnas; j++)
                    {
                        printf("%c ", R_h[j + (i * numColumnas)]);
                    }
                    printf("\n\n");
                }
            }
            else { //Forma automatica
                Sleep(1000);
                kernelMemoriaCompartida << < dimGrid, dimBlock, ((TILE_WIDTH * TILE_WIDTH) + 1) * sizeof(char*) >> > (Ad, Bd, numColumnas, numFilas, TILE_WIDTH);
                cudaMemcpy(R_h, Bd, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(Ad, Bd, size, cudaMemcpyDeviceToDevice);
                for (int i = 0; i < numFilas; i++)
                {
                    for (int j = 0; j < numColumnas; j++)
                    {
                        printf("%c ", R_h[j + (i * numColumnas)]);
                    }
                    printf("\n\n");
                }
            }

        }   
    }
    //Liberamos espacio
    cudaFree(Bd);
    cudaFree(Ad);

    return 0;
}
