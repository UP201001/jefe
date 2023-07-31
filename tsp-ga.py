import random
import math
import matplotlib.pyplot as plt

# Función que calcula la distancia entre dos ciudades (puntos) en un plano cartesiano
def distancia(ciudad1, ciudad2):
    x_distancia = abs(ciudad1[0] - ciudad2[0])
    y_distancia = abs(ciudad1[1] - ciudad2[1])
    distancia = math.sqrt(x_distancia**2 + y_distancia**2)
    return distancia

# Clase que representa el fitness (aptitud) de un conjunto de ciudades
class Aptitud:
    def __init__(self, ruta):
        self.ruta = ruta
        self.distancia = self.distancia_ruta()
        self.fitness = 1 / self.distancia

    # Método que calcula la distancia total de una ruta sumando las distancias entre ciudades adyacentes
    def distancia_ruta(self):
        distancia_ruta = 0.0
        for i in range(len(self.ruta)):
            desde_ciudad = self.ruta[i]
            hacia_ciudad = self.ruta[(i + 1) % len(self.ruta)]
            distancia_ruta += distancia(desde_ciudad, hacia_ciudad)
        return distancia_ruta

# Función que crea una ruta aleatoria a partir de una lista de ciudades
def crear_ruta(lista_ciudades):
    ruta = random.sample(lista_ciudades, len(lista_ciudades))
    ruta.append(ruta[0])  # Agregar la ciudad de inicio al final de la ruta
    return ruta

# Función que inicializa la población generando un conjunto de rutas aleatorias
def poblacion_inicial(tamano_poblacion, lista_ciudades):
    poblacion = []
    for _ in range(tamano_poblacion):
        poblacion.append(crear_ruta(lista_ciudades))
    return poblacion

# Función que ordena las rutas de la población en función de su aptitud (fitness)
def ordenar_rutas(poblacion):
    resultados_aptitud = []
    for i, ruta in enumerate(poblacion):
        aptitud = Aptitud(ruta)
        resultados_aptitud.append((i, aptitud.fitness))
    return sorted(resultados_aptitud, key=lambda item: item[1], reverse=True)

# Función de selección que elige las mejores rutas para la reproducción (basado en la aptitud)
def seleccion(poblacion_ordenada, tamano_elite):
    resultados_seleccion = []
    suma_acumulada = 0
    for i in range(len(poblacion_ordenada)):
        suma_acumulada += poblacion_ordenada[i][1]
        resultados_seleccion.append(poblacion_ordenada[i][0])
        if i >= tamano_elite - 1:
            break
    for i in range(len(poblacion_ordenada) - tamano_elite):
        seleccionado = random.random() * suma_acumulada
        for i in range(len(poblacion_ordenada)):
            seleccionado -= poblacion_ordenada[i][1]
            if seleccionado <= 0:
                resultados_seleccion.append(poblacion_ordenada[i][0])
                break
    return resultados_seleccion

# Función que crea el pool de reproducción seleccionando las rutas correspondientes
def pool_reproduccion(poblacion, resultados_seleccion):
    pool_reproduccion = []
    for i in resultados_seleccion:
        pool_reproduccion.append(poblacion[i])
    return pool_reproduccion

# Función que crea una nueva ruta combinando dos rutas padre
def cruzar(padre1, padre2):
    hijo = []
    hijo_padre1 = []
    hijo_padre2 = []

    gen_a = random.randint(0, len(padre1) - 1)
    gen_b = random.randint(0, len(padre1) - 1)

    inicio_gen = min(gen_a, gen_b)
    fin_gen = max(gen_a, gen_b)

    for i in range(inicio_gen, fin_gen):
        hijo_padre1.append(padre1[i])

    hijo_padre2 = [item for item in padre2 if item not in hijo_padre1]

    hijo.extend(hijo_padre1)
    hijo.extend(hijo_padre2)
    # Agregar la última ciudad al final de la ruta para asegurar que regrese a la ciudad inicial
    hijo.append(hijo[0])
    return hijo

# Función que crea la siguiente generación de rutas (nueva población)
def siguiente_generacion(generacion_actual, tamano_elite, tasa_mutacion):
    poblacion_ordenada = ordenar_rutas(generacion_actual)
    resultados_seleccion = seleccion(poblacion_ordenada, tamano_elite)
    pool_reproduccion_local = pool_reproduccion(generacion_actual, resultados_seleccion)  # Cambio de nombre de variable
    hijos = []
    longitud_pool_reproduccion = len(pool_reproduccion_local)  # Usar el nuevo nombre de variable
    for i in range(tamano_elite):
        hijos.append(pool_reproduccion_local[i])  # Usar el nuevo nombre de variable
    for i in range(longitud_pool_reproduccion - tamano_elite):
        hijo = cruzar(pool_reproduccion_local[i], pool_reproduccion_local[longitud_pool_reproduccion - i - 1])  # Usar el nuevo nombre de variable
        hijos.append(hijo)
    return mutar_poblacion(hijos, tasa_mutacion)

# Función que muta una ruta con una cierta tasa de mutación
def mutar(individual, tasa_mutacion):
    for intercambio in range(len(individual)):
        if random.random() < tasa_mutacion:
            intercambiar_con = random.randint(0, len(individual) - 1)
            individual[intercambio], individual[intercambiar_con] = individual[intercambiar_con], individual[intercambio]
    return individual

# Función que aplica la mutación a toda la población
def mutar_poblacion(poblacion, tasa_mutacion):
    poblacion_mutada = []
    for individuo in poblacion:
        individuo_mutado = mutar(individuo, tasa_mutacion)
        poblacion_mutada.append(individuo_mutado)
    return poblacion_mutada

# Función que ejecuta el algoritmo genético para encontrar la mejor ruta
def algoritmo_genetico(lista_ciudades, tamano_poblacion, tamano_elite, tasa_mutacion, generaciones):
    poblacion_actual = poblacion_inicial(tamano_poblacion, lista_ciudades)
    mejor_ruta = poblacion_actual[ordenar_rutas(poblacion_actual)[0][0]]
    print("Ruta sin iniciar el algoritmo genético:")
    print(mejor_ruta)
    print("Distancia a recorrer:", 1 / ordenar_rutas(poblacion_actual)[0][1])
    for _ in range(generaciones):
        poblacion_actual = siguiente_generacion(poblacion_actual, tamano_elite, tasa_mutacion)
    mejor_ruta_indice = ordenar_rutas(poblacion_actual)[0][0]
    mejor_ruta = poblacion_actual[mejor_ruta_indice]
    print("Mejor ruta encontrada:")
    print(mejor_ruta)
    print("Distancia a recorrer:", 1 / ordenar_rutas(poblacion_actual)[0][1])
    return mejor_ruta

# Función para graficar la ruta inicial y la mejor ruta encontrada
def graficar_rutas(ruta_inicial, mejor_ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Graficar la ruta inicial
    x_coords_init = [ciudad[0] for ciudad in ruta_inicial]
    y_coords_init = [ciudad[1] for ciudad in ruta_inicial]
    ax1.scatter(x_coords_init, y_coords_init, label="Ciudad")
    ax1.plot(x_coords_init + [ruta_inicial[0][0]], y_coords_init + [ruta_inicial[0][1]], label="Ruta")  # Agregar la primera ciudad al final para cerrar el ciclo
    ax1.legend(loc="upper left")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Ruta sin iniciar el algoritmo genético")
    
    # Graficar la mejor ruta encontrada
    x_coords_best = [ciudad[0] for ciudad in mejor_ruta]
    y_coords_best = [ciudad[1] for ciudad in mejor_ruta]
    ax2.scatter(x_coords_best, y_coords_best, label="Ciudad")
    ax2.plot(x_coords_best, y_coords_best, label="Ruta")
    ax2.legend(loc="upper left")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Mejor ruta encontrada")
    
    plt.tight_layout()
    plt.show()

# Coordenadas manuales para cada ciudad en el plano
ciudades = [(0, 0), (1, 3), (4, 2), (3, 6), (7, 2),
            (2, 5), (5, 1), (8, 4), (10, 10), (12, 8),
            (6, 9), (11, 3), (15, 5), (14, 7), (9, 1),
            (13, 11), (16, 9), (18, 12), (20, 15), (22, 18),
            (24, 13), (27, 11), (29, 14), (31, 17), (35, 20),
            (37, 19), (39, 22), (25, 16), (30, 21), (33, 25),
            (36, 23), (32, 28), (38, 27), (34, 29), (21, 30),
            (19, 26), (17, 31), (23, 32), (26, 33), (28, 35),
            (40, 36), (42, 38), (43, 39), (45, 37), (44, 41)]

mejor_ruta_encontrada = algoritmo_genetico(ciudades, 100, 20, 0.01, 1000)

graficar_rutas(ciudades, mejor_ruta_encontrada)
