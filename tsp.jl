# Estas 3 primeras líneas solo se corren la primera vez.
# Después se pueden comentar o quitar
# using Pkg
# Pkg.add("Flux")
# Pkg.add("Random")

# Aquí empieza el código
using Flux
using Random

# Puntos en el plano 2D
points = [(0, 0), (1, 3), (4, 2), (3, 6), (7, 2),
(2, 5), (5, 1), (8, 4), (10, 10), (12, 8),
(6, 9), (11, 3), (15, 5), (14, 7), (9, 1),
(13, 11), (16, 9), (18, 12), (20, 15), (22, 18),
(24, 13), (27, 11), (29, 14), (31, 17), (35, 20),
(37, 19), (39, 22), (25, 16), (30, 21), (33, 25),
(36, 23), (32, 28), (38, 27), (34, 29), (21, 30),
(19, 26), (17, 31), (23, 32), (26, 33), (28, 35),
(40, 36), (42, 38), (43, 39), (45, 37), (44, 41)]

# Datos de entrenamiento
num_samples = 1000000

# Épocas
epochs = 100000

# Se obtienen cuántas ciudades hay que recorrer
num_points = length(points)

# Función para calcular la distancia euclidiana entre dos puntos
function distance(p1, p2)
    return sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

# Crea una matriz de Julia llena de ceros con forma `(num_points, num_points)`.
# Esta matriz se utilizará para almacenar las distancias entre todas las parejas de puntos.
dist_matrix = zeros(num_points, num_points)

# Se modifican los valores de la matriz 'dist_matrix' por la verdadera distancia entre las parejas de puntos
for i in 1:num_points
    for j in 1:num_points
        dist_matrix[i, j] = distance(points[i], points[j])
    end
end

# Función para obtener la ruta dado el orden de visita
# Esta función toma como entrada un 'order', que representa el orden 
# en el cual se deben visitar las ciudades.
function get_route(order)
    return [points[order[i]] for i in eachindex(order)]
end

# Función para calcular la longitud total de una ruta
function route_length(route)
    total_length = 0.0
    for i in 1:length(route)-1
        total_length += distance(route[i], route[i+1])
    end
    total_length += distance(route[end], route[1])  # Volver al punto de partida
    return total_length
end

# Función de pérdida para minimizar la longitud de la ruta
function tsp_loss(order)
    route = get_route(order)
    return route_length(route)
end

# Primero se crea una lista con puntos aleatorios de entre 1 y 'num_points'
# Se repetirá el proceso 'num_samples' cantidad de veces
input_data = [shuffle(1:num_points) for _ in 1:num_samples]

# Se calcula la longitud de cada una de las ordenaciones aleatorias de puntos
output_data = [tsp_loss(order) for order in input_data]

# Entrenamiento de la red neuronal
model = Flux.Chain(
    Flux.Dense(num_points, 128),
    Flux.relu,
    Flux.Dense(128, 64),
    Flux.relu,
    Flux.Dense(64, num_points),
)
opt = ADAM(0.001)

for epoch in 1:epochs
    grads = Flux.gradient(loss, params(model))(data[i])
    for (param, grad) in zip(params(model), grads)
        Flux.update!(opt, param, -grad)
    end
end

# Obtener la ruta más corta encontrada
best_order_idx = argmin(output_data)
best_order = input_data[best_order_idx]
best_route = get_route(best_order)

println("La ruta más corta encontrada es: $best_route")
println("Longitud total de la ruta: $(tsp_loss(best_order))")
