import requests

# URL del archivo de texto en Project Gutenberg
url = "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"

# Realizar la solicitud para descargar el contenido
response = requests.get(url)

# Comprobar que la solicitud fue exitosa (código 200)
if response.status_code == 200:
    # Guardar el archivo en el disco
    with open("Gatsby.txt", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("Archivo descargado y guardado como 'Gatsby.txt'")
else:
    print("Error al descargar el archivo:", response.status_code)

with open("Gatsby.txt", "r", encoding="utf-8") as file:
    contenido = file.read()
    print(contenido[:500])  # Imprime los primeros 500 caracteres