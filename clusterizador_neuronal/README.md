# Clusterizador Neuronal

Este proyecto implementa un sencillo sistema de clustering basado en una
red neuronal autoencoder. Se utiliza **Keras** con **TensorFlow** como backend y
el conjunto de datos de dígitos de `scikit-learn` para demostrar su
funcionamiento.

## Requisitos

- Python 3.11 o superior.
- `tensorflow`
- `scikit-learn`

Instala las dependencias con:

```bash
pip install tensorflow scikit-learn
```

## Uso

Ejecuta el script principal para entrenar el autoencoder y obtener las
etiquetas resultantes del agrupamiento:

```bash
python clusterizador.py
```

El programa imprime las etiquetas de cluster asignadas a las primeras
muestras del conjunto de datos.
