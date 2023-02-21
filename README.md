# Twitter Sentiment Analysis

El objetivo del proyecto es analizar y clasificar el texto publicado en la red social Twitter según su sentimiento (positivo o negativo) mediante el uso de técnicas de machine learning. También se pretende acceder a la API de Twitter para analizar en tiempo real el sentimiento que genera en el conjunto de los usuarios un tema en concreto.

## Modelo Utilizado

Tratándose de textos cortos en su mayoría, la mejor opción en este caso es una red neuronal LSTM (Long Short-Term Memory) ya que su capacidad para recordar inputs pasados la hace perfecta para el procesamiento de textos, su único problema sería al recibir un texto de mayores dimensiones.

* Como entrada se ha utilizado una capa `Embedding` que ayudará a reducir la dimensionalidad de los vectores que previamente habían sido generados. Utiliza las 3000 palabras descritas anteriormente como vocabulario.
* Para evitar el sobreajuste, se ha optado por una capa `SpatialDropout1D` que bloquea aleatoriamente algunas de las entradas de la primera capa para fomentar la generalización.
* Capa `LSTM` de tamaño 256, permite que el modelo recuerde dependencias y que lo utilice para aprender de ellas.
* Última capa `Dense` de salida binaria, una para cada clase. Con función de activación sigmoide, producirá valores de salida entre 0 y 1 interpretables como la probabilidad de que la entrada del modelo sea de una clase determinada.