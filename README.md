# PROYECT_UDEA_CLOUD

Trabajo de implementación de modelo de machine learning con servicios en la nube Azure

Estudiantes:

Leidy Tatiana Molina Ruiz - tatiana.molina1@udea.edu.co

Carolina Alvarez Florez - carolina.alvarezf@udea.edu.co

Contexto del modelo: a partir de los contactos de los clientes a un fondo de pensiones, se pretende predecir en canal de uso preferido por el cliente entre 3 principales: línea de servicio, Oficina de servicio, Oficina virtual.

Datos: Es importante tener en cuenta que el datset completo se encuentra almacenado en el ambiente de GCP corporativo por lo tanto no es posible realizar el entrenamiento del modelo directamente con esta fuente debido a restricciones de conexión, seguridad y privacidad de la información, por esta razón, la implementación se hace con una muestra de los datos que conserva la misma estructura pero con datos ficticios.

El modelo utilizado consiste en un Sequential Feature Selector con 11 características principales y un Knn con 50 vecinos.
