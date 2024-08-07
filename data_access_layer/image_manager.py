import pyodbc


class ImageManager():


    def __init__(self,connection_string):
        self.connection = pyodbc.connect(connection_string)


    def persist_image(self,image):

        image_id=None
        with self.connection.cursor() as cursor:
            params = (image.name,image.category_key,image.category_desc)
            cursor.execute("{CALL PersistImage (?,?,?)}", params)

            for row in cursor:
                image_id=int( row[0])
            self.connection.commit()
        return  image_id


    def persist_image_matrix(self,pixel_x,pixel_y,image_id):
        image_matrix_id=None
        with self.connection.cursor() as cursor:
            params = (pixel_x,pixel_y,image_id)
            cursor.execute("{CALL PersistImageMatrix (?,?,?)}", params)

            for row in cursor:
                image_matrix_id=int( row[0])

            self.connection.commit()

        return image_matrix_id


    def persist_image_pixel(self,red,green,blue,image_matrix_id):
        image_pixel_id=None
        with self.connection.cursor() as cursor:
            params = (red, green, blue, image_matrix_id)
            cursor.execute("{CALL PersistImagePixel (?,?,?,?)}", params)

            for row in cursor:
                image_pixel_id=int(row[0])

            self.connection.commit()

        return image_pixel_id


