class Image():

    def __init__(self,p_name,p_category_key,p_category_desc,p_pixels,p_classif=None):
        self.id=None
        self.name=p_name
        self.category_key=p_category_key
        self.category_desc=p_category_desc
        self.pixels=p_pixels
        self.classif=p_classif