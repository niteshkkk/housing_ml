class BaseDataFrame(object):

    def __init__(self, data_frame):
        self.data_frame = data_frame

    def head(self):
        if self.data_frame is None:
            return None
        return self.data_frame.head()

    def describe(self):
        if self.data_frame is None:
            return None
        return self.data_frame.describe()

    def info(self):

        if self.data_frame is None:
            return None
        return self.data_frame.info()

    def get_data_frame(self):
        return self.data_frame
