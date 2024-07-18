from utils import *
from config import *

df = read_dataframe_excel(BASE_PATH, DATA_PATH, FILE_NAME)
print(df.head())