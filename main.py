from pageindex_open import *

PDF_FILE = "~/Downloads/2023-annual-report-truncated.pdf"
QUERY = "what about financial stability?"


pio = PIO(PDF_FILE)
pio.build_index() 

answer = pio.query(QUERY, top_k=2)
print(answer)