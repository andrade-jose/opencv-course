"""Operações bitwise: AND, OR, XOR e NOT."""
import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype="uint8")

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow("Retangulo", rectangle)
cv.imshow("Circulo", circle)

# AND --> regiões que se intersectam
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow("Bitwise AND", bitwise_and)

# OR --> regiões que se intersectam e que não se intersectam
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow("Bitwise OR", bitwise_or)

# XOR --> apenas as regiões que NÃO se intersectam
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow("Bitwise XOR", bitwise_xor)

# NOT --> inverte a imagem
bitwise_not = cv.bitwise_not(circle)
cv.imshow("Circulo NOT", bitwise_not)

cv.waitKey(0)
cv.destroyAllWindows()
