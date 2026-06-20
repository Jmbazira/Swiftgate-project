from simulation import simulate_airport_conditions
import cv2

test_image = "data/UTKFace/9_1_2_20161219204347420.jpg.chip.jpg"

img = simulate_airport_conditions(test_image)

if img is not None:
    cv2.imwrite("test_output.jpg", img)
    print("Simulation successful")
else:
    print("Simulation failed")
