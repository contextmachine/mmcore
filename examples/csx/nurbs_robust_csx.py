import numpy as np
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface



surface = NURBSSurface( # Spiral Ruled Surface
    np.array(
        [
            [
                [65.66864862623089, 93.74205665737186, 41.41100531879138, 1.0],
                [65.66864862623089, 93.74205665737186, 0.10498601801243446, 1.0],
            ],
            [
                [65.70381308003529, 92.80184185418611, 41.41100531879138, 1.0],
                [65.70381308003529, 92.80184185418611, 0.10498601801243446, 1.0],
            ],
            [
                [65.93866781565862, 90.93344804677268, 41.41100531879138, 1.0],
                [65.93866781565862, 90.93344804677268, 0.10498601801243446, 1.0],
            ],
            [
                [66.76764474869523, 88.25139426624095, 41.41100531879138, 1.0],
                [66.76764474869523, 88.25139426624095, 0.10498601801243446, 1.0],
            ],
            [
                [68.04240133132686, 85.7698431189295, 41.41100531879138, 1.0],
                [68.04240133132686, 85.7698431189295, 0.10498601801243446, 1.0],
            ],
            [
                [69.71866508308432, 83.5616093222356, 41.41100531879138, 1.0],
                [69.71866508308432, 83.5616093222356, 0.10498601801243446, 1.0],
            ],
            [
                [71.7405653611562, 81.69030895188432, 41.41100531879138, 1.0],
                [71.7405653611562, 81.69030895188432, 0.10498601801243446, 1.0],
            ],
            [
                [74.04239056810582, 80.208508216853, 41.41100531879138, 1.0],
                [74.04239056810582, 80.208508216853, 0.10498601801243446, 1.0],
            ],
            [
                [76.55073895845743, 79.1562631175434, 41.41100531879138, 1.0],
                [76.55073895845743, 79.1562631175434, 0.10498601801243446, 1.0],
            ],
            [
                [79.18684761002137, 78.5600515829087, 41.41100531879138, 1.0],
                [79.18684761002137, 78.5600515829087, 0.10498601801243446, 1.0],
            ],
            [
                [81.86906475574814, 78.43213634667575, 41.41100531879138, 1.0],
                [81.86906475574814, 78.43213634667575, 0.10498601801243446, 1.0],
            ],
            [
                [84.51537751121424, 78.77037006251331, 41.41100531879138, 1.0],
                [84.51537751121424, 78.77037006251331, 0.10498601801243446, 1.0],
            ],
            [
                [87.04591956978467, 79.55844392285438, 41.41100531879138, 1.0],
                [87.04591956978467, 79.55844392285438, 0.10498601801243446, 1.0],
            ],
            [
                [89.38538144158962, 80.76656645646, 41.41100531879138, 1.0],
                [89.38538144158962, 80.76656645646, 0.10498601801243446, 1.0],
            ],
            [
                [91.46525070332797, 82.35254636479925, 41.41100531879138, 1.0],
                [91.46525070332797, 82.35254636479925, 0.10498601801243446, 1.0],
            ],
            [
                [93.22581559441008, 84.2632410840891, 41.41100531879138, 1.0],
                [93.22581559441008, 84.2632410840891, 0.10498601801243446, 1.0],
            ],
            [
                [94.61787346921517, 86.43632200091871, 41.41100531879138, 1.0],
                [94.61787346921517, 86.43632200091871, 0.10498601801243446, 1.0],
            ],
            [
                [95.60409540762578, 88.80229811220822, 41.41100531879138, 1.0],
                [95.60409540762578, 88.80229811220822, 0.10498601801243446, 1.0],
            ],
            [
                [96.16000948361588, 91.28673269687489, 41.41100531879138, 1.0],
                [96.16000948361588, 91.28673269687489, 0.10498601801243446, 1.0],
            ],
            [
                [96.27457740208934, 93.81258244954697, 41.41100531879138, 1.0],
                [96.27457740208934, 93.81258244954697, 0.10498601801243446, 1.0],
            ],
            [
                [95.95035205531579, 96.30258565083841, 41.41100531879138, 1.0],
                [95.95035205531579, 96.30258565083841, 0.10498601801243446, 1.0],
            ],
            [
                [95.20321661514595, 98.68162537521448, 41.41100531879138, 1.0],
                [95.20321661514595, 98.68162537521448, 0.10498601801243446, 1.0],
            ],
            [
                [94.06171865923878, 100.87899545606837, 41.41100531879138, 1.0],
                [94.06171865923878, 100.87899545606837, 0.10498601801243446, 1.0],
            ],
            [
                [92.56602513215233, 102.83050085686249, 41.41100531879138, 1.0],
                [92.56602513215233, 102.83050085686249, 0.10498601801243446, 1.0],
            ],
            [
                [90.76653529163374, 104.48033008807467, 41.41100531879138, 1.0],
                [90.76653529163374, 104.48033008807467, 0.10498601801243446, 1.0],
            ],
            [
                [88.72219884713252, 105.78264515104536, 41.41100531879138, 1.0],
                [88.72219884713252, 105.78264515104536, 0.10498601801243446, 1.0],
            ],
            [
                [86.49859496608252, 106.70284391559045, 41.41100531879138, 1.0],
                [86.49859496608252, 106.70284391559045, 0.10498601801243446, 1.0],
            ],
            [
                [84.16583446139505, 107.21846053641033, 41.41100531879138, 1.0],
                [84.16583446139505, 107.21846053641033, 0.10498601801243446, 1.0],
            ],
            [
                [81.79635209827242, 107.3196811361933, 41.41100531879138, 1.0],
                [81.79635209827242, 107.3196811361933, 0.10498601801243446, 1.0],
            ],
            [
                [79.46265845209487, 107.00946415873321, 41.41100531879138, 1.0],
                [79.46265845209487, 107.00946415873321, 0.10498601801243446, 1.0],
            ],
            [
                [77.23512106166146, 106.30326713866779, 41.41100531879138, 1.0],
                [77.23512106166146, 106.30326713866779, 0.10498601801243446, 1.0],
            ],
            [
                [75.17984277182609, 105.22839376047693, 41.41100531879138, 1.0],
                [75.17984277182609, 105.22839376047693, 0.10498601801243446, 1.0],
            ],
            [
                [73.35670123195811, 103.8229866146385, 41.41100531879138, 1.0],
                [73.35670123195811, 103.8229866146385, 0.10498601801243446, 1.0],
            ],
            [
                [71.81760766062072, 102.1347016528925, 41.41100531879138, 1.0],
                [71.81760766062072, 102.1347016528925, 0.10498601801243446, 1.0],
            ],
            [
                [70.60503540948311, 100.21910968071924, 41.41100531879138, 1.0],
                [70.60503540948311, 100.21910968071924, 0.10498601801243446, 1.0],
            ],
            [
                [69.75085981880395, 98.13787802990888, 41.41100531879138, 1.0],
                [69.75085981880395, 98.13787802990888, 0.10498601801243446, 1.0],
            ],
            [
                [69.27554065315417, 95.9567916052006, 41.41100531879138, 1.0],
                [69.27554065315417, 95.9567916052006, 0.10498601801243446, 1.0],
            ],
            [
                [69.1876673720617, 93.7436766316274, 41.41100531879138, 1.0],
                [69.1876673720617, 93.7436766316274, 0.10498601801243446, 1.0],
            ],
            [
                [69.48387598020832, 91.56629254056372, 41.41100531879138, 1.0],
                [69.48387598020832, 91.56629254056372, 0.10498601801243446, 1.0],
            ],
            [
                [70.14913458016932, 89.49025748407303, 41.41100531879138, 1.0],
                [70.14913458016932, 89.49025748407303, 0.10498601801243446, 1.0],
            ],
            [
                [71.15738338064384, 87.57707098525617, 41.41100531879138, 1.0],
                [71.15738338064384, 87.57707098525617, 0.10498601801243446, 1.0],
            ],
            [
                [72.47250414523428, 85.8822933063143, 41.41100531879138, 1.0],
                [72.47250414523428, 85.8822933063143, 0.10498601801243446, 1.0],
            ],
            [
                [74.04958422820772, 84.45393539485173, 41.41100531879138, 1.0],
                [74.04958422820772, 84.45393539485173, 0.10498601801243446, 1.0],
            ],
            [
                [75.83643172805296, 83.3311059555472, 41.41100531879138, 1.0],
                [75.83643172805296, 83.3311059555472, 0.10498601801243446, 1.0],
            ],
            [
                [77.77529114862375, 82.54295353873391, 41.41100531879138, 1.0],
                [77.77529114862375, 82.54295353873391, 0.10498601801243446, 1.0],
            ],
            [
                [79.80470349335278, 82.10793182825427, 41.41100531879138, 1.0],
                [79.80470349335278, 82.10793182825427, 0.10498601801243446, 1.0],
            ],
            [
                [81.86145107737659, 82.03340586585229, 41.41100531879138, 1.0],
                [81.86145107737659, 82.03340586585229, 0.10498601801243446, 1.0],
            ],
            [
                [83.88252561332632, 82.31560610468546, 41.41100531879138, 1.0],
                [83.88252561332632, 82.31560610468546, 0.10498601801243446, 1.0],
            ],
            [
                [85.80705833587436, 82.93992628454203, 41.41100531879138, 1.0],
                [85.80705833587436, 82.93992628454203, 0.10498601801243446, 1.0],
            ],
            [
                [87.5781530436727, 83.88155050730022, 41.41100531879138, 1.0],
                [87.5781530436727, 83.88155050730022, 0.10498601801243446, 1.0],
            ],
            [
                [89.14456686168843, 85.10638489064266, 41.41100531879138, 1.0],
                [89.14456686168843, 85.10638489064266, 0.10498601801243446, 1.0],
            ],
            [
                [90.46218911327622, 86.57226009484354, 41.41100531879138, 1.0],
                [90.46218911327622, 86.57226009484354, 0.10498601801243446, 1.0],
            ],
            [
                [91.49527574074764, 88.23036312236076, 41.41100531879138, 1.0],
                [91.49527574074764, 88.23036312236076, 0.10498601801243446, 1.0],
            ],
            [
                [92.21740498369505, 90.02685031269195, 41.41100531879138, 1.0],
                [92.21740498369505, 90.02685031269195, 0.10498601801243446, 1.0],
            ],
            [
                [92.61212923900459, 91.90458857744179, 41.41100531879138, 1.0],
                [92.61212923900459, 91.90458857744179, 0.10498601801243446, 1.0],
            ],
            [
                [92.67330788271605, 93.80496877191614, 41.41100531879138, 1.0],
                [92.67330788271605, 93.80496877191614, 0.10498601801243446, 1.0],
            ],
            [
                [92.40511601319636, 95.66973375275204, 41.41100531879138, 1.0],
                [92.40511601319636, 95.66973375275204, 0.10498601801243446, 1.0],
            ],
            [
                [91.82173425344422, 97.44276414135732, 41.41100531879138, 1.0],
                [91.82173425344422, 97.44276414135732, 0.10498601801243446, 1.0],
            ],
            [
                [90.94673460840232, 99.07176705813723, 41.41100531879138, 1.0],
                [90.94673460840232, 99.07176705813723, 0.10498601801243446, 1.0],
            ],
            [
                [89.8121866063079, 100.50981701522677, 41.41100531879138, 1.0],
                [89.8121866063079, 100.50981701522677, 0.10498601801243446, 1.0],
            ],
            [
                [88.45751628087959, 101.71670360693977, 41.41100531879138, 1.0],
                [88.45751628087959, 101.71670360693977, 0.10498601801243446, 1.0],
            ],
            [
                [86.92815772569035, 102.66004742257813, 41.41100531879138, 1.0],
                [86.92815772569035, 102.66004742257813, 0.10498601801243446, 1.0],
            ],
            [
                [85.27404276559884, 103.31615349165962, 41.41100531879138, 1.0],
                [85.27404276559884, 103.31615349165962, 0.10498601801243446, 1.0],
            ],
            [
                [83.54797858082817, 103.67058029179904, 41.41100531879138, 1.0],
                [83.54797858082817, 103.67058029179904, 0.10498601801243446, 1.0],
            ],
            [
                [81.80396577590322, 103.71841161682002, 41.41100531879138, 1.0],
                [81.80396577590322, 103.71841161682002, 0.10498601801243446, 1.0],
            ],
            [
                [80.09551035018126, 103.46422811661378, 41.41100531879138, 1.0],
                [80.09551035018126, 103.46422811661378, 0.10498601801243446, 1.0],
            ],
            [
                [78.4739822955186, 102.92178477696604, 41.41100531879138, 1.0],
                [78.4739822955186, 102.92178477696604, 0.10498601801243446, 1.0],
            ],
            [
                [76.98707116975723, 102.1134097096405, 41.41100531879138, 1.0],
                [76.98707116975723, 102.1134097096405, 0.10498601801243446, 1.0],
            ],
            [
                [75.67738507359383, 101.06914808879408, 41.41100531879138, 1.0],
                [75.67738507359383, 101.06914808879408, 0.10498601801243446, 1.0],
            ],
            [
                [74.5812341417556, 99.82568264213833, 41.41100531879138, 1.0],
                [74.5812341417556, 99.82568264213833, 0.10498601801243446, 1.0],
            ],
            [
                [73.72763313795036, 98.42506855927712, 41.41100531879138, 1.0],
                [73.72763313795036, 98.42506855927712, 0.10498601801243446, 1.0],
            ],
            [
                [73.13755024273476, 96.91332582942516, 41.41100531879138, 1.0],
                [73.13755024273476, 96.91332582942516, 0.10498601801243446, 1.0],
            ],
            [
                [72.82342089776546, 95.33893572463373, 41.41100531879138, 1.0],
                [72.82342089776546, 95.33893572463373, 0.10498601801243446, 1.0],
            ],
            [
                [72.78893689143499, 93.7512903092582, 41.41100531879138, 1.0],
                [72.78893689143499, 93.7512903092582, 0.10498601801243446, 1.0],
            ],
            [
                [73.02911202232774, 92.19914443865015, 41.41100531879138, 1.0],
                [73.02911202232774, 92.19914443865015, 0.10498601801243446, 1.0],
            ],
            [
                [73.53061694187109, 90.72911871793013, 41.41100531879138, 1.0],
                [73.53061694187109, 90.72911871793013, 0.10498601801243446, 1.0],
            ],
            [
                [74.27236743148026, 89.38429938318734, 41.41100531879138, 1.0],
                [74.27236743148026, 89.38429938318734, 0.10498601801243446, 1.0],
            ],
            [
                [75.22634267107871, 88.20297714795004, 41.41100531879138, 1.0],
                [75.22634267107871, 88.20297714795004, 0.10498601801243446, 1.0],
            ],
            [
                [76.35860323896188, 87.21756187598662, 41.41100531879138, 1.0],
                [76.35860323896188, 87.21756187598662, 0.10498601801243446, 1.0],
            ],
            [
                [77.63047284949508, 86.45370368401446, 41.41100531879138, 1.0],
                [77.63047284949508, 86.45370368401446, 0.10498601801243446, 1.0],
            ],
            [
                [78.99984334910746, 85.92964396266473, 41.41100531879138, 1.0],
                [78.99984334910746, 85.92964396266473, 0.10498601801243446, 1.0],
            ],
            [
                [80.42255937391965, 85.65581207286556, 41.41100531879138, 1.0],
                [80.42255937391965, 85.65581207286556, 0.10498601801243446, 1.0],
            ],
            [
                [81.85383739974577, 85.63467538522558, 41.41100531879138, 1.0],
                [81.85383739974577, 85.63467538522558, 0.10498601801243446, 1.0],
            ],
            [
                [83.24967371523994, 85.86084214680488, 41.41100531879138, 1.0],
                [83.24967371523994, 85.86084214680488, 0.10498601801243446, 1.0],
            ],
            [
                [84.56819710201722, 86.32140864624378, 41.41100531879138, 1.0],
                [84.56819710201722, 86.32140864624378, 0.10498601801243446, 1.0],
            ],
            [
                [85.77092464574159, 86.99653455813666, 41.41100531879138, 1.0],
                [85.77092464574159, 86.99653455813666, 0.10498601801243446, 1.0],
            ],
            [
                [86.8238830200527, 87.86022341648709, 41.41100531879138, 1.0],
                [86.8238830200527, 87.86022341648709, 0.10498601801243446, 1.0],
            ],
            [
                [87.69856263214132, 88.8812791055977, 41.41100531879138, 1.0],
                [87.69856263214132, 88.8812791055977, 0.10498601801243446, 1.0],
            ],
            [
                [88.37267801228043, 90.02440424380293, 41.41100531879138, 1.0],
                [88.37267801228043, 90.02440424380293, 0.10498601801243446, 1.0],
            ],
            [
                [88.83071455976423, 91.2514025131756, 41.41100531879138, 1.0],
                [88.83071455976423, 91.2514025131756, 0.10498601801243446, 1.0],
            ],
            [
                [89.0642489943933, 92.52244445800866, 41.41100531879138, 1.0],
                [89.0642489943933, 92.52244445800866, 0.10498601801243446, 1.0],
            ],
            [
                [89.07203836334277, 93.79735509428534, 41.41100531879138, 1.0],
                [89.07203836334277, 93.79735509428534, 0.10498601801243446, 1.0],
            ],
            [
                [88.85987997107694, 95.03688185466562, 41.41100531879138, 1.0],
                [88.85987997107694, 95.03688185466562, 0.10498601801243446, 1.0],
            ],
            [
                [88.4402518917424, 96.2039029075002, 41.41100531879138, 1.0],
                [88.4402518917424, 96.2039029075002, 0.10498601801243446, 1.0],
            ],
            [
                [87.831750557566, 97.2645386602061, 41.41100531879138, 1.0],
                [87.831750557566, 97.2645386602061, 0.10498601801243446, 1.0],
            ],
            [
                [87.0583480804631, 98.18913317359093, 41.41100531879138, 1.0],
                [87.0583480804631, 98.18913317359093, 0.10498601801243446, 1.0],
            ],
            [
                [86.1484972701268, 98.9530771258053, 41.41100531879138, 1.0],
                [86.1484972701268, 98.9530771258053, 0.10498601801243446, 1.0],
            ],
            [
                [85.13411660424308, 99.53744969410937, 41.41100531879138, 1.0],
                [85.13411660424308, 99.53744969410937, 0.10498601801243446, 1.0],
            ],
            [
                [84.04949056513439, 99.9294630677344, 41.41100531879138, 1.0],
                [84.04949056513439, 99.9294630677344, 0.10498601801243446, 1.0],
            ],
            [
                [82.93012270018926, 100.12270004716686, 41.41100531879138, 1.0],
                [82.93012270018926, 100.12270004716686, 0.10498601801243446, 1.0],
            ],
            [
                [81.81157945380292, 100.11714209752466, 41.41100531879138, 1.0],
                [81.81157945380292, 100.11714209752466, 0.10498601801243446, 1.0],
            ],
            [
                [80.72836224726413, 99.91899207420349, 41.41100531879138, 1.0],
                [80.72836224726413, 99.91899207420349, 0.10498601801243446, 1.0],
            ],
            [
                [79.71284353312089, 99.54030241634977, 41.41100531879138, 1.0],
                [79.71284353312089, 99.54030241634977, 0.10498601801243446, 1.0],
            ],
            [
                [78.79429955371126, 98.99842565475291, 41.41100531879138, 1.0],
                [78.79429955371126, 98.99842565475291, 0.10498601801243446, 1.0],
            ],
            [
                [77.99806896739302, 98.31530957806882, 41.41100531879138, 1.0],
                [77.99806896739302, 98.31530957806882, 0.10498601801243446, 1.0],
            ],
            [
                [77.34486042821379, 97.51666357495867, 41.41100531879138, 1.0],
                [77.34486042821379, 97.51666357495867, 0.10498601801243446, 1.0],
            ],
            [
                [76.85023159296097, 96.63102764841777, 41.41100531879138, 1.0],
                [76.85023159296097, 96.63102764841777, 0.10498601801243446, 1.0],
            ],
            [
                [76.52423795516884, 95.6887728430358, 41.41100531879138, 1.0],
                [76.52423795516884, 95.6887728430358, 0.10498601801243446, 1.0],
            ],
            [
                [76.37131126182027, 94.72108277710663, 41.41100531879138, 1.0],
                [76.37131126182027, 94.72108277710663, 0.10498601801243446, 1.0],
            ],
            [
                [76.38390439702991, 94.07962526877671, 41.41100531879138, 1.0],
                [76.38390439702991, 94.07962526877671, 0.10498601801243446, 1.0],
            ],
            [
                [76.41774580834283, 93.76478202139324, 41.41100531879138, 1.0],
                [76.41774580834283, 93.76478202139324, 0.10498601801243446, 1.0],
            ],
        ],
        dtype=float,
    ),

    degree=(3, 1)

)

curve = NURBSCurve(np.array([[82.590441764984575, 93.433423150668617, 11.677832532680595],
                             [74.593468409418904, 107.28458730928287, 11.677832532680595],
                             [67.794196489838100, 107.28458730928287, 11.677832532680595],
                             [66.055171926376289, 100.79445928283329, 11.677832532680595],
                             [79.915180863183949, 94.044880918038430, 11.677832532680595],
                             [80.435717914257950, 90.849708670366539, 11.677832532680595],
                             [77.964498783262869, 89.422949639859965, 11.677832532680595],
                             [71.405593505923505, 93.209735367321116, 11.677832532680595],
                             [64.604393131427372, 97.136410234349071, 11.677832532680595],
                             [60.529016298654042, 90.077650499996565, 11.677832532680595],
                             [61.852706508449124, 85.137571383559802, 11.677832532680595],
                             [65.517514250421826, 86.119553658436743, 11.677832532680595],
                             [81.279475115702652, 88.787741522939271, 11.677832532680595],
                             [82.966008476892654, 86.544302022017760, 11.677832532680595],
                             [74.432499273102437, 87.204450993082304, 11.677832532680595],
                             [72.507462324364056, 80.052285970560419, 11.677832532680595]]
                            ), degree=3)


from mmcore.numeric.intersection.csx import nurbs_csx
tolerance=1e-6
import time
s=time.perf_counter_ns()
result=nurbs_csx(curve,surface,tolerance,1e-10)
e=time.perf_counter_ns()-s


# CHECK

# To verify the robustness of this the implementation, let's check the distance between
# the point estimated on the curve and on the surface in the intersection parameters.

for typ,pt ,(t,u,v)in result:

    pt1=surface.evaluate_v2(u, v) # evaluate point on surface
    pt2=curve.evaluate(t) # evaluate point on curve
    print(pt1,pt2)
    dist=np.linalg.norm(surface.evaluate_v2(u, v) - curve.evaluate(t)) # Must be less than the tolerance
    print(f'error: {dist} (must be less than {tolerance})')

    assert dist<tolerance # If dist>=tolerance an AssertionError will be raised

print(f"CSX performed at: {e*1e-9} secs.")