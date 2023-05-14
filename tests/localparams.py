ABCD = [[-220175.307469, -38456.999234, 20521],
              [-211734.667469, -9397.999234, 13016.199506],
              [-171710.667469, -8217.999234, 5829],
              [-152984.00562, -28444.1358, 6172.86857]]

import numpy as np
arrABCD=np.array(ABCD)
a1=np.mean(arrABCD,axis=-1)
a2=np.mean(arrABCD,axis=0)
A,B,C,D=((arrABCD-a2)*1e-3).tolist()