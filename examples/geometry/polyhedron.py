from mmcore.geom.shapes.base import PolyHedron

# points ordered by faces
points = [[[90.29389953613281, 188.57188415527344, 90.29389953613281],
           [144.94248962402344, 114.26821899414062, 145.0714111328125],
           [193.2566375732422, 114.26821899414062, 0.1289348155260086]],
          [[90.29389953613281, 188.57188415527344, 90.29389953613281],
           [193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [120.7938461303711, 206.7416534423828, 0.0]],
          [[144.94248962402344, 114.26821899414062, 145.0714111328125],
           [152.19894409179688, 20.307313919067383, 152.327880859375],
           [210.79168701171875, 0.4763064980506897, 0.12893584370613098]],
          [[144.94248962402344, 114.26821899414062, 145.0714111328125],
           [210.79168701171875, 0.4763064980506897, 0.12893584370613098],
           [193.2566375732422, 114.26821899414062, 0.1289348155260086]],
          [[193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [210.79168701171875, 0.4763064980506897, 0.12893584370613098],
           [158.09376525878906, 0.4763064980506897, -163.72080993652344]],
          [[193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [158.09376525878906, 0.4763064980506897, -163.72080993652344],
           [144.94248962402344, 114.26821899414062, -144.8135528564453]],
          [[120.7938461303711, 206.7416534423828, 0.0],
           [193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [144.94248962402344, 114.26821899414062, -144.8135528564453]],
          [[120.7938461303711, 206.7416534423828, 0.0],
           [144.94248962402344, 114.26821899414062, -144.8135528564453],
           [90.29389953613281, 188.57188415527344, -90.29389953613281]],
          [[-90.29389953613281, 188.57188415527344, -90.29389953613281],
           [-144.94248962402344, 114.26821899414062, -144.8135528564453],
           [-193.2566375732422, 114.26821899414062, 0.1289348155260086]],
          [[-90.29389953613281, 188.57188415527344, -90.29389953613281],
           [-193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [-120.7938461303711, 206.7416534423828, 0.0]],
          [[-144.94248962402344, 114.26821899414062, -144.8135528564453],
           [-152.19894409179688, 0.4763064980506897, -163.72080993652344],
           [-210.79168701171875, 0.4763064980506897, 0.12893380224704742]],
          [[-144.94248962402344, 114.26821899414062, -144.8135528564453],
           [-210.79168701171875, 0.4763064980506897, 0.12893380224704742],
           [-193.2566375732422, 114.26821899414062, 0.1289348155260086]],
          [[-193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [-210.79168701171875, 0.4763064980506897, 0.12893380224704742],
           [-158.09376525878906, 20.307313919067383, 152.327880859375]],
          [[-193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [-158.09376525878906, 20.307313919067383, 152.327880859375],
           [-144.94248962402344, 114.26821899414062, 145.0714111328125]],
          [[-120.7938461303711, 206.7416534423828, 0.0],
           [-193.2566375732422, 114.26821899414062, 0.1289348155260086],
           [-144.94248962402344, 114.26821899414062, 145.0714111328125]],
          [[-120.7938461303711, 206.7416534423828, 0.0],
           [-144.94248962402344, 114.26821899414062, 145.0714111328125],
           [-90.29389953613281, 188.57188415527344, 90.29389953613281]],
          [[90.29389953613281, 188.57188415527344, -90.29389953613281],
           [0.0, 206.74163818359375, -120.7938461303711],
           [0.0, 230.10281372070312, 0.0]],
          [[90.29389953613281, 188.57188415527344, -90.29389953613281],
           [0.0, 230.10281372070312, 0.0],
           [120.7938461303711, 206.7416534423828, 0.0]],
          [[0.0, 206.74163818359375, -120.7938461303711],
           [-90.29389953613281, 188.57188415527344, -90.29389953613281],
           [-120.7938461303711, 206.7416534423828, 0.0]],
          [[0.0, 206.74163818359375, -120.7938461303711],
           [-120.7938461303711, 206.7416534423828, 0.0],
           [0.0, 230.10281372070312, 0.0]],
          [[0.0, 230.10281372070312, 0.0],
           [-120.7938461303711, 206.7416534423828, 0.0],
           [-90.29389953613281, 188.57188415527344, 90.29389953613281]],
          [[0.0, 230.10281372070312, 0.0],
           [-90.29389953613281, 188.57188415527344, 90.29389953613281],
           [0.0, 206.74163818359375, 120.7938461303711]],
          [[120.7938461303711, 206.7416534423828, 0.0],
           [0.0, 230.10281372070312, 0.0],
           [0.0, 206.74163818359375, 120.7938461303711]],
          [[120.7938461303711, 206.7416534423828, 0.0],
           [0.0, 206.74163818359375, 120.7938461303711],
           [90.29389953613281, 188.57188415527344, 90.29389953613281]],
          [[118.52995300292969, -183.03985595703125, 118.52995300292969],
           [-9.5367431640625e-07, -203.09744262695312, 163.64630126953125],
           [0.0, -228.88580322265625, 0.0]],
          [[118.52995300292969, -183.03985595703125, 118.52995300292969],
           [0.0, -228.88580322265625, 0.0],
           [163.64630126953125, -203.09744262695312, 9.5367431640625e-07]],
          [[-9.5367431640625e-07, -203.09744262695312, 163.64630126953125],
           [-118.52995300292969, -183.03985595703125, 118.52994537353516],
           [-163.64630126953125, -203.09744262695312, -9.5367431640625e-07]],
          [[-9.5367431640625e-07, -203.09744262695312, 163.64630126953125],
           [-163.64630126953125, -203.09744262695312, -9.5367431640625e-07],
           [0.0, -228.88580322265625, 0.0]],
          [[0.0, -228.88580322265625, 0.0],
           [-163.64630126953125, -203.09744262695312, -9.5367431640625e-07],
           [-118.52994537353516, -183.03985595703125, -118.52995300292969]],
          [[0.0, -228.88580322265625, 0.0],
           [-118.52994537353516, -183.03985595703125, -118.52995300292969],
           [9.5367431640625e-07, -203.09744262695312, -163.64630126953125]],
          [[163.64630126953125, -203.09744262695312, 9.5367431640625e-07],
           [0.0, -228.88580322265625, 0.0],
           [9.5367431640625e-07, -203.09744262695312, -163.64630126953125]],
          [[163.64630126953125, -203.09744262695312, 9.5367431640625e-07],
           [9.5367431640625e-07, -203.09744262695312, -163.64630126953125],
           [118.52995300292969, -183.03985595703125, -118.52994537353516]],
          [[-90.29389953613281, 188.57188415527344, 90.29389953613281],
           [-144.94248962402344, 114.26821899414062, 145.0714111328125],
           [0.0, 114.26821899414062, 193.3855743408203]],
          [[-90.29389953613281, 188.57188415527344, 90.29389953613281],
           [0.0, 114.26821899414062, 193.3855743408203],
           [0.0, 206.74163818359375, 120.7938461303711]],
          [[-144.94248962402344, 114.26821899414062, 145.0714111328125],
           [-158.09376525878906, 20.307313919067383, 152.327880859375],
           [-1.0134924650628818e-06, 20.307313919067383, 203.06085205078125]],
          [[-144.94248962402344, 114.26821899414062, 145.0714111328125],
           [-1.0134924650628818e-06, 20.307313919067383, 203.06085205078125],
           [0.0, 114.26821899414062, 193.3855743408203]],
          [[0.0, 114.26821899414062, 193.3855743408203],
           [-1.0134924650628818e-06, 20.307313919067383, 203.06085205078125],
           [152.19894409179688, 20.307313919067383, 152.327880859375]],
          [[0.0, 114.26821899414062, 193.3855743408203],
           [152.19894409179688, 20.307313919067383, 152.327880859375],
           [144.94248962402344, 114.26821899414062, 145.0714111328125]],
          [[0.0, 206.74163818359375, 120.7938461303711],
           [0.0, 114.26821899414062, 193.3855743408203],
           [144.94248962402344, 114.26821899414062, 145.0714111328125]],
          [[0.0, 206.74163818359375, 120.7938461303711],
           [144.94248962402344, 114.26821899414062, 145.0714111328125],
           [90.29389953613281, 188.57188415527344, 90.29389953613281]],
          [[90.29389953613281, 188.57188415527344, -90.29389953613281],
           [144.94248962402344, 114.26821899414062, -144.8135528564453],
           [0.0, 114.26821899414062, -193.12770080566406]],
          [[90.29389953613281, 188.57188415527344, -90.29389953613281],
           [0.0, 114.26821899414062, -193.12770080566406],
           [0.0, 206.74163818359375, -120.7938461303711]],
          [[144.94248962402344, 114.26821899414062, -144.8135528564453],
           [158.09376525878906, 0.4763064980506897, -163.72080993652344],
           [1.0134924650628818e-06, 0.4763064980506897, -214.45379638671875]],
          [[144.94248962402344, 114.26821899414062, -144.8135528564453],
           [1.0134924650628818e-06, 0.4763064980506897, -214.45379638671875],
           [0.0, 114.26821899414062, -193.12770080566406]],
          [[0.0, 114.26821899414062, -193.12770080566406],
           [1.0134924650628818e-06, 0.4763064980506897, -214.45379638671875],
           [-152.19894409179688, 0.4763064980506897, -163.72080993652344]],
          [[0.0, 114.26821899414062, -193.12770080566406],
           [-152.19894409179688, 0.4763064980506897, -163.72080993652344],
           [-144.94248962402344, 114.26821899414062, -144.8135528564453]],
          [[0.0, 206.74163818359375, -120.7938461303711],
           [0.0, 114.26821899414062, -193.12770080566406],
           [-144.94248962402344, 114.26821899414062, -144.8135528564453]],
          [[0.0, 206.74163818359375, -120.7938461303711],
           [-144.94248962402344, 114.26821899414062, -144.8135528564453],
           [-90.29389953613281, 188.57188415527344, -90.29389953613281]],
          [[118.52995300292969, -183.03985595703125, -118.52994537353516],
           [154.56912231445312, -92.735107421875, -154.56912231445312],
           [206.0921630859375, -92.735107421875, 3.814697265625e-06]],
          [[118.52995300292969, -183.03985595703125, -118.52994537353516],
           [206.0921630859375, -92.735107421875, 3.814697265625e-06],
           [163.64630126953125, -203.09744262695312, 9.5367431640625e-07]],
          [[154.56912231445312, -92.735107421875, -154.56912231445312],
           [158.09376525878906, 0.4763064980506897, -163.72080993652344],
           [210.79168701171875, 0.4763064980506897, 0.12893584370613098]],
          [[154.56912231445312, -92.735107421875, -154.56912231445312],
           [210.79168701171875, 0.4763064980506897, 0.12893584370613098],
           [206.0921630859375, -92.735107421875, 3.814697265625e-06]],
          [[206.0921630859375, -92.735107421875, 3.814697265625e-06],
           [210.79168701171875, 0.4763064980506897, 0.12893584370613098],
           [152.19894409179688, 20.307313919067383, 152.327880859375]],
          [[206.0921630859375, -92.735107421875, 3.814697265625e-06],
           [152.19894409179688, 20.307313919067383, 152.327880859375],
           [154.56912231445312, -92.735107421875, 154.56912231445312]],
          [[163.64630126953125, -203.09744262695312, 9.5367431640625e-07],
           [206.0921630859375, -92.735107421875, 3.814697265625e-06],
           [154.56912231445312, -92.735107421875, 154.56912231445312]],
          [[163.64630126953125, -203.09744262695312, 9.5367431640625e-07],
           [154.56912231445312, -92.735107421875, 154.56912231445312],
           [118.52995300292969, -183.03985595703125, 118.52995300292969]],
          [[-118.52994537353516, -183.03985595703125, -118.52995300292969],
           [-154.56912231445312, -92.735107421875, -154.56912231445312],
           [3.814697265625e-06, -92.735107421875, -206.0921630859375]],
          [[-118.52994537353516, -183.03985595703125, -118.52995300292969],
           [3.814697265625e-06, -92.735107421875, -206.0921630859375],
           [9.5367431640625e-07, -203.09744262695312, -163.64630126953125]],
          [[-154.56912231445312, -92.735107421875, -154.56912231445312],
           [-152.19894409179688, 0.4763064980506897, -163.72080993652344],
           [1.0134924650628818e-06, 0.4763064980506897, -214.45379638671875]],
          [[-154.56912231445312, -92.735107421875, -154.56912231445312],
           [1.0134924650628818e-06, 0.4763064980506897, -214.45379638671875],
           [3.814697265625e-06, -92.735107421875, -206.0921630859375]],
          [[3.814697265625e-06, -92.735107421875, -206.0921630859375],
           [1.0134924650628818e-06, 0.4763064980506897, -214.45379638671875],
           [158.09376525878906, 0.4763064980506897, -163.72080993652344]],
          [[3.814697265625e-06, -92.735107421875, -206.0921630859375],
           [158.09376525878906, 0.4763064980506897, -163.72080993652344],
           [154.56912231445312, -92.735107421875, -154.56912231445312]],
          [[9.5367431640625e-07, -203.09744262695312, -163.64630126953125],
           [3.814697265625e-06, -92.735107421875, -206.0921630859375],
           [154.56912231445312, -92.735107421875, -154.56912231445312]],
          [[9.5367431640625e-07, -203.09744262695312, -163.64630126953125],
           [154.56912231445312, -92.735107421875, -154.56912231445312],
           [118.52995300292969, -183.03985595703125, -118.52994537353516]],
          [[-118.52995300292969, -183.03985595703125, 118.52994537353516],
           [-154.56912231445312, -92.735107421875, 154.56912231445312],
           [-206.0921630859375, -92.735107421875, -3.814697265625e-06]],
          [[-118.52995300292969, -183.03985595703125, 118.52994537353516],
           [-206.0921630859375, -92.735107421875, -3.814697265625e-06],
           [-163.64630126953125, -203.09744262695312, -9.5367431640625e-07]],
          [[-154.56912231445312, -92.735107421875, 154.56912231445312],
           [-158.09376525878906, 20.307313919067383, 152.327880859375],
           [-210.79168701171875, 0.4763064980506897, 0.12893380224704742]],
          [[-154.56912231445312, -92.735107421875, 154.56912231445312],
           [-210.79168701171875, 0.4763064980506897, 0.12893380224704742],
           [-206.0921630859375, -92.735107421875, -3.814697265625e-06]],
          [[-206.0921630859375, -92.735107421875, -3.814697265625e-06],
           [-210.79168701171875, 0.4763064980506897, 0.12893380224704742],
           [-152.19894409179688, 0.4763064980506897, -163.72080993652344]],
          [[-206.0921630859375, -92.735107421875, -3.814697265625e-06],
           [-152.19894409179688, 0.4763064980506897, -163.72080993652344],
           [-154.56912231445312, -92.735107421875, -154.56912231445312]],
          [[-163.64630126953125, -203.09744262695312, -9.5367431640625e-07],
           [-206.0921630859375, -92.735107421875, -3.814697265625e-06],
           [-154.56912231445312, -92.735107421875, -154.56912231445312]],
          [[-163.64630126953125, -203.09744262695312, -9.5367431640625e-07],
           [-154.56912231445312, -92.735107421875, -154.56912231445312],
           [-118.52994537353516, -183.03985595703125, -118.52995300292969]],
          [[118.52995300292969, -183.03985595703125, 118.52995300292969],
           [154.56912231445312, -92.735107421875, 154.56912231445312],
           [-3.814697265625e-06, -92.735107421875, 206.0921630859375]],
          [[118.52995300292969, -183.03985595703125, 118.52995300292969],
           [-3.814697265625e-06, -92.735107421875, 206.0921630859375],
           [-9.5367431640625e-07, -203.09744262695312, 163.64630126953125]],
          [[154.56912231445312, -92.735107421875, 154.56912231445312],
           [152.19894409179688, 20.307313919067383, 152.327880859375],
           [-1.0134924650628818e-06, 20.307313919067383, 203.06085205078125]],
          [[154.56912231445312, -92.735107421875, 154.56912231445312],
           [-1.0134924650628818e-06, 20.307313919067383, 203.06085205078125],
           [-3.814697265625e-06, -92.735107421875, 206.0921630859375]],
          [[-3.814697265625e-06, -92.735107421875, 206.0921630859375],
           [-1.0134924650628818e-06, 20.307313919067383, 203.06085205078125],
           [-158.09376525878906, 20.307313919067383, 152.327880859375]],
          [[-3.814697265625e-06, -92.735107421875, 206.0921630859375],
           [-158.09376525878906, 20.307313919067383, 152.327880859375],
           [-154.56912231445312, -92.735107421875, 154.56912231445312]],
          [[-9.5367431640625e-07, -203.09744262695312, 163.64630126953125],
           [-3.814697265625e-06, -92.735107421875, 206.0921630859375],
           [-154.56912231445312, -92.735107421875, 154.56912231445312]],
          [[-9.5367431640625e-07, -203.09744262695312, 163.64630126953125],
           [-154.56912231445312, -92.735107421875, 154.56912231445312],
           [-118.52995300292969, -183.03985595703125, 118.52994537353516]]]
polyhedron = PolyHedron(uuid="test_polyhedron")

if __name__ == "__main__":

    for face in points:
        polyhedron.add_shape(face)
    # повторная треангуля

    polyhedron.to_mesh().dump("polyhedron.json")
