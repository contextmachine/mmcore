# GroupFabric

![image_3.png](image_3.png)

```python
from mmcore.geom.vec import *
from mmcore.geom.box import Box
from mmcore.common.viewer import DefaultGroupFabric

vecs = unit(np.random.random((2, 4, 3)))

boxes = [Box(10, 20, 10), Box(5, 5, 5), Box(15, 5, 5), Box(25, 20, 2)]
for i in range(4):
    boxes[i].xaxis = vecs[0, i, :]
    boxes[i].origin = vecs[1, i, :] * np.random.randint(0, 20)
    boxes[i].refine(('y', 'z'))
from mmcore.common.viewer import DefaultGroupFabric

group = DefaultGroupFabric([bx.to_mesh() for bx in boxes], uuid='fabric-group')

```

Запускаем `serve`

``` python
from mmcore.base.sharedstate import serve
serve.start()
```

```
    +×  mmcore <version>
        server: uvicorn
        app: mmcore.base.sharedstate:serve_app
        local: http://localhost:7711/
        openapi ui: http://localhost:7711/docs
    
    
```

Сейчас по рест запросу `http://localhost:7711/fetch/fabric-group` в сцене, вы получите:

![image.png](image.png)

Вы можете менять любые значения и видеть результат во вьювере и в консоле

![image_2.png](image_2.png)