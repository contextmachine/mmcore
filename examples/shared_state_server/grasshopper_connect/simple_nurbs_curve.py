# run this example in python console

"""
-----------
EN
-----------
First of all:
1. Close the grasshopper
2. Run ipython/PythonConsole/jupyter/... In an environment with mmcore installed.
3. Run SharedStateServer there using the lines below:

>>> from mmcore.base.sharedstate import serve
... serve.start()

4. Run component (first time it should return null), passing in x list of points .
5. Return to adult python.
6. Create a handler. For example:

>>> from mmcore.geom.parametric.nurbs import NurbsCurve
>>> def nurbs_resolver(**kws):
...     n=NurbsCurve(kws.get("x"))
...     return n.evaluate(np.linspace(0,1,100)).tolist()

7. Connect the handler to the server:
>>> ID = "nurbs_resolver"
>>> serve.add_resolver(ID,nurbs_resolver)
Or use the decorator, in which case the ID will be identical to the function name:

>>> @serve.resolver
... def nurbs_resolver(**kws):
...     n=NurbsCurve(kws.get(x))
...     return n.evaluate(np.linspace(0,1,100)).tolist()

* ID can be any string. The main thing is that the ID of the handler
and the value of the ID variable in the component you want to use are the same.
Below I use ghenv.Component.ComponentGuid (Guid of the current component ) as ID.
The ID is also displayed in a message below the component.
For simplicity, this component prints out the code of the command you need.
Just replace <resolver> with the function you specify.

8. Open the grasshopper
9. Open and run the file mmcl.py in this directory with the ScriptEditor component

Translated with www.DeepL.com/Translator (free version)
-----------
RU
-----------
Прежде всего:
1. Закройте грассхоппер
2. Запустите ipython/PythonConsole/jupyter/... в окружении с установленным mmcore.
3. Запустите там SharedStateServer с помощью строк ниже:

>>> from mmcore.base.sharedstate import serve
... serve.start()

4. Выполните компонент (первый раз он должен вернуть null), передав в x список точек .
5. Вернитесь во взрослый python.
6. Создайте обработчик. Например:

>>> from mmcore.geom.parametric.nurbs import NurbsCurve
>>> def nurbs_resolver(**kws):
...     n=NurbsCurve(kws.get("x"))
...     return n.evaluate(np.linspace(0,1,100)).tolist()

7. Подключите обработчик к серверу:
>>> ID = "nurbs_resolver"
>>> serve.add_resolver(ID,nurbs_resolver)
Или используйте декоратор, в этом случае ID будет идентично имени функции:
>>> @serve.resolver
... def nurbs_resolver(**kws):
...     n=NurbsCurve(kws.get(x))
...     return n.evaluate(np.linspace(0,1,100)).tolist()

* ID может быть любой строкой. Главное чтобы ID обработчика
и значение переменной ID в нужном вам компоненте совпадали.
Ниже я использую в качестве ID ghenv.Component.ComponentGuid (Guid текущего компонента ).
ID также отображается в сообщении под компонентом.
Для упрощения, этот компонент печатает в out код необходимой вам команды.
Просто замените <resolver> на определенную вами функцию.

8. Oткройте грассхоппер
9. Откройте и запустите файл mmcl.py из этой директории с помощью компонента ScriptEditor
"""

from mmcore.geom.parametric.nurbs import NurbsCurve
from mmcore.base.sharedstate import serve
import numpy as np

serve.start()  # Server run in non-blocked thread on http://localhost:7711


# It dynamically updates when you will create a new objects or change exist

@serve.resolver
def nurbs_resolver(**kws):
    n = NurbsCurve(kws.get('x'))
    return n.evaluate(np.linspace(0, 1, 100)).tolist()
