"""

Прежде всего:
1. Закройте грассхоппер
2. Запустите ipython/PythonConsole/jupyter/... в окружении с установленным mmcore.
3. Запустите там SharedStateServer с помощью строк ниже:

from mmcore.base.sharedstate import serve
serve.start()

4. Выполните компонент (первый раз он должен вернуть null), передав в x список точек .
5. Вернитесь во взрослый python.
6. Создайте обработчик. Например:

from mmcore.geom.parametric.nurbs import NurbsCurve

def nurbs_resolver(**kws):
    n=NurbsCurve(kws.get("x"))
    return n.evaluate(np.linspace(0,1,100)).tolist()

7. Подключите обработчик к серверу:

serve.add_resolver(ID)=nurbs_resolver
Или используйте декоратор, в этом случае ID будет идентично имени функции:

@serve.resolver
def nurbs_resolver(**kws):
    n=NurbsCurve(kws.get(x))
    return n.evaluate(np.linspace(0,1,100)).tolist()

* ID может быть любой строкой. Главное чтобы ID обработчика
и значение переменной ID в нужном вам компоненте совпадали.
Ниже я использую в качестве ID ghenv.Component.ComponentGuid (Guid текущего компонента ).
ID также отображается в сообщении под компонентом.
Для упрощения, этот компонент печатает в out код необходимой вам команды.
Просто замените <resolver> на определенную вами функцию.
"""

# ! python 3

#ID = str(ghenv.Component.ComponentGuid)
ID = "nurbs_resolver"
ghenv.Component.set_Message("id: " + ID)
ghenv.Component.set_Name("mmcore example client")
ghenv.Component.set_NickName("mmcl")
ghenv.Component.set_Description("This component allows direct access to the running mmcore instance. " \
                                "You can send, return and execute any data in real time by communicating via IPython, PycharmConsole, Jupyter etc. ")
print(f'Команда добавления обработчика в ipython, PythonConsole, jupyter :\n\nserve.add_resolver({ID}, <resolver>)')

import Rhino.Geometry as rg
import ghpythonlib.treehelpers as th
import requests
import json


def pt(pts):
    return pts.X, pts.Y, pts.Z


def pts(rpts):
    for rpt in rpts:
        yield pt(rpt)


ns = dict()

resp = requests.post(f"http://localhost:7711/resolver/{ID}",
                     json={
                        "x": list(pts(x))
                        }
                     )
ans = resp.json()
if ans:
    a = th.list_to_tree(resp.json())