#
from mmcore.base.sharedstate import serve
points_storage = []
@serve.app.post('/cpts')
def cptss(data:dict):
    points_storage.append(list(data['data'].values()))
    return {"msg":'OK'}
@serve.app.get('/cpts/clear')
def cptss_clear():
    points_storage.clear()
    return {"msg":'OK'}

serve.start()
