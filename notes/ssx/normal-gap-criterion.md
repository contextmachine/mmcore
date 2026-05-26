# Normal-angle (or normal-distance) gap

Идея алгоритма уточнения точки пересечения двух NURBS-поверхностей, проста. У вас есть начальное предположение в виде $s,t,u,v$. Вы хотите итеративно уточнить это предположение, и тем самым, найти s,t,u,v точки лежащей на двух поверхностях. 

Поскольку s,t и u,v никак не связаны. Нам необходимо использовать внешние, не связанные с параметрами метрики. Чаще и подробнее всего в литературе описывается метрика на основе пространственного расстояния между S_1(s,t) и S_2(u,v). 

$$
||S_1(s, t)- S_2(u, v)||
$$
Тем не менее этого становится не достаточно в случае когда поверхности очень медленно приближаются друг к другу. Причина проста, расстояние между точками может стать меньше spt вдали от истинной точки пересечения. 
Это отлично видно на примере ниже:

![](../images/Frame%201430102030.png)

## Рабочее решение
Понятно что требовался второй критерий, как-то учитывающий угол между нормалями. Конкретным решением, которое сработало стало следующее:


$$
\sin\theta=\frac{\|\,n_1\times n_2\,\|}{\|n_1\|\|n_2\|}\le\varepsilon_\theta
\quad\text{or}\quad
\|\,n_1\cdot (S_1-S_2)\|\le\varepsilon_n .
$$

Вторая форма (скалярное проецирование вектора $S_1(s, t)-S_2(u, v)$ на нормаль) дешева и работает для почти касательных случаев.

### Результат ДО и После:

![](../images/Frame%201430102031.png)

Обратите внимание на качество кривой на сложном участке:
![](../images/Frame%201430102032.png)
 
### Конкретная реализация 

```python
def calculate_eps_n(spt, angle_tol):
    return (spt**2)/(angle_tol+10e-12)

def normal_angle_gap(n1, n2):
    """
    Compute sin(theta) between normals n1 and n2:
        sin θ = ||n1 × n2|| / (||n1|| · ||n2||)
    """
    num = np.linalg.norm(np.cross(n1, n2))
    den = np.linalg.norm(n1) * np.linalg.norm(n2)
    return num / den

def normal_distance_gap(n, S1, S2):
    """
    Compute the scalar projection of the residual (S1 - S2) on normal n:
        |n · (S1 - S2)| / ||n||
    If n is already unit-length you can omit the division by ||n||.
    """
 
    diff = S1- S2
    return abs(np.dot(n, diff))

def within_normal_gap(n1, n2, S1, S2, eps_theta, eps_n):
    """
    Check whether either gap metrics is below its threshold:
      sin θ ≤ eps_theta
      OR
      |n1·(S1−S2)| ≤ eps_n
    """
    if normal_angle_gap(n1, n2) <= eps_theta:
        return True
    if normal_distance_gap(n1, S1, S2) <= eps_n:
        return True
    return False

```