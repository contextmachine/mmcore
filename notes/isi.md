

Чтобы найти обратную матрицу Якобиана $ J $, нужно вычислить обратную матрицу $ 2 \times 2 $. Дана матрица якобиана:

$$ J = \begin{pmatrix}
\nabla f_1 \cdot \nabla f_1(q_k + \alpha \nabla f_1 + \beta \nabla f_2) & \nabla f_1 \cdot \nabla f_2(q_k + \alpha \nabla f_1 + \beta \nabla f_2) \\\\
\nabla f_2 \cdot \nabla f_1(q_k + \alpha \nabla f_1 + \beta \nabla f_2) & \nabla f_2 \cdot \nabla f_2(q_k + \alpha \nabla f_1 + \beta \nabla f_2)
\end{pmatrix} $$

Обратная $ 2 \times 2 $ матрица $ A = \begin{pmatrix} a & b \\\\ c & d \end{pmatrix} $ дается:

$$ A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\\ -c & a \end{pmatrix} $$

Применим это к нашей матрице Якобиана $ J $:

1. Пусть:
   $$
   a = \nabla f_1 \cdot \nabla f_1(q_k + \alpha \nabla f_1 + \beta \nabla f_2)
   $$
   $$
   b = \nabla f_1 \cdot \nabla f_2(q_k + \alpha \nabla f_1 + \beta \nabla f_2)
   $$
   $$
   c = \nabla f_2 \cdot \nabla f_1(q_k + \alpha \nabla f_1 + \beta \nabla f_2)
   $$
   $$
   d = \nabla f_2 \cdot \nabla f_2(q_k + \alpha \nabla f_1 + \beta \nabla f_2)
   $$

2. Определитель $ J $ равен:
   $$
   \text{det}(J) = ad - bc
   $$

3. Обратным показателем $ J $ является:
   $$
   J^{-1} = \frac{1}{ad - bc} \begin{pmatrix}
   d & -b \\\\
   -c & a
   \end{pmatrix}
   $$

Итак, с точки зрения градиентов в точке $ q_k + \alpha \nabla f_1 + \beta \nabla f_2 $, обратная величина $ J $ будет:

$$ J^{-1} = \frac{1}{(\nabla f_1 \cdot \nabla f_1) (\nabla f_2 \cdot \nabla f_2) - (\nabla f_1 \cdot \nabla f_2) (\nabla f_2 \cdot \nabla f_1)} \begin{pmatrix}
\nabla f_2 \cdot \nabla f_2 & -\nabla f_1 \cdot \nabla f_2 \\\\
-\nabla f_2 \cdot \nabla f_1 & \nabla f_1 \cdot \nabla f_1
\end{pmatrix} $$

Здесь все точечные произведения оцениваются в точке $ q_k + \alpha \nabla f_1 + \beta \nabla f_2 $.


$q$ - вектор ($x,y,z$), $f1(q),f2(q)$ функции принимающие вектор и возвращущие скаляр, $α,β$ - скалярные переменные, которые необходимо вычислить
$$
q=x,y,z\\
∆k = α*∇f1(q)+β*∇f2(q)\\
{∆k}\cdot{∇f_1(q)}=−f_1(q)\\
{∆k}\cdot {∇f_2(q)}=−f_2(qk)
$$

Given:
- $ q = (x, y, z) $
- $ f_1(q) $ and $ f_2(q) $ are functions that take a vector $ q $ and return a scalar.
- $ \alpha $ and $ \beta $ are scalar variables to be computed.

The system of equations is:
$$
\Delta k = \alpha \nabla f_1(q) + \beta \nabla f_2(q)
$$
$$
\Delta k \cdot \nabla f_1(q) = -f_1(q)
$$
$$
\Delta k \cdot \nabla f_2(q) = -f_2(q)
$$

Let's denote:
- $\nabla f_1(q) = \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix}$
- $\nabla f_2(q) = \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix}$
- $\Delta k = \begin{pmatrix} \Delta k_x \\ \Delta k_y \\ \Delta k_z \end{pmatrix}$

First, we express $\Delta k$ in terms of $\alpha$ and $\beta$:
$$
\begin{pmatrix} \Delta k_x \\ \Delta k_y \\ \Delta k_z \end{pmatrix} = \alpha \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} + \beta \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix}
$$

Next, we write the dot product equations:
$$
\begin{pmatrix} \Delta k_x \\ \Delta k_y \\ \Delta k_z \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} = -f_1(q)
$$
$$
\begin{pmatrix} \Delta k_x \\ \Delta k_y \\ \Delta k_z \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} = -f_2(q)
$$

Substituting $\Delta k$ from the first equation into the dot products, we get:
$$
\left( \alpha \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} + \beta \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \right) \cdot \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} = -f_1(q)
$$
$$
\left( \alpha \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} + \beta \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \right) \cdot \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} = -f_2(q)
$$

Simplifying the dot products, we get two linear equations in terms of $\alpha$ and $\beta$:
$$
\alpha \left( \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} \right) + \beta \left( \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} \right) = -f_1(q)
$$
$$
\alpha \left( \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \right) + \beta \left( \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \right) = -f_2(q)
$$

Denoting:
$$
A_{11} = \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix}
$$
$$
A_{12} = \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix}
$$
$$
A_{21} = \begin{pmatrix} \frac{\partial f_1}{\partial x} \\ \frac{\partial f_1}{\partial y} \\ \frac{\partial f_1}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix}
$$
$$
A_{22} = \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \\ \frac{\partial f_2}{\partial z} \end{pmatrix}
$$

The equations can be written as:
$$
\alpha A_{11} + \beta A_{12} = -f_1(q)
$$
$$
\alpha A_{21} + \beta A_{22} =
 -f_2(q)
$$

In matrix form, this is:
$$
\begin{pmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{pmatrix}
\begin{pmatrix}
\alpha \\
\beta
\end{pmatrix}
=
\begin{pmatrix}
-f_1(q) \\
-f_2(q)
\end{pmatrix}
$$

